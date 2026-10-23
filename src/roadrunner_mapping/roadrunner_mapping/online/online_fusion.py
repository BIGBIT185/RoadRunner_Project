"""ROS1 在线可通行图的无 ROS 纯函数和轻量状态结构。

本模块只处理窗口、坐标和概率图层，不负责订阅消息或启动线程，便于在没有
ROS master 的单元测试中验证地图布局和风险方向。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from roadrunner_mapping.geometry.coordinates import (
    body_xy_to_grid,
    transform_points,
    validate_transform,
)
from roadrunner_mapping.geometry.schema import MapSpec

from .incremental_semantic_grid import (  # noqa: F401 - compatibility export
    IncrementalSemanticGrid,
)


@dataclass(frozen=True)
class CompactSemanticFrame:
    """一帧已投影语义点的紧凑表示，供离线/对照栅格化使用。"""
    timestamp: float
    xyz_world: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    dynamic: np.ndarray

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_world, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.uint8)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        dynamic = np.asarray(self.dynamic, dtype=bool)
        count = xyz.shape[0]
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_world must have shape [N,3]")
        if any(value.shape != (count,) for value in (labels, confidence, valid, dynamic)):
            raise ValueError("compact semantic fields must match the point count")
        if (
            not np.isfinite(self.timestamp)
            or not np.isfinite(xyz).all()
            or not np.isfinite(confidence).all()
        ):
            raise ValueError("semantic frame time and coordinates must be finite")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("semantic confidence must be in [0,1]")
        object.__setattr__(self, "xyz_world", xyz)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "confidence", confidence.astype(np.float16))
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "dynamic", dynamic)


class RecentFrameWindow:
    """按时间递增保存最近几何帧的窗口。

    注意：窗口只负责记录活动帧；非 strict 在线 GPU 地图可以保留更长的
    增量状态，严格清理由调用方决定。
    """

    def __init__(self, duration_sec: float) -> None:
        if not np.isfinite(duration_sec) or duration_sec <= 0.0:
            raise ValueError("duration_sec must be finite and positive")
        self.duration_sec = float(duration_sec)
        self._frames = deque()

    def append(self, frame):
        """追加一帧并返回早于时间窗口的过期帧。"""
        timestamp = float(frame.timestamp)
        if self._frames and timestamp <= float(self._frames[-1].timestamp):
            raise ValueError("online frames must have strictly increasing timestamps")
        self._frames.append(frame)
        cutoff = timestamp - self.duration_sec
        expired = []
        while self._frames and float(self._frames[0].timestamp) < cutoff:
            expired.append(self._frames.popleft())
        return tuple(expired)

    def __iter__(self):
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    @property
    def span_sec(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return float(self._frames[-1].timestamp - self._frames[0].timestamp)


def rasterize_semantic_window(
    frames: Iterable[CompactSemanticFrame],
    *,
    target_timestamp: float,
    target_world_from_body: np.ndarray,
    map_spec: MapSpec,
    semantic_risk: Mapping[int, float],
    time_tau: float = 30.0,
    conflict_threshold: float = 0.3,
    grass_class_ids=frozenset({3}),
) -> dict[str, np.ndarray]:
    """把内存中的语义帧重新栅格化到目标时刻车体系 BEV。

    这是离线/回归对照实现；实时生产路径使用
    :class:`IncrementalSemanticGrid`，避免每帧遍历整个历史窗口。
    """

    target_pose = validate_transform(target_world_from_body)
    target_from_world = np.linalg.inv(target_pose)
    shape = map_spec.shape
    risk_sum = np.zeros(shape, dtype=np.float64)
    weight_sum = np.zeros(shape, dtype=np.float64)
    grass_sum = np.zeros(shape, dtype=np.float64)
    confidence_map = np.zeros(shape, dtype=np.float32)
    count_map = np.zeros(shape, dtype=np.int32)
    minimum = np.full(shape, np.inf, dtype=np.float32)
    maximum = np.full(shape, -np.inf, dtype=np.float32)
    latest_risk = np.zeros(shape, dtype=np.float32)
    latest_confidence = np.zeros(shape, dtype=np.float32)
    dynamic_map = np.zeros(shape, dtype=bool)

    used = 0
    for frame in frames:
        # 所有历史帧先由世界系转换到目标车体系，再落入固定 BEV 网格。
        age = float(target_timestamp - frame.timestamp)
        if age < -1e-6:
            raise ValueError("semantic window contains a future frame")
        points_body = transform_points(frame.xyz_world, target_from_world)
        rows, columns, inside = body_xy_to_grid(points_body[:, :2], map_spec)
        selected = np.flatnonzero(frame.valid & inside & (frame.confidence > 0.0))
        if not selected.size:
            continue
        used += 1
        rr = rows[selected].astype(np.int64, copy=False)
        cc = columns[selected].astype(np.int64, copy=False)
        labels = frame.labels[selected].astype(np.int64, copy=False)
        point_confidence = frame.confidence[selected].astype(np.float32)
        risks = np.fromiter(
            (float(semantic_risk.get(int(label), 1.0)) for label in labels),
            dtype=np.float32,
            count=labels.size,
        )
        weights = point_confidence * np.float32(np.exp(-age / time_tau))
        np.add.at(risk_sum, (rr, cc), risks * weights)
        np.add.at(weight_sum, (rr, cc), weights)
        np.add.at(count_map, (rr, cc), 1)
        grass = np.isin(labels, tuple(grass_class_ids))
        if np.any(grass):
            np.add.at(grass_sum, (rr[grass], cc[grass]), weights[grass])
        np.maximum.at(confidence_map, (rr, cc), point_confidence)
        np.minimum.at(minimum, (rr, cc), risks)
        np.maximum.at(maximum, (rr, cc), risks)
        latest_risk[rr, cc] = risks
        latest_confidence[rr, cc] = point_confidence
        dynamic = selected[frame.dynamic[selected]]
        if dynamic.size:
            dynamic_map[rows[dynamic], columns[dynamic]] = True

    valid = weight_sum > 0.0
    risk = np.zeros(shape, dtype=np.float32)
    grass_probability = np.zeros(shape, dtype=np.float32)
    risk[valid] = (risk_sum[valid] / weight_sum[valid]).astype(np.float32)
    grass_probability[valid] = (grass_sum[valid] / weight_sum[valid]).astype(
        np.float32
    )
    use_latest = valid & (((maximum - minimum) > conflict_threshold) | dynamic_map)
    risk[use_latest] = latest_risk[use_latest]
    confidence_map[use_latest] = latest_confidence[use_latest]
    return {
        "semantic_hindsight_target": np.clip(risk, 0.0, 1.0),
        "semantic_hindsight_valid_mask": valid,
        "semantic_hindsight_confidence": confidence_map,
        "semantic_hindsight_observation_count": count_map,
        "grass_probability": grass_probability,
        "grass_observation_weight": weight_sum.astype(np.float32),
        "grass_mask": valid & (grass_probability > 0.6) & (weight_sum >= 1.0),
        "semantic_window_used_frames": used,
    }


def map_to_occupancy_order(values: np.ndarray) -> np.ndarray:
    """把 RoadRunner 的前方/左方 BEV 行列顺序转换为 ROS x/y 顺序。"""

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("map must be two-dimensional")
    return array[::-1, ::-1].T


def probability_elevation_layer(
    probability: np.ndarray,
    valid: np.ndarray,
    map_spec: MapSpec,
) -> np.ndarray:
    """把可通行概率包装成高程式图层，无效区域保留为 NaN。"""

    probability_map = np.asarray(probability, dtype=np.float32)
    valid_map = np.asarray(valid, dtype=bool)
    if probability_map.shape != map_spec.shape or valid_map.shape != map_spec.shape:
        raise ValueError("probability and valid maps must match map_spec.shape")
    elevation = np.full(map_spec.shape, np.nan, dtype=np.float32)
    elevation[valid_map] = np.clip(probability_map[valid_map], 0.0, 1.0)
    return elevation


def traversability_probability_layers(
    *,
    geometry_risk: np.ndarray,
    geometry_valid: np.ndarray,
    semantic_risk: np.ndarray,
    semantic_valid: np.ndarray,
    final_risk: np.ndarray,
    final_valid: np.ndarray,
    map_spec: MapSpec,
) -> dict[str, np.ndarray]:
    """将几何、语义和最终风险统一转换为 0～1 可通行概率图层。"""

    return {
        "geometry_probability": probability_elevation_layer(
            1.0 - np.clip(geometry_risk, 0.0, 1.0),
            geometry_valid,
            map_spec,
        ),
        "semantic_probability": probability_elevation_layer(
            1.0 - np.clip(semantic_risk, 0.0, 1.0),
            semantic_valid,
            map_spec,
        ),
        "final_probability": probability_elevation_layer(
            1.0 - np.clip(final_risk, 0.0, 1.0),
            final_valid,
            map_spec,
        ),
    }


def grid_map_column_major_data(layer: np.ndarray) -> np.ndarray:
    """按 Eigen/GridMap 约定的列主序展开二维数组。"""

    array = np.asarray(layer, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("GridMap layer must be two-dimensional")
    return np.ascontiguousarray(array.T).reshape(-1)
