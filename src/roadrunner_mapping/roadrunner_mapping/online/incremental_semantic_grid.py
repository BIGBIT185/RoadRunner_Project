"""GOOSE 在线世界系增量语义栅格。

语义点只在进入地图时做一次世界系整数网格聚合；输出时根据当前机器人位姿
读取局部 BEV，而不是重新遍历完整的历史语义点云。这是实时链避免语义窗口
重复栅格化的核心状态模块。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from roadrunner_mapping.geometry.coordinates import (
    grid_to_body_xy,
    validate_transform,
)
from roadrunner_mapping.geometry.schema import MapSpec

from .risk_statistics import histogram_cvar, weighted_risk_histogram


_EMPTY_KEY = np.iinfo(np.int64).min


class IncrementalSemanticGrid:
    """每个语义点只聚合一次，再按当前车体位姿采样 BEV。"""

    def __init__(
        self,
        *,
        map_spec: MapSpec,
        semantic_risk: Mapping[int, float],
        grass_class_ids=frozenset({3}),
        decay_tau_sec: float = 30.0,
        stale_sec: float = 90.0,
        conflict_threshold: float = 0.3,
        risk_histogram_bins: int = 16,
        risk_cvar_alpha: float = 0.8,
    ) -> None:
        if not np.isfinite(decay_tau_sec) or decay_tau_sec <= 0.0:
            raise ValueError("decay_tau_sec must be finite and positive")
        if not np.isfinite(stale_sec) or stale_sec <= 0.0:
            raise ValueError("stale_sec must be finite and positive")
        if stale_sec < decay_tau_sec:
            raise ValueError("stale_sec must not be shorter than decay_tau_sec")
        if not np.isfinite(conflict_threshold) or conflict_threshold < 0.0:
            raise ValueError("conflict_threshold must be finite and non-negative")
        if int(risk_histogram_bins) != risk_histogram_bins or risk_histogram_bins < 2:
            raise ValueError("risk_histogram_bins must be an integer >= 2")
        if not np.isfinite(risk_cvar_alpha) or not 0.0 < risk_cvar_alpha < 1.0:
            raise ValueError("risk_cvar_alpha must be finite and in (0,1)")

        self.map_spec = map_spec
        self.decay_tau_sec = float(decay_tau_sec)
        self.stale_sec = float(stale_sec)
        self.conflict_threshold = float(conflict_threshold)
        self.risk_histogram_bins = int(risk_histogram_bins)
        self.risk_cvar_alpha = float(risk_cvar_alpha)
        x_length = float(map_spec.x_max - map_spec.x_min)
        y_length = float(map_spec.y_max - map_spec.y_min)
        self.storage_size = int(
            np.ceil(np.hypot(x_length, y_length) / map_spec.resolution)
        ) + 3
        self.storage_half_extent = (
            0.5 * (self.storage_size - 2) * map_spec.resolution
        )

        self._risk_lut = np.ones(256, dtype=np.float32)
        for class_id, risk in semantic_risk.items():
            class_index = int(class_id)
            if class_index < 0 or class_index >= self._risk_lut.size:
                raise ValueError("semantic class ids must be in [0,255]")
            if not np.isfinite(risk):
                raise ValueError("semantic risks must be finite")
            self._risk_lut[class_index] = np.float32(np.clip(risk, 0.0, 1.0))
        self._grass_lut = np.zeros(256, dtype=bool)
        for class_id in grass_class_ids:
            class_index = int(class_id)
            if class_index < 0 or class_index >= self._grass_lut.size:
                raise ValueError("grass class ids must be in [0,255]")
            self._grass_lut[class_index] = True

        rows, columns = np.indices(map_spec.shape)
        self._output_body_xy = grid_to_body_xy(rows, columns, map_spec).reshape(-1, 2)
        self._shape = (self.storage_size, self.storage_size)
        self.clear()

    def clear(self) -> None:
        """清空世界系哈希槽和所有衰减/冲突状态。"""
        self._key_x = np.full(self._shape, _EMPTY_KEY, dtype=np.int64)
        self._key_y = np.full(self._shape, _EMPTY_KEY, dtype=np.int64)
        self._risk_sum = np.zeros(self._shape, dtype=np.float32)
        self._risk_sq_sum = np.zeros(self._shape, dtype=np.float32)
        self._weight_sum = np.zeros(self._shape, dtype=np.float32)
        self._risk_histogram = [
            np.zeros(self._shape, dtype=np.float32)
            for _ in range(self.risk_histogram_bins)
        ]
        self._grass_sum = np.zeros(self._shape, dtype=np.float32)
        self._count_sum = np.zeros(self._shape, dtype=np.float32)
        self._confidence = np.zeros(self._shape, dtype=np.float32)
        self._latest_risk = np.zeros(self._shape, dtype=np.float32)
        self._latest_confidence = np.zeros(self._shape, dtype=np.float32)
        self._last_update = np.full(self._shape, -np.inf, dtype=np.float64)
        self._last_conflict = np.full(self._shape, -np.inf, dtype=np.float64)
        self._last_dynamic = np.full(self._shape, -np.inf, dtype=np.float64)
        self._last_timestamp = None
        self._last_position = None
        self._last_input_points = 0
        self._last_aggregated_cells = 0

    def _reset_slots(self, rows: np.ndarray, columns: np.ndarray) -> None:
        """清理环形存储中被新世界网格键占用的旧槽位。"""
        if not rows.size:
            return
        index = (rows, columns)
        self._key_x[index] = _EMPTY_KEY
        self._key_y[index] = _EMPTY_KEY
        self._risk_sum[index] = 0.0
        self._risk_sq_sum[index] = 0.0
        self._weight_sum[index] = 0.0
        for histogram in self._risk_histogram:
            histogram[index] = 0.0
        self._grass_sum[index] = 0.0
        self._count_sum[index] = 0.0
        self._confidence[index] = 0.0
        self._latest_risk[index] = 0.0
        self._latest_confidence[index] = 0.0
        self._last_update[index] = -np.inf
        self._last_conflict[index] = -np.inf
        self._last_dynamic[index] = -np.inf

    def _global_keys(self, world_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把世界平面坐标转换成与地图分辨率绑定的整数键。"""
        keys = np.floor(
            np.asarray(world_xy, dtype=np.float64) / self.map_spec.resolution
        ).astype(np.int64)
        return keys[:, 0], keys[:, 1]

    def _aggregate_frame(
        self,
        *,
        timestamp: float,
        position: np.ndarray,
        xyz_world: np.ndarray,
        labels: np.ndarray,
        confidence: np.ndarray,
        valid: np.ndarray,
        dynamic: np.ndarray,
    ) -> tuple[int, int]:
        """聚合一帧语义点并更新世界系槽位。

        同一帧同一网格内先用置信度加权平均，随后再和该网格的历史状态按
        时间衰减合并。冲突类别或动态类别会短暂使用最新可靠观测。
        """
        xyz = np.asarray(xyz_world, dtype=np.float32)
        label_values = np.asarray(labels, dtype=np.uint8)
        confidence_values = np.asarray(confidence, dtype=np.float32)
        valid_values = np.asarray(valid, dtype=bool)
        dynamic_values = np.asarray(dynamic, dtype=bool)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_world must have shape [N,3]")
        count = xyz.shape[0]
        if any(
            value.shape != (count,)
            for value in (
                label_values,
                confidence_values,
                valid_values,
                dynamic_values,
            )
        ):
            raise ValueError("semantic fields must match the point count")

        # 只让有效、有限、置信度为正且位于滚动存储范围内的点更新地图。
        selected = (
            valid_values
            & np.isfinite(xyz).all(axis=1)
            & np.isfinite(confidence_values)
            & (confidence_values > 0.0)
            & (confidence_values <= 1.0)
            & (np.abs(xyz[:, 0] - position[0]) < self.storage_half_extent)
            & (np.abs(xyz[:, 1] - position[1]) < self.storage_half_extent)
        )
        input_points = int(np.count_nonzero(selected))
        if not input_points:
            return 0, 0

        xyz = xyz[selected]
        label_values = label_values[selected]
        confidence_values = confidence_values[selected]
        dynamic_values = dynamic_values[selected]
        key_x, key_y = self._global_keys(xyz[:, :2])
        # inverse 把每个输入点映射到唯一世界网格，避免保存同一单元的大量重复点。
        unique_keys, inverse = np.unique(
            np.stack((key_x, key_y), axis=1),
            axis=0,
            return_inverse=True,
        )
        cell_count = unique_keys.shape[0]
        risks = self._risk_lut[label_values]
        weights = confidence_values
        risk_sum = np.bincount(
            inverse, weights=risks * weights, minlength=cell_count
        ).astype(np.float32)
        risk_sq_sum = np.bincount(
            inverse, weights=risks * risks * weights, minlength=cell_count
        ).astype(np.float32)
        weight_sum = np.bincount(
            inverse, weights=weights, minlength=cell_count
        ).astype(np.float32)
        point_count = np.bincount(inverse, minlength=cell_count).astype(np.float32)
        grass_sum = np.bincount(
            inverse,
            weights=weights * self._grass_lut[label_values],
            minlength=cell_count,
        ).astype(np.float32)
        maximum_confidence = np.zeros(cell_count, dtype=np.float32)
        minimum_risk = np.full(cell_count, np.inf, dtype=np.float32)
        maximum_risk = np.full(cell_count, -np.inf, dtype=np.float32)
        dynamic_cell = np.zeros(cell_count, dtype=bool)
        np.maximum.at(maximum_confidence, inverse, confidence_values)
        np.minimum.at(minimum_risk, inverse, risks)
        np.maximum.at(maximum_risk, inverse, risks)
        np.logical_or.at(dynamic_cell, inverse, dynamic_values)
        frame_risk = np.divide(
            risk_sum,
            weight_sum,
            out=np.zeros_like(risk_sum),
            where=weight_sum > 0.0,
        )

        slot_rows = np.mod(unique_keys[:, 0], self.storage_size)
        slot_columns = np.mod(unique_keys[:, 1], self.storage_size)
        slots = (slot_rows, slot_columns)
        same_key = (
            (self._key_x[slots] == unique_keys[:, 0])
            & (self._key_y[slots] == unique_keys[:, 1])
        )
        self._reset_slots(slot_rows[~same_key], slot_columns[~same_key])
        self._key_x[slots] = unique_keys[:, 0]
        self._key_y[slots] = unique_keys[:, 1]

        # 先衰减旧权重，再加入当前帧；未来时间戳不会增加历史权重。
        previous_age = np.maximum(0.0, timestamp - self._last_update[slots])
        previous_decay = np.zeros(cell_count, dtype=np.float32)
        previous_decay[same_key] = np.exp(
            -previous_age[same_key] / self.decay_tau_sec
        ).astype(np.float32)
        previous_weight = self._weight_sum[slots] * previous_decay
        previous_risk = np.divide(
            self._risk_sum[slots],
            self._weight_sum[slots],
            out=np.zeros(cell_count, dtype=np.float32),
            where=self._weight_sum[slots] > 0.0,
        )
        conflict = (
            (maximum_risk - minimum_risk) > self.conflict_threshold
        ) | (
            same_key
            & (previous_weight > 0.0)
            & (np.abs(previous_risk - frame_risk) > self.conflict_threshold)
        )

        self._risk_sum[slots] = (
            self._risk_sum[slots] * previous_decay + risk_sum
        )
        self._risk_sq_sum[slots] = (
            self._risk_sq_sum[slots] * previous_decay + risk_sq_sum
        )
        self._weight_sum[slots] = previous_weight + weight_sum
        frame_histogram = weighted_risk_histogram(
            np, inverse, risks, weights, cell_count, self.risk_histogram_bins
        )
        for index, histogram in enumerate(self._risk_histogram):
            histogram[slots] = (
                histogram[slots] * previous_decay + frame_histogram[index]
            )
        self._grass_sum[slots] = (
            self._grass_sum[slots] * previous_decay + grass_sum
        )
        self._count_sum[slots] = (
            self._count_sum[slots] * previous_decay + point_count
        )
        self._confidence[slots] = np.maximum(
            self._confidence[slots] * previous_decay,
            maximum_confidence,
        )
        self._latest_risk[slots] = frame_risk
        self._latest_confidence[slots] = maximum_confidence
        self._last_update[slots] = timestamp
        self._last_conflict[slot_rows[conflict], slot_columns[conflict]] = timestamp
        self._last_dynamic[
            slot_rows[dynamic_cell], slot_columns[dynamic_cell]
        ] = timestamp
        return input_points, int(cell_count)

    def _sample(
        self,
        *,
        timestamp: float,
        world_from_body: np.ndarray,
        input_points: int,
        aggregated_cells: int,
    ) -> dict[str, np.ndarray]:
        """从世界系槽位读取当前车体系 BEV，形状由 ``map_spec`` 决定。"""
        rotation = world_from_body[:2, :2]
        translation = world_from_body[:2, 3]
        # 输出网格固定在 base_link，当前位姿只负责把每个车体单元映射回世界键。
        world_xy = self._output_body_xy @ rotation.T + translation
        key_x, key_y = self._global_keys(world_xy)
        slot_rows = np.mod(key_x, self.storage_size)
        slot_columns = np.mod(key_y, self.storage_size)
        slots = (slot_rows, slot_columns)
        key_match = (
            (self._key_x[slots] == key_x)
            & (self._key_y[slots] == key_y)
        )
        age = np.maximum(0.0, timestamp - self._last_update[slots])
        stale = key_match & (age > self.stale_sec)
        self._reset_slots(slot_rows[stale], slot_columns[stale])
        key_match[stale] = False
        decay = np.exp(-age / self.decay_tau_sec).astype(np.float32)
        effective_weight = self._weight_sum[slots] * decay
        not_future = self._last_update[slots] <= timestamp + 1e-6
        valid = (
            key_match
            & not_future
            & (age <= self.stale_sec)
            & (effective_weight > 1e-6)
        )
        risk_mean = np.divide(
            self._risk_sum[slots],
            self._weight_sum[slots],
            out=np.zeros_like(effective_weight),
            where=self._weight_sum[slots] > 0.0,
        )
        risk_sq_mean = np.divide(
            self._risk_sq_sum[slots],
            self._weight_sum[slots],
            out=np.zeros_like(effective_weight),
            where=self._weight_sum[slots] > 0.0,
        )
        risk_std = np.sqrt(
            np.maximum(risk_sq_mean - risk_mean * risk_mean, 0.0)
        )
        risk_histogram = [
            histogram[slots] * decay for histogram in self._risk_histogram
        ]
        risk_cvar = histogram_cvar(
            np,
            risk_histogram,
            effective_weight,
            self.risk_cvar_alpha,
        )
        risk = risk_mean.copy()
        confidence = self._confidence[slots] * decay
        conflict = (timestamp - self._last_conflict[slots]) <= self.stale_sec
        dynamic = (timestamp - self._last_dynamic[slots]) <= self.stale_sec
        use_latest = valid & (conflict | dynamic)
        risk[use_latest] = self._latest_risk[slots][use_latest]
        confidence[use_latest] = (
            self._latest_confidence[slots][use_latest] * decay[use_latest]
        )
        risk_cvar[use_latest] = np.maximum(
            risk_cvar[use_latest], risk[use_latest]
        )
        grass_weight = self._grass_sum[slots] * decay
        grass_probability = np.divide(
            grass_weight,
            effective_weight,
            out=np.zeros_like(grass_weight),
            where=effective_weight > 0.0,
        )
        effective_count = np.rint(self._count_sum[slots] * decay).astype(np.int32)
        effective_count[valid & (effective_count < 1)] = 1
        effective_count[~valid] = 0

        shape = self.map_spec.shape
        valid_map = valid.reshape(shape)
        weight_map = effective_weight.reshape(shape).astype(np.float32)
        grass_map = grass_probability.reshape(shape).astype(np.float32)
        return {
            "semantic_hindsight_target": np.clip(
                risk.reshape(shape), 0.0, 1.0
            ).astype(np.float32),
            "semantic_hindsight_valid_mask": valid_map,
            "semantic_hindsight_confidence": np.clip(
                confidence.reshape(shape), 0.0, 1.0
            ).astype(np.float32),
            "semantic_hindsight_observation_count": effective_count.reshape(shape),
            "semantic_hindsight_risk_std": np.where(
                valid, risk_std, 0.0
            ).reshape(shape).astype(np.float32),
            "semantic_hindsight_risk_cvar": np.where(
                valid, np.clip(risk_cvar, 0.0, 1.0), 0.0
            ).reshape(shape).astype(np.float32),
            "semantic_hindsight_risk_distribution_valid": valid.reshape(shape),
            "grass_probability": grass_map,
            "grass_observation_weight": weight_map,
            "grass_mask": valid_map & (grass_map > 0.6) & (weight_map >= 1.0),
            "semantic_input_points": int(input_points),
            "semantic_aggregated_cells": int(aggregated_cells),
            "semantic_valid_cells": int(np.count_nonzero(valid)),
        }

    def sample_world(
        self,
        *,
        timestamp: float,
        center,
        world_map_length: float,
        resolution: float,
    ) -> dict[str, np.ndarray]:
        """读取 CPU 回退栅格的世界坐标快照。"""

        timestamp_value = float(timestamp)
        center = np.asarray(center, dtype=np.float64).reshape(2)
        world_map_length = float(world_map_length)
        resolution = float(resolution)
        if (
            not np.isfinite(timestamp_value)
            or not np.isfinite(center).all()
            or not np.isfinite(world_map_length)
            or world_map_length <= 0.0
            or not np.isfinite(resolution)
            or resolution <= 0.0
        ):
            raise ValueError("world map parameters must be finite and positive")
        cell_n = int(round(world_map_length / resolution))
        if cell_n <= 0:
            raise ValueError("world map must contain at least one cell")
        rows, columns = np.indices((cell_n, cell_n), dtype=np.float64)
        world_x = center[0] + 0.5 * world_map_length - (rows + 0.5) * resolution
        world_y = center[1] + 0.5 * world_map_length - (columns + 0.5) * resolution
        key_x, key_y = self._global_keys(
            np.stack((world_x.reshape(-1), world_y.reshape(-1)), axis=1)
        )
        slot_rows = np.mod(key_x, self.storage_size)
        slot_columns = np.mod(key_y, self.storage_size)
        slots = (slot_rows, slot_columns)
        key_match = (
            (self._key_x[slots] == key_x)
            & (self._key_y[slots] == key_y)
        )
        age = np.maximum(0.0, timestamp_value - self._last_update[slots])
        valid = (
            key_match
            & (self._last_update[slots] <= timestamp_value + 1e-6)
            & (age <= self.stale_sec)
            & (self._weight_sum[slots] > 1e-6)
        )
        decay = np.exp(-age / self.decay_tau_sec).astype(np.float32)
        effective_weight = self._weight_sum[slots] * decay
        risk_mean = np.divide(
            self._risk_sum[slots],
            np.maximum(self._weight_sum[slots], 1e-12),
            out=np.zeros_like(effective_weight),
            where=self._weight_sum[slots] > 0.0,
        )
        risk_sq_mean = np.divide(
            self._risk_sq_sum[slots],
            np.maximum(self._weight_sum[slots], 1e-12),
            out=np.zeros_like(effective_weight),
            where=self._weight_sum[slots] > 0.0,
        )
        risk_std = np.sqrt(
            np.maximum(risk_sq_mean - risk_mean * risk_mean, 0.0)
        )
        risk_cvar = histogram_cvar(
            np,
            [histogram[slots] * decay for histogram in self._risk_histogram],
            effective_weight,
            self.risk_cvar_alpha,
        )
        risk = risk_mean.copy()
        confidence = self._confidence[slots] * decay
        conflict = (timestamp_value - self._last_conflict[slots]) <= self.stale_sec
        dynamic = (timestamp_value - self._last_dynamic[slots]) <= self.stale_sec
        use_latest = valid & (conflict | dynamic)
        risk[use_latest] = self._latest_risk[slots][use_latest]
        confidence[use_latest] = (
            self._latest_confidence[slots][use_latest] * decay[use_latest]
        )
        risk_cvar[use_latest] = np.maximum(
            risk_cvar[use_latest], risk[use_latest]
        )
        grass_weight = self._grass_sum[slots] * decay
        effective_count = np.rint(
            self._count_sum[slots] * decay
        ).astype(np.int32)
        effective_count[valid & (effective_count < 1)] = 1
        effective_count[~valid] = 0
        grass_probability = np.divide(
            grass_weight,
            np.maximum(effective_weight, 1e-12),
            out=np.zeros_like(grass_weight),
            where=effective_weight > 0.0,
        )
        shape = (cell_n, cell_n)
        return {
            "semantic_hindsight_target": np.clip(risk.reshape(shape), 0.0, 1.0).astype(np.float32),
            "semantic_hindsight_valid_mask": valid.reshape(shape),
            "semantic_hindsight_confidence": np.clip(
                confidence.reshape(shape), 0.0, 1.0
            ).astype(np.float32),
            "semantic_hindsight_observation_count": effective_count.reshape(shape),
            "semantic_hindsight_risk_std": np.where(
                valid, risk_std, 0.0
            ).reshape(shape).astype(np.float32),
            "semantic_hindsight_risk_cvar": np.where(
                valid, np.clip(risk_cvar, 0.0, 1.0), 0.0
            ).reshape(shape).astype(np.float32),
            "semantic_hindsight_risk_distribution_valid": valid.reshape(shape),
            "grass_probability": grass_probability.reshape(shape).astype(np.float32),
            "grass_observation_weight": effective_weight.reshape(shape).astype(np.float32),
            "grass_mask": (
                valid
                & (grass_probability > 0.6)
                & (effective_weight >= 1.0)
            ).reshape(shape),
            "semantic_input_points": int(self._last_input_points),
            "semantic_aggregated_cells": int(self._last_aggregated_cells),
            "semantic_valid_cells": int(np.count_nonzero(valid)),
        }

    def update(
        self,
        *,
        timestamp: float,
        world_from_body: np.ndarray,
        xyz_world: np.ndarray,
        labels: np.ndarray,
        confidence: np.ndarray,
        valid: np.ndarray,
        dynamic: np.ndarray,
    ) -> dict[str, int]:
        """加入一帧语义观测，但不改变输出采样时钟。"""

        timestamp_value = float(timestamp)
        if not np.isfinite(timestamp_value):
            raise ValueError("timestamp must be finite")
        pose = validate_transform(world_from_body)
        position = pose[:2, 3]
        # rosbag 重播导致时间倒退时，旧地图不能与新 epoch 混合。
        if self._last_timestamp is not None and timestamp_value < self._last_timestamp:
            self.clear()
        # 机器人移动超过存储半径后，旧世界槽位不再能可靠代表当前局部地图。
        if (
            self._last_position is not None
            and np.linalg.norm(position - self._last_position)
            >= self.storage_half_extent
        ):
            self.clear()
        input_points, aggregated_cells = self._aggregate_frame(
            timestamp=timestamp_value,
            position=position,
            xyz_world=xyz_world,
            labels=labels,
            confidence=confidence,
            valid=valid,
            dynamic=dynamic,
        )
        self._last_timestamp = timestamp_value
        self._last_position = position.copy()
        self._last_input_points = input_points
        self._last_aggregated_cells = aggregated_cells
        return {
            "semantic_input_points": int(input_points),
            "semantic_aggregated_cells": int(aggregated_cells),
        }

    def sample(
        self,
        *,
        timestamp: float,
        world_from_body: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """读取当前车体系 BEV，不向语义地图写入新观测。"""

        timestamp_value = float(timestamp)
        if not np.isfinite(timestamp_value):
            raise ValueError("timestamp must be finite")
        pose = validate_transform(world_from_body)
        return self._sample(
            timestamp=timestamp_value,
            world_from_body=pose,
            input_points=self._last_input_points,
            aggregated_cells=self._last_aggregated_cells,
        )

    def advance(
        self,
        *,
        timestamp: float,
        world_from_body: np.ndarray,
        xyz_world: np.ndarray,
        labels: np.ndarray,
        confidence: np.ndarray,
        valid: np.ndarray,
        dynamic: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compatibility helper that updates and samples at one timestamp."""

        self.update(
            timestamp=timestamp,
            world_from_body=world_from_body,
            xyz_world=xyz_world,
            labels=labels,
            confidence=confidence,
            valid=valid,
            dynamic=dynamic,
        )
        return self.sample(
            timestamp=timestamp,
            world_from_body=world_from_body,
        )
