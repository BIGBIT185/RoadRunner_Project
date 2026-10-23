"""GPU 增量语义栅格，并复用几何链的滚动索引。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ..geometry.coordinates import validate_transform
from ..geometry.gpu_rolling_grid import (
    GpuRollingGrid,
    bilinear_world_xy_samples,
    weighted_neighbor_sample,
)
from ..geometry.schema import MapSpec
from .risk_statistics import histogram_cvar, weighted_risk_histogram


class GpuIncrementalSemanticGrid:
    """在共享 GPU 滚动窗口中增量聚合语义风险。"""

    def __init__(
        self,
        *,
        map_spec: MapSpec,
        semantic_risk: Mapping[int, float],
        grass_class_ids=frozenset({3}),
        decay_tau_sec: float = 30.0,
        stale_sec: float = 10.0,
        rolling_grid: GpuRollingGrid,
        sample_mode: str = "nearest",
        sample_min_valid_weight: float = 0.5,
        risk_histogram_bins: int = 16,
        risk_cvar_alpha: float = 0.8,
    ) -> None:
        if not np.isfinite(decay_tau_sec) or decay_tau_sec <= 0.0:
            raise ValueError("decay_tau_sec must be finite and positive")
        if not np.isfinite(stale_sec) or stale_sec < decay_tau_sec:
            raise ValueError("stale_sec must be finite and >= decay_tau_sec")
        if sample_mode not in {"bilinear", "nearest"}:
            raise ValueError("sample_mode must be bilinear or nearest")
        if (
            not np.isfinite(sample_min_valid_weight)
            or not 0.0 <= sample_min_valid_weight <= 1.0
        ):
            raise ValueError("sample_min_valid_weight must be finite and in [0,1]")
        if int(risk_histogram_bins) != risk_histogram_bins or risk_histogram_bins < 2:
            raise ValueError("risk_histogram_bins must be an integer >= 2")
        if not np.isfinite(risk_cvar_alpha) or not 0.0 < risk_cvar_alpha < 1.0:
            raise ValueError("risk_cvar_alpha must be finite and in (0,1)")
        self.map_spec = map_spec
        self.rolling_grid = rolling_grid
        self.cp = rolling_grid.cp
        self.device = rolling_grid.device
        self.decay_tau_sec = float(decay_tau_sec)
        self.stale_sec = float(stale_sec)
        self.sample_mode = str(sample_mode)
        self.sample_min_valid_weight = float(sample_min_valid_weight)
        self.risk_histogram_bins = int(risk_histogram_bins)
        self.risk_cvar_alpha = float(risk_cvar_alpha)
        self._risk_lut = self.cp.ones(256, dtype=self.cp.float32)
        for class_id, risk in semantic_risk.items():
            class_index = int(class_id)
            if class_index < 0 or class_index >= 256:
                raise ValueError("semantic class ids must be in [0,255]")
            if not np.isfinite(risk):
                raise ValueError("semantic risks must be finite")
            self._risk_lut[class_index] = np.float32(np.clip(risk, 0.0, 1.0))
        self._grass_lut = self.cp.zeros(256, dtype=self.cp.bool_)
        for class_id in grass_class_ids:
            class_index = int(class_id)
            if class_index < 0 or class_index >= 256:
                raise ValueError("grass class ids must be in [0,255]")
            self._grass_lut[class_index] = True

        shape = rolling_grid.shape
        with self.device:
            self._risk_sum = self.cp.zeros(shape, dtype=self.cp.float32)
            self._risk_sq_sum = self.cp.zeros(shape, dtype=self.cp.float32)
            self._weight_sum = self.cp.zeros(shape, dtype=self.cp.float32)
            self._risk_histogram = [
                self.cp.zeros(shape, dtype=self.cp.float32)
                for _ in range(self.risk_histogram_bins)
            ]
            self._grass_sum = self.cp.zeros(shape, dtype=self.cp.float32)
            self._count_sum = self.cp.zeros(shape, dtype=self.cp.float32)
            self._confidence = self.cp.zeros(shape, dtype=self.cp.float32)
            self._latest_risk = self.cp.zeros(shape, dtype=self.cp.float32)
            self._latest_confidence = self.cp.zeros(shape, dtype=self.cp.float32)
            self._last_update = self.cp.full(shape, -self.cp.inf, dtype=self.cp.float64)
            self._last_conflict = self.cp.full(shape, -self.cp.inf, dtype=self.cp.float64)
            self._last_dynamic = self.cp.full(shape, -self.cp.inf, dtype=self.cp.float64)
        for layer, fill in (
            (self._risk_sum, 0.0),
            (self._risk_sq_sum, 0.0),
            (self._weight_sum, 0.0),
            (self._grass_sum, 0.0),
            (self._count_sum, 0.0),
            (self._confidence, 0.0),
            (self._latest_risk, 0.0),
            (self._latest_confidence, 0.0),
            (self._last_update, -self.cp.inf),
            (self._last_conflict, -self.cp.inf),
            (self._last_dynamic, -self.cp.inf),
        ):
            rolling_grid.register_layer(layer, fill)
        for histogram in self._risk_histogram:
            rolling_grid.register_layer(histogram, 0.0)

        rows, columns = np.indices(map_spec.shape)
        body_x = map_spec.x_max - (rows + 0.5) * map_spec.resolution
        body_y = map_spec.y_max - (columns + 0.5) * map_spec.resolution
        with self.device:
            self._body_x = self.cp.asarray(body_x.reshape(-1), dtype=self.cp.float32)
            self._body_y = self.cp.asarray(body_y.reshape(-1), dtype=self.cp.float32)
        self._last_timestamp = None
        self._last_position = None
        self._last_input_points = 0
        self._last_aggregated_cells = 0

    def _reset_slots(self, rows, columns) -> None:
        if rows.size == 0:
            return
        index = (rows, columns)
        for layer, fill in (
            (self._risk_sum, 0.0),
            (self._risk_sq_sum, 0.0),
            (self._weight_sum, 0.0),
            (self._grass_sum, 0.0),
            (self._count_sum, 0.0),
            (self._confidence, 0.0),
            (self._latest_risk, 0.0),
            (self._latest_confidence, 0.0),
            (self._last_update, -self.cp.inf),
            (self._last_conflict, -self.cp.inf),
            (self._last_dynamic, -self.cp.inf),
        ):
            layer[index] = fill
        for histogram in self._risk_histogram:
            histogram[index] = 0.0

    def clear(self) -> None:
        """清空语义图层；共享窗口中心由几何地图负责维护。"""

        with self.rolling_grid.lock:
            with self.device:
                indices = self.cp.indices(self.rolling_grid.shape)
                self._reset_slots(indices[0].ravel(), indices[1].ravel())
        self._last_timestamp = None
        self._last_position = None
        self._last_input_points = 0
        self._last_aggregated_cells = 0

    def _aggregate_frame(
        self,
        *,
        timestamp: float,
        position: np.ndarray,
        xyz_world,
        labels,
        confidence,
        valid,
        dynamic,
    ) -> tuple[int, int]:
        cp = self.cp
        xyz = cp.asarray(xyz_world, dtype=cp.float32)
        label_values = cp.asarray(labels, dtype=cp.uint8)
        confidence_values = cp.asarray(confidence, dtype=cp.float32)
        valid_values = cp.asarray(valid, dtype=cp.bool_)
        dynamic_values = cp.asarray(dynamic, dtype=cp.bool_)
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
        if not self.rolling_grid.initialized:
            return 0, 0

        rows, columns, inside = self.rolling_grid.world_indices(xyz)
        selected = (
            inside
            & valid_values
            & cp.isfinite(confidence_values)
            & (confidence_values > 0.0)
            & (confidence_values <= 1.0)
        )
        input_points = int(cp.count_nonzero(selected).get())
        if not input_points:
            return 0, 0

        flat = (rows[selected] * self.rolling_grid.cell_n + columns[selected]).astype(
            cp.int64
        )
        labels_selected = label_values[selected]
        confidence_selected = confidence_values[selected]
        dynamic_selected = dynamic_values[selected]
        risks = self._risk_lut[labels_selected]
        unique_flat, inverse = cp.unique(flat, return_inverse=True)
        cell_count = int(unique_flat.size)
        risk_sum = cp.bincount(
            inverse, weights=risks * confidence_selected, minlength=cell_count
        ).astype(cp.float32)
        risk_sq_sum = cp.bincount(
            inverse,
            weights=risks * risks * confidence_selected,
            minlength=cell_count,
        ).astype(cp.float32)
        weight_sum = cp.bincount(
            inverse, weights=confidence_selected, minlength=cell_count
        ).astype(cp.float32)
        point_count = cp.bincount(inverse, minlength=cell_count).astype(cp.float32)
        grass_sum = cp.bincount(
            inverse,
            weights=confidence_selected * self._grass_lut[labels_selected],
            minlength=cell_count,
        ).astype(cp.float32)
        maximum_confidence = cp.zeros(cell_count, dtype=cp.float32)
        minimum_risk = cp.full(cell_count, cp.inf, dtype=cp.float32)
        maximum_risk = cp.full(cell_count, -cp.inf, dtype=cp.float32)
        # CuPy 尚未为 bool 支持 ``logical_or.at``，用 uint8 最大值实现同样
        # 的按网格逻辑或，再转换回布尔掩码。
        dynamic_cell_int32 = cp.zeros(cell_count, dtype=cp.int32)
        cp.maximum.at(maximum_confidence, inverse, confidence_selected)
        cp.minimum.at(minimum_risk, inverse, risks)
        cp.maximum.at(maximum_risk, inverse, risks)
        cp.maximum.at(dynamic_cell_int32, inverse, dynamic_selected.astype(cp.int32))
        dynamic_cell = dynamic_cell_int32.astype(cp.bool_)
        frame_risk = cp.where(
            weight_sum > 0.0,
            risk_sum / cp.maximum(weight_sum, 1e-12),
            0.0,
        )

        flat_risk = self._risk_sum.ravel()
        flat_risk_sq = self._risk_sq_sum.ravel()
        flat_weight = self._weight_sum.ravel()
        flat_histogram = [histogram.ravel() for histogram in self._risk_histogram]
        flat_grass = self._grass_sum.ravel()
        flat_count = self._count_sum.ravel()
        flat_confidence = self._confidence.ravel()
        flat_latest_risk = self._latest_risk.ravel()
        flat_latest_confidence = self._latest_confidence.ravel()
        flat_last_update = self._last_update.ravel()
        flat_last_conflict = self._last_conflict.ravel()
        flat_last_dynamic = self._last_dynamic.ravel()
        previous_age = cp.maximum(0.0, np.float64(timestamp) - flat_last_update[unique_flat])
        previous_decay = cp.exp(-previous_age / self.decay_tau_sec).astype(cp.float32)
        previous_weight = flat_weight[unique_flat] * previous_decay
        previous_risk = cp.where(
            flat_weight[unique_flat] > 0.0,
            flat_risk[unique_flat] / cp.maximum(flat_weight[unique_flat], 1e-12),
            0.0,
        )
        conflict = (
            (maximum_risk - minimum_risk) > 0.3
        ) | (
            (previous_weight > 0.0)
            & (cp.abs(previous_risk - frame_risk) > 0.3)
        )
        flat_risk[unique_flat] = flat_risk[unique_flat] * previous_decay + risk_sum
        flat_risk_sq[unique_flat] = (
            flat_risk_sq[unique_flat] * previous_decay + risk_sq_sum
        )
        flat_weight[unique_flat] = previous_weight + weight_sum
        frame_histogram = weighted_risk_histogram(
            cp, inverse, risks, confidence_selected, cell_count, self.risk_histogram_bins
        )
        for index, histogram in enumerate(flat_histogram):
            histogram[unique_flat] = (
                histogram[unique_flat] * previous_decay + frame_histogram[index]
            )
        flat_grass[unique_flat] = (
            flat_grass[unique_flat] * previous_decay + grass_sum
        )
        flat_count[unique_flat] = (
            flat_count[unique_flat] * previous_decay + point_count
        )
        flat_confidence[unique_flat] = cp.maximum(
            flat_confidence[unique_flat] * previous_decay,
            maximum_confidence,
        )
        flat_latest_risk[unique_flat] = frame_risk
        flat_latest_confidence[unique_flat] = maximum_confidence
        flat_last_update[unique_flat] = np.float64(timestamp)
        flat_last_conflict[unique_flat[conflict]] = np.float64(timestamp)
        flat_last_dynamic[unique_flat[dynamic_cell]] = np.float64(timestamp)
        return input_points, cell_count

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
        timestamp_value = float(timestamp)
        if not np.isfinite(timestamp_value):
            raise ValueError("timestamp must be finite")
        pose = validate_transform(world_from_body)
        position = pose[:2, 3]
        if self._last_timestamp is not None and timestamp_value < self._last_timestamp:
            self.clear()
        if (
            self._last_position is not None
            and np.linalg.norm(position - self._last_position)
            >= 0.5 * self.rolling_grid.world_map_length
        ):
            self.clear()
        with self.rolling_grid.lock:
            with self.device:
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
            "semantic_input_points": input_points,
            "semantic_aggregated_cells": aggregated_cells,
        }

    def sample(
        self,
        *,
        timestamp: float,
        world_from_body: np.ndarray,
    ) -> dict[str, np.ndarray]:
        with self.rolling_grid.lock:
            return self._sample_locked(
                timestamp=timestamp,
                world_from_body=world_from_body,
            )

    def _sample_locked(
        self,
        *,
        timestamp: float,
        world_from_body: np.ndarray,
    ) -> dict[str, np.ndarray]:
        timestamp_value = float(timestamp)
        pose = validate_transform(world_from_body)
        shape = self.map_spec.shape
        if not self.rolling_grid.initialized:
            return {
                "semantic_hindsight_target": np.zeros(shape, dtype=np.float32),
                "semantic_hindsight_valid_mask": np.zeros(shape, dtype=bool),
                "semantic_hindsight_confidence": np.zeros(shape, dtype=np.float32),
                "semantic_hindsight_observation_count": np.zeros(shape, dtype=np.int32),
                "semantic_hindsight_risk_std": np.zeros(shape, dtype=np.float32),
                "semantic_hindsight_risk_cvar": np.zeros(shape, dtype=np.float32),
                "semantic_hindsight_risk_distribution_valid": np.zeros(
                    shape, dtype=bool
                ),
                "grass_probability": np.zeros(shape, dtype=np.float32),
                "grass_observation_weight": np.zeros(shape, dtype=np.float32),
                "grass_mask": np.zeros(shape, dtype=bool),
                "semantic_input_points": self._last_input_points,
                "semantic_aggregated_cells": self._last_aggregated_cells,
                "semantic_valid_cells": 0,
            }
        cp = self.cp
        rotation = cp.asarray(pose[:2, :2], dtype=cp.float32)
        translation = cp.asarray(pose[:2, 3], dtype=cp.float32)
        world_x = self._body_x * rotation[0, 0] + self._body_y * rotation[0, 1] + translation[0]
        world_y = self._body_x * rotation[1, 0] + self._body_y * rotation[1, 1] + translation[1]
        if self.sample_mode == "nearest":
            rows, columns, inside = self.rolling_grid.world_xy_indices(
                world_x,
                world_y,
            )
            safe_rows = cp.clip(rows, 0, self.rolling_grid.cell_n - 1)
            safe_columns = cp.clip(columns, 0, self.rolling_grid.cell_n - 1)
            slots = (safe_rows, safe_columns)
            age = cp.maximum(
                0.0,
                np.float64(timestamp_value) - self._last_update[slots],
            )
            stale = inside & (age > self.stale_sec)
            self._reset_slots(safe_rows[stale], safe_columns[stale])
            inside = inside & ~stale
            decay = cp.exp(-age / self.decay_tau_sec).astype(cp.float32)
            effective_weight = self._weight_sum[slots] * decay
            valid_map = (
                inside
                & (self._last_update[slots] <= timestamp_value + 1e-6)
                & (age <= self.stale_sec)
                & (effective_weight > 1e-6)
            )
            risk_mean = cp.where(
                self._weight_sum[slots] > 0.0,
                self._risk_sum[slots]
                / cp.maximum(self._weight_sum[slots], 1e-12),
                0.0,
            )
            risk_sq_mean = cp.where(
                self._weight_sum[slots] > 0.0,
                self._risk_sq_sum[slots]
                / cp.maximum(self._weight_sum[slots], 1e-12),
                0.0,
            )
            risk_std = cp.sqrt(
                cp.maximum(risk_sq_mean - risk_mean * risk_mean, 0.0)
            )
            risk_cvar = histogram_cvar(
                cp,
                [histogram[slots] * decay for histogram in self._risk_histogram],
                effective_weight,
                self.risk_cvar_alpha,
            )
            risk = risk_mean.copy()
            confidence = self._confidence[slots] * decay
            conflict = (
                timestamp_value - self._last_conflict[slots]
            ) <= self.stale_sec
            dynamic = (
                timestamp_value - self._last_dynamic[slots]
            ) <= self.stale_sec
            use_latest = valid_map & (conflict | dynamic)
            risk[use_latest] = self._latest_risk[slots][use_latest]
            confidence[use_latest] = (
                self._latest_confidence[slots][use_latest]
                * decay[use_latest]
            )
            risk_cvar[use_latest] = cp.maximum(
                risk_cvar[use_latest], risk[use_latest]
            )
            grass_weight = self._grass_sum[slots] * decay
            effective_count = cp.rint(
                self._count_sum[slots] * decay
            ).astype(cp.int32)
        else:
            rows, columns, weights, _ = bilinear_world_xy_samples(
                world_x,
                world_y,
                center=self.rolling_grid.center,
                world_map_length=self.rolling_grid.world_map_length,
                resolution=self.rolling_grid.resolution,
                cell_n=self.rolling_grid.cell_n,
                xp=cp,
            )
            age_map = cp.maximum(
                0.0,
                np.float64(timestamp_value) - self._last_update,
            )
            support_map = (
                cp.isfinite(self._last_update)
                & (self._last_update <= timestamp_value + 1e-6)
                & (age_map <= self.stale_sec)
                & (self._weight_sum > 1e-6)
            )
            decay_map = cp.exp(-age_map / self.decay_tau_sec).astype(cp.float32)

            def sample(values, fill_value=0.0):
                return weighted_neighbor_sample(
                    values,
                    rows,
                    columns,
                    weights,
                    support=support_map,
                    min_valid_weight=self.sample_min_valid_weight,
                    xp=cp,
                    fill_value=fill_value,
                )[0]

            effective_weight = sample(self._weight_sum * decay_map)
            risk_sum = sample(self._risk_sum * decay_map)
            risk_mean = cp.where(
                effective_weight > 1e-6,
                risk_sum / cp.maximum(effective_weight, 1e-12),
                0.0,
            )
            risk_sq_sum = sample(self._risk_sq_sum * decay_map)
            risk_sq_mean = cp.where(
                effective_weight > 1e-6,
                risk_sq_sum / cp.maximum(effective_weight, 1e-12),
                0.0,
            )
            risk_std = cp.sqrt(
                cp.maximum(risk_sq_mean - risk_mean * risk_mean, 0.0)
            )
            risk_cvar = histogram_cvar(
                cp,
                [sample(histogram * decay_map) for histogram in self._risk_histogram],
                effective_weight,
                self.risk_cvar_alpha,
            )
            risk = risk_mean.copy()
            confidence = sample(self._confidence * decay_map)
            grass_weight = sample(self._grass_sum * decay_map)
            effective_count = cp.rint(
                sample(self._count_sum * decay_map)
            ).astype(cp.int32)
            latest_risk = sample(self._latest_risk)
            latest_confidence = sample(self._latest_confidence * decay_map)
            neighbor_support = support_map[rows, columns]
            effective_neighbor_weights = cp.where(
                neighbor_support,
                weights,
                0.0,
            )
            conflict_neighbors = (
                (timestamp_value - self._last_conflict[rows, columns])
                <= self.stale_sec
            )
            dynamic_neighbors = (
                (timestamp_value - self._last_dynamic[rows, columns])
                <= self.stale_sec
            )
            conflict = cp.any(
                conflict_neighbors & (effective_neighbor_weights > 0.0),
                axis=0,
            )
            dynamic = cp.any(
                dynamic_neighbors & (effective_neighbor_weights > 0.0),
                axis=0,
            )
            inside = (
                cp.sum(effective_neighbor_weights, axis=0)
                >= self.sample_min_valid_weight
            )
            valid_map = inside & (effective_weight > 1e-6)
            use_latest = valid_map & (conflict | dynamic)
            risk[use_latest] = latest_risk[use_latest]
            confidence[use_latest] = latest_confidence[use_latest]
            risk_cvar[use_latest] = cp.maximum(
                risk_cvar[use_latest], risk[use_latest]
            )
        grass_probability = cp.where(
            effective_weight > 0.0,
            grass_weight / cp.maximum(effective_weight, 1e-12),
            0.0,
        )
        effective_count[valid_map & (effective_count < 1)] = 1
        effective_count[~valid_map] = 0
        valid_count = int(cp.count_nonzero(valid_map).get())
        return {
            "semantic_hindsight_target": cp.asnumpy(
                cp.clip(risk.reshape(shape), 0.0, 1.0).astype(cp.float32)
            ),
            "semantic_hindsight_valid_mask": cp.asnumpy(valid_map.reshape(shape)),
            "semantic_hindsight_confidence": cp.asnumpy(
                cp.clip(confidence.reshape(shape), 0.0, 1.0).astype(cp.float32)
            ),
            "semantic_hindsight_observation_count": cp.asnumpy(
                effective_count.reshape(shape)
            ),
            "semantic_hindsight_risk_std": cp.asnumpy(
                cp.where(valid_map, risk_std, 0.0).reshape(shape).astype(cp.float32)
            ),
            "semantic_hindsight_risk_cvar": cp.asnumpy(
                cp.where(
                    valid_map, cp.clip(risk_cvar, 0.0, 1.0), 0.0
                ).reshape(shape).astype(cp.float32)
            ),
            "semantic_hindsight_risk_distribution_valid": cp.asnumpy(
                valid_map.reshape(shape)
            ),
            "grass_probability": cp.asnumpy(grass_probability.reshape(shape).astype(cp.float32)),
            "grass_observation_weight": cp.asnumpy(effective_weight.reshape(shape).astype(cp.float32)),
            "grass_mask": cp.asnumpy(
                (valid_map & (grass_probability > 0.6) & (effective_weight >= 1.0)).reshape(shape)
            ),
            "semantic_input_points": self._last_input_points,
            "semantic_aggregated_cells": self._last_aggregated_cells,
            "semantic_valid_cells": valid_count,
        }

    def sample_world(
        self,
        *,
        timestamp: float,
        center=None,
        world_map_length: float | None = None,
        resolution: float | None = None,
    ) -> dict[str, np.ndarray]:
        """读取共享 GPU 滚动栅格的世界坐标图，不重采样到车体系。"""

        with self.rolling_grid.lock:
            return self._sample_world_locked(
                timestamp=timestamp,
                center=center,
                world_map_length=world_map_length,
                resolution=resolution,
            )

    def _sample_world_locked(
        self,
        *,
        timestamp: float,
        center=None,
        world_map_length: float | None = None,
        resolution: float | None = None,
    ) -> dict[str, np.ndarray]:
        """读取世界图的内部实现，调用方必须持有滚动栅格锁。"""

        timestamp_value = float(timestamp)
        if not np.isfinite(timestamp_value):
            raise ValueError("timestamp must be finite")
        expected_length = self.rolling_grid.world_map_length
        expected_resolution = self.rolling_grid.resolution
        if world_map_length is not None and not np.isclose(
            float(world_map_length), expected_length
        ):
            raise ValueError("world_map_length does not match rolling grid")
        if resolution is not None and not np.isclose(
            float(resolution), expected_resolution
        ):
            raise ValueError("resolution does not match rolling grid")
        if center is not None and not np.allclose(
            np.asarray(center, dtype=np.float64).reshape(2),
            self.rolling_grid.center,
        ):
            raise ValueError("world map center does not match rolling grid")

        shape = self.rolling_grid.shape
        if not self.rolling_grid.initialized:
            return {
                "semantic_hindsight_target": np.zeros(shape, dtype=np.float32),
                "semantic_hindsight_valid_mask": np.zeros(shape, dtype=bool),
                "semantic_hindsight_confidence": np.zeros(shape, dtype=np.float32),
                "semantic_hindsight_observation_count": np.zeros(shape, dtype=np.int32),
                "grass_probability": np.zeros(shape, dtype=np.float32),
                "grass_observation_weight": np.zeros(shape, dtype=np.float32),
                "grass_mask": np.zeros(shape, dtype=bool),
                "semantic_input_points": self._last_input_points,
                "semantic_aggregated_cells": self._last_aggregated_cells,
                "semantic_valid_cells": 0,
            }

        cp = self.cp
        age = cp.maximum(0.0, np.float64(timestamp_value) - self._last_update)
        support = (
            cp.isfinite(self._last_update)
            & (self._last_update <= timestamp_value + 1e-6)
            & (age <= self.stale_sec)
            & (self._weight_sum > 1e-6)
        )
        decay = cp.exp(-age / self.decay_tau_sec).astype(cp.float32)
        effective_weight = self._weight_sum * decay
        risk_mean = cp.where(
            self._weight_sum > 0.0,
            self._risk_sum / cp.maximum(self._weight_sum, 1e-12),
            0.0,
        )
        risk_sq_mean = cp.where(
            self._weight_sum > 0.0,
            self._risk_sq_sum / cp.maximum(self._weight_sum, 1e-12),
            0.0,
        )
        risk_std = cp.sqrt(
            cp.maximum(risk_sq_mean - risk_mean * risk_mean, 0.0)
        )
        risk_cvar = histogram_cvar(
            cp,
            [histogram * decay for histogram in self._risk_histogram],
            effective_weight,
            self.risk_cvar_alpha,
        )
        risk = risk_mean.copy()
        confidence = self._confidence * decay
        conflict = (timestamp_value - self._last_conflict) <= self.stale_sec
        dynamic = (timestamp_value - self._last_dynamic) <= self.stale_sec
        use_latest = support & (conflict | dynamic)
        risk[use_latest] = self._latest_risk[use_latest]
        confidence[use_latest] = self._latest_confidence[use_latest] * decay[use_latest]
        risk_cvar[use_latest] = cp.maximum(
            risk_cvar[use_latest], risk[use_latest]
        )
        grass_weight = self._grass_sum * decay
        effective_count = cp.rint(self._count_sum * decay).astype(cp.int32)
        effective_count[support & (effective_count < 1)] = 1
        effective_count[~support] = 0
        grass_probability = cp.where(
            effective_weight > 0.0,
            grass_weight / cp.maximum(effective_weight, 1e-12),
            0.0,
        )
        return {
            "semantic_hindsight_target": cp.asnumpy(
                cp.clip(risk, 0.0, 1.0).astype(cp.float32)
            ),
            "semantic_hindsight_valid_mask": cp.asnumpy(support),
            "semantic_hindsight_confidence": cp.asnumpy(
                cp.clip(confidence, 0.0, 1.0).astype(cp.float32)
            ),
            "semantic_hindsight_observation_count": cp.asnumpy(effective_count),
            "semantic_hindsight_risk_std": cp.asnumpy(
                cp.where(support, risk_std, 0.0).astype(cp.float32)
            ),
            "semantic_hindsight_risk_cvar": cp.asnumpy(
                cp.where(support, cp.clip(risk_cvar, 0.0, 1.0), 0.0).astype(
                    cp.float32
                )
            ),
            "semantic_hindsight_risk_distribution_valid": cp.asnumpy(support),
            "grass_probability": cp.asnumpy(grass_probability.astype(cp.float32)),
            "grass_observation_weight": cp.asnumpy(effective_weight.astype(cp.float32)),
            "grass_mask": cp.asnumpy(
                support & (grass_probability > 0.6) & (effective_weight >= 1.0)
            ),
            "semantic_input_points": self._last_input_points,
            "semantic_aggregated_cells": self._last_aggregated_cells,
            "semantic_valid_cells": int(cp.count_nonzero(support).get()),
        }

    def advance(self, **kwargs) -> dict[str, np.ndarray]:
        self.update(**kwargs)
        return self.sample(
            timestamp=kwargs["timestamp"],
            world_from_body=kwargs["world_from_body"],
        )
