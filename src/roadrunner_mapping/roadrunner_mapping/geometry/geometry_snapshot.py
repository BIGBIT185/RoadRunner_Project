"""Deterministic packed geometry inputs shared by CPU and CUDA fitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

from .coordinates import body_xy_to_grid, transform_points, validate_transform
from .ground_fit import GroundFitContract, fit_ground_plane_irls_cpu
from .schema import MapSpec


VoxelKey = Tuple[int, int, int]


@dataclass(frozen=True)
class GrassGeometryConfig:
    grass_class_id: int = 3
    grass_probability_threshold: float = 0.5
    grass_min_semantic_weight: float = 1.5
    grass_positive_step_scale: float = 0.2
    grass_negative_min_ground_confidence: float = 0.6
    grass_negative_min_support_count: int = 5
    grass_prior_min_risk: float = 0.15
    grass_prior_max_risk: float = 0.40
    grass_density_low: float = 0.30
    grass_density_high: float = 0.70

    def __post_init__(self) -> None:
        unit_interval = (
            self.grass_probability_threshold,
            self.grass_positive_step_scale,
            self.grass_negative_min_ground_confidence,
            self.grass_prior_min_risk,
            self.grass_prior_max_risk,
            self.grass_density_low,
            self.grass_density_high,
        )
        if self.grass_class_id < 0:
            raise ValueError("grass_class_id must be non-negative")
        if not np.isfinite(unit_interval).all() or any(
            value < 0.0 or value > 1.0 for value in unit_interval
        ):
            raise ValueError("grass geometry probabilities must be in [0,1]")
        if self.grass_min_semantic_weight < 0.0 or not np.isfinite(
            self.grass_min_semantic_weight
        ):
            raise ValueError("grass_min_semantic_weight must be finite and non-negative")
        if self.grass_negative_min_support_count < 0:
            raise ValueError("grass_negative_min_support_count must be non-negative")
        if self.grass_prior_max_risk < self.grass_prior_min_risk:
            raise ValueError("grass prior risk bounds must be increasing")
        if self.grass_density_high < self.grass_density_low:
            raise ValueError("grass density bounds must be increasing")


DEFAULT_GRASS_GEOMETRY_CONFIG = GrassGeometryConfig()


@dataclass(frozen=True)
class PackedGeometrySnapshot:
    timestamp: float
    map_spec: MapSpec
    elevation: np.ndarray
    observation_count: np.ndarray
    last_observed_time: np.ndarray
    geometry_age: np.ndarray
    conflict_score: np.ndarray
    semantic_sum: np.ndarray
    semantic_weight: np.ndarray
    cell_offsets: np.ndarray
    support_points: np.ndarray
    support_weights: np.ndarray

    def cell_slice(self, row: int, column: int) -> slice:
        flat_index = int(row) * self.map_spec.shape[1] + int(column)
        return slice(
            int(self.cell_offsets[flat_index]),
            int(self.cell_offsets[flat_index + 1]),
        )

    def cell_points(self, row: int, column: int) -> np.ndarray:
        return self.support_points[self.cell_slice(row, column)]

    def cell_weights(self, row: int, column: int) -> np.ndarray:
        return self.support_weights[self.cell_slice(row, column)]


def pack_geometry_snapshot(
    voxel_snapshot: Mapping[VoxelKey, object],
    *,
    voxel_size: float,
    semantic_class_count: int,
    timestamp: float,
    world_from_body: np.ndarray,
    map_spec: MapSpec,
    min_geometry_reliability: float,
    max_geometry_age_sec: float,
    min_observation_count: int,
    max_conflict_score: float,
    min_elevation_m: float,
    max_elevation_m: float,
) -> PackedGeometrySnapshot:
    """Pack a voxel snapshot in sorted-key and row-major cell order."""

    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if semantic_class_count <= 0:
        raise ValueError("semantic_class_count must be positive")
    if not np.isfinite(timestamp):
        raise ValueError("snapshot timestamp must be finite")
    if min_elevation_m >= max_elevation_m:
        raise ValueError("elevation bounds must be increasing")
    pose = validate_transform(world_from_body)
    target_from_world = np.linalg.inv(pose)
    height, width = map_spec.shape
    shape = (height, width)
    elevation_min = np.full(shape, np.inf, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int32)
    last_observed = np.full(shape, -1.0, dtype=np.float64)
    age_layer = np.zeros(shape, dtype=np.float32)
    conflict = np.zeros(shape, dtype=np.float64)
    semantic_sum = np.zeros(
        (height, width, semantic_class_count), dtype=np.float64
    )
    semantic_weight = np.zeros(shape, dtype=np.float64)
    points_by_cell: list[list[np.ndarray]] = [
        [] for _ in range(height * width)
    ]
    weights_by_cell: list[list[float]] = [
        [] for _ in range(height * width)
    ]

    keys = sorted(voxel_snapshot)
    if keys:
        grid_centers_world = (np.asarray(keys, dtype=np.float64) + 0.5) * voxel_size
        grid_centers_body = transform_points(grid_centers_world, target_from_world)
        rows, columns, inside = body_xy_to_grid(
            grid_centers_body[:, :2], map_spec
        )
        for index, key in enumerate(keys):
            if not inside[index]:
                continue
            state = voxel_snapshot[key]
            row = int(rows[index])
            column = int(columns[index])
            center_world = grid_centers_world[index].copy()
            center_world[2] = float(state.elevation_mean)
            center_body = transform_points(center_world[None], target_from_world)[0]
            elevation = float(center_body[2])
            if elevation < min_elevation_m or elevation > max_elevation_m:
                continue
            elevation_min[row, column] = min(elevation_min[row, column], elevation)
            age = max(0.0, float(timestamp) - float(state.last_observed_time))
            support_ok = (
                state.reliability >= min_geometry_reliability
                and age <= max_geometry_age_sec
                and not state.dynamic
                and state.observation_count >= min_observation_count
                and state.conflict_score <= max_conflict_score
            )
            flat_index = row * width + column
            if support_ok:
                freshness = np.exp(-age / max(max_geometry_age_sec, 1e-6))
                weight = (
                    max(float(state.reliability), 1e-6)
                    * freshness
                    * max(float(state.ground_probability), 0.05)
                    * max(float(state.observation_count), 1.0)
                )
                points_by_cell[flat_index].append(center_body)
                weights_by_cell[flat_index].append(float(weight))
            count[row, column] += int(state.observation_count)
            conflict[row, column] = max(
                conflict[row, column], float(state.conflict_score)
            )
            last_observed[row, column] = max(
                last_observed[row, column], float(state.last_observed_time)
            )
            semantic_state = np.asarray(state.semantic_probabilities, dtype=np.float64)
            if semantic_state.shape != (semantic_class_count,):
                raise ValueError("voxel semantic class count does not match snapshot")
            semantic_state_weight = max(float(state.reliability), 1e-6)
            semantic_sum[row, column] += semantic_state * semantic_state_weight
            semantic_weight[row, column] += semantic_state_weight

    observed = count > 0
    elevation = np.zeros(shape, dtype=np.float32)
    elevation[observed] = elevation_min[observed].astype(np.float32)
    age_layer[observed] = (
        float(timestamp) - last_observed[observed]
    ).clip(min=0.0).astype(np.float32)
    cell_offsets = np.zeros(height * width + 1, dtype=np.int64)
    for index, cell_points in enumerate(points_by_cell):
        cell_offsets[index + 1] = cell_offsets[index] + len(cell_points)
    support_count = int(cell_offsets[-1])
    support_points = np.empty((support_count, 3), dtype=np.float64)
    support_weights = np.empty(support_count, dtype=np.float64)
    for index, cell_points in enumerate(points_by_cell):
        start = int(cell_offsets[index])
        end = int(cell_offsets[index + 1])
        if start == end:
            continue
        support_points[start:end] = np.asarray(cell_points, dtype=np.float64)
        support_weights[start:end] = np.asarray(
            weights_by_cell[index], dtype=np.float64
        )

    return PackedGeometrySnapshot(
        timestamp=float(timestamp),
        map_spec=map_spec,
        elevation=elevation,
        observation_count=count,
        last_observed_time=last_observed,
        geometry_age=age_layer,
        conflict_score=conflict.astype(np.float32),
        semantic_sum=semantic_sum,
        semantic_weight=semantic_weight,
        cell_offsets=cell_offsets,
        support_points=support_points,
        support_weights=support_weights,
    )


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return float(value >= edge1)
    x = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def _combine_geometry_risks(
    slope_risk: float,
    roughness_risk: float,
    positive_step_risk: float,
    negative_step_risk: float,
) -> float:
    """Combine physical blocking hazards while retaining roughness as diagnostics."""

    del roughness_risk
    return float(max(slope_risk, positive_step_risk, negative_step_risk))


def apply_grass_geometry_risk(
    *,
    grass_mask: np.ndarray,
    slope_risk: np.ndarray,
    positive_step_risk: np.ndarray,
    negative_step_risk: np.ndarray,
    non_ground_ratio: np.ndarray,
    ground_confidence: np.ndarray,
    ground_support_count: np.ndarray,
    config: GrassGeometryConfig = DEFAULT_GRASS_GEOMETRY_CONFIG,
) -> dict[str, np.ndarray]:
    """Apply the shared semantic grass rule after CPU or CUDA geometry fitting."""

    raw_mask = np.asarray(grass_mask)
    shape = raw_mask.shape
    if raw_mask.ndim != 2:
        raise ValueError("grass geometry layers must be 2D")

    def finite_unit(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError("grass geometry layers must share a shape")
        return np.clip(
            np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

    if raw_mask.dtype == bool:
        explicit_grass = raw_mask.copy()
    else:
        explicit_grass = np.isfinite(raw_mask) & (raw_mask > 0.5)
    slope = finite_unit(slope_risk)
    positive = finite_unit(positive_step_risk)
    negative = finite_unit(negative_step_risk)
    density = finite_unit(non_ground_ratio)
    confidence = np.asarray(ground_confidence, dtype=np.float32)
    support = np.asarray(ground_support_count, dtype=np.float32)
    if confidence.shape != shape or support.shape != shape:
        raise ValueError("grass geometry layers must share a shape")
    confidence = np.clip(
        np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    support = np.maximum(
        np.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )

    original = np.maximum.reduce((slope, positive, negative))
    grass_positive = np.where(
        explicit_grass,
        config.grass_positive_step_scale * positive,
        0.0,
    ).astype(np.float32)
    negative_valid = (
        explicit_grass
        & (confidence > config.grass_negative_min_ground_confidence)
        & (support >= config.grass_negative_min_support_count)
    )
    grass_negative = np.where(negative_valid, negative, 0.0).astype(np.float32)

    denominator = max(
        config.grass_density_high - config.grass_density_low,
        float(np.finfo(np.float32).eps),
    )
    density_t = np.clip(
        (density - config.grass_density_low) / denominator,
        0.0,
        1.0,
    )
    grass_density = np.where(explicit_grass, density_t, 0.0).astype(np.float32)
    prior = config.grass_prior_min_risk + density_t * (
        config.grass_prior_max_risk - config.grass_prior_min_risk
    )
    grass_prior = np.where(explicit_grass, prior, 0.0).astype(np.float32)
    grass_geometry = np.maximum.reduce(
        (slope, grass_positive, grass_negative, prior.astype(np.float32))
    )
    geometry = np.clip(
        np.where(explicit_grass, grass_geometry, original),
        0.0,
        1.0,
    ).astype(np.float32)
    return {
        "geometry_risk": geometry,
        "grass_mask": explicit_grass,
        "grass_density_risk": grass_density,
        "grass_prior_risk": grass_prior,
        "grass_positive_step_risk": grass_positive,
        "grass_negative_step_risk": grass_negative,
        "grass_negative_step_valid": negative_valid,
    }


def _quantile(values: np.ndarray, quantile: float) -> float:
    if values.size == 0:
        raise ValueError("quantile requires values")
    ordered = np.sort(values)
    index = int(np.searchsorted(np.arange(1, ordered.size + 1), quantile * ordered.size))
    return float(ordered[min(index, ordered.size - 1)])


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError("weighted quantile requires matching non-empty vectors")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    if np.any(weights < 0.0) or not np.isfinite(weights).all():
        raise ValueError("quantile weights must be finite and non-negative")
    order = np.argsort(values)
    ordered = values[order]
    ordered_weights = weights[order]
    total = float(np.sum(ordered_weights))
    if total <= 0.0:
        return _quantile(values, quantile)
    index = int(
        np.searchsorted(
            np.cumsum(ordered_weights), quantile * total, side="left"
        )
    )
    return float(ordered[min(index, ordered.size - 1)])


def fit_packed_geometry_cpu(
    packed: PackedGeometrySnapshot,
    *,
    neighborhood_cells: int = 7,
    min_support_points: int = 4,
    ground_fit_max_iterations: int = 10,
    ground_fit_residual_scale: float = 0.10,
    max_fit_residual: float = 0.20,
    max_geometry_age_sec: float = 5.0,
    slope_safe_deg: float = 8.0,
    slope_critical_deg: float = 20.0,
    lateral_slope_safe_deg: float = 6.0,
    lateral_slope_critical_deg: float = 15.0,
    roughness_safe_m: float = 0.03,
    roughness_critical_m: float = 0.12,
    positive_step_safe_m: float = 0.05,
    positive_step_critical_m: float = 0.25,
    negative_step_safe_m: float = 0.05,
    negative_step_critical_m: float = 0.25,
    min_elevation_m: float = -2.0,
    max_elevation_m: float = 3.0,
) -> dict[str, np.ndarray]:
    """Reference fitter over the exact packed representation used by CUDA."""

    height, width = packed.map_spec.shape
    shape = (height, width)
    observed = packed.observation_count > 0
    output = {
        name: np.zeros(shape, dtype=np.float32)
        for name in (
            "geometry_risk", "geometry_confidence", "slope",
            "longitudinal_slope", "lateral_slope", "roughness",
            "positive_step_height", "negative_step_height", "non_ground_ratio",
            "ground_fit_residual", "slope_risk", "roughness_risk",
            "positive_step_risk", "negative_step_risk",
        )
    }
    support_count = np.zeros(shape, dtype=np.int32)
    fit_valid = np.zeros(shape, dtype=bool)
    radius = max(0, int(neighborhood_cells) // 2)
    slope_safe, slope_critical = np.deg2rad([slope_safe_deg, slope_critical_deg])
    lateral_safe, lateral_critical = np.deg2rad(
        [lateral_slope_safe_deg, lateral_slope_critical_deg]
    )
    contract = GroundFitContract(
        max_iterations=max(1, int(ground_fit_max_iterations)),
        residual_scale=max(float(ground_fit_residual_scale), 1e-6),
    )
    elevation = packed.elevation.copy()
    for row, column in zip(*np.nonzero(observed)):
        neighbors, weights = [], []
        for rr in range(max(0, row - radius), min(height, row + radius + 1)):
            for cc in range(max(0, column - radius), min(width, column + radius + 1)):
                points = packed.cell_points(rr, cc)
                base_weights = packed.cell_weights(rr, cc)
                if points.size == 0:
                    continue
                distance = np.linalg.norm([rr - row, cc - column])
                factor = np.exp(-(distance / max(radius, 1)) ** 2)
                neighbors.extend(points)
                weights.extend(base_weights * factor)
        support_count[row, column] = len(neighbors)
        if len(neighbors) < min_support_points:
            continue
        points = np.asarray(neighbors, dtype=np.float64)
        coeffs, _, residual, valid = fit_ground_plane_irls_cpu(
            points, np.asarray(weights), contract=contract
        )
        if not valid or residual > max_fit_residual:
            continue
        a, b, c = coeffs
        center_x = packed.map_spec.x_max - (row + 0.5) * packed.map_spec.resolution
        center_y = packed.map_spec.y_max - (column + 0.5) * packed.map_spec.resolution
        fitted = float(a * center_x + b * center_y + c)
        if not min_elevation_m <= fitted <= max_elevation_m:
            continue
        cell_points = packed.cell_points(row, column)
        cell_weights = packed.cell_weights(row, column)
        if cell_points.size == 0:
            continue
        cell_residual = cell_points[:, 2] - (
            a * cell_points[:, 0] + b * cell_points[:, 1] + c
        )
        median = _weighted_quantile(cell_residual, cell_weights, 0.5)
        mad = 1.4826 * _weighted_quantile(
            np.abs(cell_residual - median), cell_weights, 0.5
        )
        pos = max(0.0, _weighted_quantile(cell_residual, cell_weights, 0.95))
        neg = max(0.0, _weighted_quantile(-cell_residual, cell_weights, 0.95))
        weight_sum = max(float(np.sum(cell_weights)), 1e-12)
        slope = float(np.arctan(np.sqrt(a * a + b * b)))
        longitudinal = float(np.arctan(abs(a)))
        lateral = float(np.arctan(abs(b)))
        risks = (
            max(_smoothstep(slope_safe, slope_critical, slope),
                _smoothstep(lateral_safe, lateral_critical, lateral)),
            _smoothstep(roughness_safe_m, roughness_critical_m, mad),
            _smoothstep(positive_step_safe_m, positive_step_critical_m, pos),
            _smoothstep(negative_step_safe_m, negative_step_critical_m, neg),
        )
        elevation[row, column] = fitted
        for name, value in (
            ("slope", slope), ("longitudinal_slope", longitudinal),
            ("lateral_slope", lateral), ("roughness", mad),
            ("positive_step_height", pos), ("negative_step_height", neg),
            ("non_ground_ratio", max(
                float(np.sum(cell_weights[cell_residual > positive_step_safe_m]))
                / weight_sum,
                float(np.sum(cell_weights[cell_residual < -negative_step_safe_m]))
                / weight_sum,
            )),
            ("ground_fit_residual", residual), ("slope_risk", risks[0]),
            ("roughness_risk", risks[1]), ("positive_step_risk", risks[2]),
            ("negative_step_risk", risks[3]),
            ("geometry_risk", _combine_geometry_risks(*risks)),
        ):
            output[name][row, column] = value
        confidence = (
            (1.0 - np.exp(-len(neighbors) / 12.0))
            * np.exp(-packed.geometry_age[row, column] / max(max_geometry_age_sec, 1e-6))
            * np.exp(-residual / max(ground_fit_residual_scale, 1e-6))
            / (1.0 + packed.conflict_score[row, column])
        )
        output["geometry_confidence"][row, column] = np.clip(confidence, 0.0, 1.0)
        fit_valid[row, column] = True
    semantic_ids = np.full(shape, -1, dtype=np.int32)
    semantic_confidence = np.zeros(shape, dtype=np.float32)
    semantic_valid = packed.semantic_weight > 0.0
    probabilities = np.zeros_like(packed.semantic_sum)
    probabilities[semantic_valid] = (
        packed.semantic_sum[semantic_valid] / packed.semantic_weight[semantic_valid, None]
    )
    semantic_ids[semantic_valid] = np.argmax(probabilities[semantic_valid], axis=1)
    semantic_confidence[semantic_valid] = np.max(probabilities[semantic_valid], axis=1)
    output.update({
        "elevation": elevation, "geometry_reliability": output["geometry_confidence"],
        "semantic_ids": semantic_ids, "semantic_confidence": semantic_confidence,
        "observation_count": packed.observation_count,
        "last_observed_time": packed.last_observed_time.astype(np.float32),
        "observed_mask": observed,
        "geometry_valid_mask": (output["geometry_confidence"] > 0).astype(np.float32),
        "geometry_support_count": support_count,
        "geometry_age": packed.geometry_age,
        "geometry_conflict_score": packed.conflict_score,
        "ground_fit_valid": fit_valid.astype(np.float32),
        "geometry_version": np.asarray("v2_local_ground_multi_factor"),
    })
    return output
