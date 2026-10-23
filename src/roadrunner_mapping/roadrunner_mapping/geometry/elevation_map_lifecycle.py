"""Backend-independent elevation-map lifecycle primitives."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ElevationMapLifecycleConfig:
    """Validated configuration for elevation-map lifecycle updates."""

    enabled: bool = False
    enable_visibility_cleanup: bool = True
    enable_upper_bound: bool = True
    enable_height_drift_compensation: bool = True
    enable_overhead_filter: bool = True
    enable_wall_edge_sharpening: bool = True
    enable_overlap_clearance: bool = True
    enable_continuous_validity: bool = True
    overhead_slope: float = 0.30
    overhead_distance_offset_m: float = 1.0
    overhead_height_intercept_m: float = 0.20
    overhead_height_cap_m: float = 2.0
    drift_validity_min: float = 0.8
    drift_variance_max: float = 0.01
    drift_min_frames: int = 3
    drift_slope_max_deg: float = 8.0
    drift_residual_inlier_m: float = 0.15
    drift_min_cells: int = 30
    drift_alpha: float = 0.2
    drift_max_correction_per_frame_m: float = 0.05
    wall_point_count_threshold: int = 20
    wall_min_vertical_span_m: float = 0.30
    wall_lower_point_tolerance_m: float = 0.05
    max_visibility_rays_per_frame: int = 4096
    visibility_max_ray_length_m: float = 40.0
    visibility_endpoint_margin_m: float = 0.30
    visibility_sigma_factor: float = 1.0
    visibility_height_margin_m: float = 0.01
    visibility_min_age_sec: float = 0.5
    visibility_cleanup_step: float = 0.2
    visibility_cleanup_cos_threshold: float = 0.5
    elevation_validity_threshold: float = 0.5
    upper_bound_max_age_sec: float = 10.0
    overlap_clear_radius_m: float = 2.0
    overlap_clear_range_z_m: float = 2.0

    def __post_init__(self) -> None:
        boolean_names = (
            "enabled",
            "enable_visibility_cleanup",
            "enable_upper_bound",
            "enable_height_drift_compensation",
            "enable_overhead_filter",
            "enable_wall_edge_sharpening",
            "enable_overlap_clearance",
            "enable_continuous_validity",
        )
        for name in boolean_names:
            if not isinstance(getattr(self, name), bool):
                raise TypeError("{} must be a boolean".format(name))

        integer_names = (
            "drift_min_frames",
            "drift_min_cells",
            "wall_point_count_threshold",
            "max_visibility_rays_per_frame",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError("{} must be a positive integer".format(name))
            if value <= 0:
                raise ValueError("{} must be a positive integer".format(name))

        real_names = tuple(
            name
            for name in self.__dataclass_fields__
            if name not in boolean_names and name not in integer_names
        )
        for name in real_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError("{} must be a finite real number".format(name))
            if not math.isfinite(float(value)):
                raise ValueError("{} must be finite".format(name))

        nonnegative_names = (
            "overhead_slope",
            "overhead_distance_offset_m",
            "wall_lower_point_tolerance_m",
            "visibility_endpoint_margin_m",
            "visibility_sigma_factor",
            "visibility_height_margin_m",
            "visibility_min_age_sec",
            "overlap_clear_radius_m",
        )
        for name in nonnegative_names:
            if getattr(self, name) < 0.0:
                raise ValueError("{} must be nonnegative".format(name))

        positive_names = (
            "overhead_height_intercept_m",
            "drift_variance_max",
            "drift_residual_inlier_m",
            "drift_max_correction_per_frame_m",
            "wall_min_vertical_span_m",
            "visibility_max_ray_length_m",
            "upper_bound_max_age_sec",
            "overlap_clear_range_z_m",
        )
        for name in positive_names:
            if getattr(self, name) <= 0.0:
                raise ValueError("{} must be positive".format(name))

        # GpuRollingElevationMap validates drift_variance_max against Gaussian max_variance at construction.
        for name in ("drift_validity_min", "visibility_cleanup_cos_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("{} must be within [0, 1]".format(name))

        for name in ("drift_alpha", "visibility_cleanup_step"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError("{} must be within (0, 1]".format(name))

        if not 0.0 < self.elevation_validity_threshold < 1.0:
            raise ValueError("elevation_validity_threshold must be within (0, 1)")
        if not 0.0 < self.drift_slope_max_deg < 90.0:
            raise ValueError("drift_slope_max_deg must be within (0, 90)")
        if self.overhead_height_cap_m < self.overhead_height_intercept_m:
            raise ValueError("overhead height cap must not be below the intercept")
        if self.visibility_endpoint_margin_m >= self.visibility_max_ray_length_m:
            raise ValueError("visibility endpoint margin must be below ray length")


@dataclass(frozen=True)
class HeightDriftEstimate:
    """One-frame estimate of the persistent height bias."""

    new_bias: float
    correction: float
    sample_count: int


@dataclass(frozen=True)
class WallEdgeSelection:
    """Point selection produced by wall-edge sharpening."""

    keep_mask: Any
    ignored_mask: Any
    sharpened_cell_ids: Any


@dataclass(frozen=True)
class VisibilityRaySamples:
    """Grid cells and ray geometry retained by visibility sampling."""

    cell_ids: Any
    heights: Any
    directions: Any
    ray_indices: Any
    distances: Any


@dataclass(frozen=True)
class VisibilityEvidence:
    """Per-frame visibility evidence derived from an immutable map snapshot."""

    upper_bound_cell_ids: Any
    upper_bound_candidates: Any
    penetrated_cell_ids: Any


@dataclass(frozen=True)
class VisibilityUpdateResult:
    """Cell indices changed while committing one frame of visibility state."""

    validity_updated_cell_ids: Any
    cleared_cell_ids: Any
    upper_bound_updated_cell_ids: Any
    accepted_cell_ids: Any


def _as_calculation_float(xp, values, name):
    values = xp.asarray(values)
    if values.dtype.kind not in "fiu" or values.dtype.kind == "b":
        raise TypeError("{} must have a real numeric dtype".format(name))
    return values.astype(xp.result_type(values.dtype, xp.float32), copy=False)


def _device_error_flags(xp, *conditions):
    """Evaluate device predicates with one device-to-host synchronization."""

    flags = xp.stack(tuple(xp.any(condition) for condition in conditions))
    return tuple(bool(value) for value in flags.tolist())


def _validate_positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("{} must be a finite positive real number".format(name))
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return value


def _empty_visibility_ray_samples(xp, float_dtype):
    return VisibilityRaySamples(
        cell_ids=xp.empty((0,), dtype=xp.int64),
        heights=xp.empty((0,), dtype=float_dtype),
        directions=xp.empty((0, 3), dtype=float_dtype),
        ray_indices=xp.empty((0,), dtype=xp.int64),
        distances=xp.empty((0,), dtype=float_dtype),
    )


def _empty_visibility_evidence(xp, float_dtype):
    return VisibilityEvidence(
        upper_bound_cell_ids=xp.empty((0,), dtype=xp.int64),
        upper_bound_candidates=xp.empty((0,), dtype=float_dtype),
        penetrated_cell_ids=xp.empty((0,), dtype=xp.int64),
    )


def _empty_visibility_update_result(xp):
    empty = xp.empty((0,), dtype=xp.int64)
    return VisibilityUpdateResult(
        validity_updated_cell_ids=empty,
        cleared_cell_ids=empty.copy(),
        upper_bound_updated_cell_ids=empty.copy(),
        accepted_cell_ids=empty.copy(),
    )


def _require_mutable_backend_array(xp, values, name):
    if not isinstance(values, xp.ndarray):
        raise TypeError("{} must be a mutable backend array".format(name))
    flags = getattr(values, "flags", None)
    if flags is not None and getattr(flags, "writeable", True) is False:
        raise ValueError("{} must be writable".format(name))
    return values


def _require_float_scalar_representable(xp, value, target, name):
    dtype_info = xp.finfo(target.dtype)
    limit = float(dtype_info.max)
    smallest = float(dtype_info.smallest_subnormal)
    if value < -limit or value > limit:
        raise ValueError("{} must be representable by target dtype".format(name))
    if value != 0.0 and abs(value) < smallest:
        raise ValueError("{} must be representable by target dtype".format(name))
    return value


def select_visibility_endpoints(xp, points_world, max_rays):
    """Select an ordered, deterministic, evenly spaced endpoint subset."""

    points = xp.asarray(points_world)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape [N,3]")
    if points.dtype.kind not in "fiu" or points.dtype.kind == "b":
        raise TypeError("points_world must have a real numeric dtype")
    if isinstance(max_rays, bool) or not isinstance(max_rays, numbers.Integral):
        raise TypeError("max_rays must be a positive integer")
    if max_rays <= 0:
        raise ValueError("max_rays must be a positive integer")
    if _device_error_flags(xp, ~xp.isfinite(points))[0]:
        raise ValueError("points_world must be finite")
    if points.shape[0] <= max_rays:
        return points

    indices = xp.linspace(
        0,
        points.shape[0] - 1,
        num=int(max_rays),
        dtype=xp.int64,
    )
    return points[indices]


def raycast_grid_samples(
    xp,
    sensor_origin,
    endpoints,
    resolution,
    max_length,
    endpoint_margin,
    world_xy_to_flat_ids,
):
    """Sample visibility rays in batches and retain one sample per run of cells."""

    origin = xp.asarray(sensor_origin)
    endpoints = xp.asarray(endpoints)
    if origin.ndim != 1 or origin.shape[0] != 3:
        raise ValueError("sensor_origin must have shape [3]")
    if endpoints.ndim != 2 or endpoints.shape[1] != 3:
        raise ValueError("endpoints must have shape [M,3]")
    if origin.dtype.kind not in "fiu" or origin.dtype.kind == "b":
        raise TypeError("sensor_origin must have a real numeric dtype")
    if endpoints.dtype.kind not in "fiu" or endpoints.dtype.kind == "b":
        raise TypeError("endpoints must have a real numeric dtype")

    resolution = _validate_positive_real(resolution, "resolution")
    max_length = _validate_positive_real(max_length, "max_length")
    if isinstance(endpoint_margin, bool) or not isinstance(
        endpoint_margin, numbers.Real
    ):
        raise TypeError("endpoint_margin must be a finite real number")
    endpoint_margin = float(endpoint_margin)
    if not math.isfinite(endpoint_margin):
        raise ValueError("endpoint_margin must be finite")
    if endpoint_margin < 0.0 or endpoint_margin >= max_length:
        raise ValueError(
            "endpoint_margin must be nonnegative and below max_length"
        )
    if not callable(world_xy_to_flat_ids):
        raise TypeError("world_xy_to_flat_ids must be callable")

    origin_nonfinite, endpoints_nonfinite = _device_error_flags(
        xp,
        ~xp.isfinite(origin),
        ~xp.isfinite(endpoints),
    )
    if origin_nonfinite:
        raise ValueError("sensor_origin must be finite")
    if endpoints_nonfinite:
        raise ValueError("endpoints must be finite")

    float_dtype = xp.result_type(origin.dtype, endpoints.dtype, xp.float32)
    origin = origin.astype(float_dtype, copy=False)
    endpoints = endpoints.astype(float_dtype, copy=False)
    if endpoints.shape[0] == 0:
        return _empty_visibility_ray_samples(xp, float_dtype)

    step = resolution / math.sqrt(2.0)
    samples_per_ray = int(math.ceil(max_length / step))
    distances = xp.arange(samples_per_ray, dtype=float_dtype) * step
    deltas = endpoints - origin[None, :]
    lengths = xp.sqrt(xp.sum(deltas * deltas, axis=1))
    safe_lengths = xp.where(lengths > 0.0, lengths, 1.0)
    directions = deltas / safe_lengths[:, None]
    sample_points = (
        origin[None, None, :]
        + directions[:, None, :] * distances[None, :, None]
    )

    callback_result = world_xy_to_flat_ids(
        sample_points[:, :, 0],
        sample_points[:, :, 1],
    )
    if not isinstance(callback_result, (tuple, list)) or len(callback_result) != 2:
        raise TypeError(
            "world_xy_to_flat_ids must return (cell_ids, in_map_mask)"
        )
    cell_ids = xp.asarray(callback_result[0])
    in_map = xp.asarray(callback_result[1])
    expected_shape = (endpoints.shape[0], samples_per_ray)
    if cell_ids.shape != expected_shape or in_map.shape != expected_shape:
        raise ValueError("grid callback outputs must match sampled ray shape")
    if cell_ids.dtype.kind != "i":
        raise TypeError("grid callback cell_ids must have a signed integer dtype")
    if in_map.dtype.kind != "b":
        raise TypeError("grid callback in-map mask must have boolean dtype")

    ray_limits = xp.where(
        lengths > max_length,
        max_length,
        lengths - endpoint_margin,
    )
    retained = (
        (lengths[:, None] > 0.0)
        & (distances[None, :] < ray_limits[:, None])
        & in_map
    )
    deduplicated = retained.copy()
    deduplicated[:, 1:] &= (
        ~retained[:, :-1] | (cell_ids[:, 1:] != cell_ids[:, :-1])
    )
    flat_retained = deduplicated.reshape(-1)
    retained_cell_ids = cell_ids.reshape(-1)[flat_retained]
    all_ray_indices = xp.repeat(
        xp.arange(endpoints.shape[0], dtype=xp.int64),
        samples_per_ray,
    )
    retained_ray_indices = all_ray_indices[flat_retained]
    retained_distances = xp.broadcast_to(
        distances[None, :], expected_shape
    ).reshape(-1)[flat_retained]
    retained_heights = sample_points[:, :, 2].reshape(-1)[flat_retained]

    if retained_cell_ids.size == 0:
        return _empty_visibility_ray_samples(xp, float_dtype)
    return VisibilityRaySamples(
        cell_ids=retained_cell_ids,
        heights=retained_heights,
        directions=directions[retained_ray_indices],
        ray_indices=retained_ray_indices,
        distances=retained_distances,
    )


def _validated_map_shape(map_shape):
    if not isinstance(map_shape, (tuple, list)) or len(map_shape) != 2:
        raise TypeError("map_shape must be a two-element tuple or list")
    shape = []
    for value in map_shape:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError("map_shape entries must be positive integers")
        if value <= 0:
            raise ValueError("map_shape entries must be positive integers")
        shape.append(int(value))
    return tuple(shape)


def _validated_visibility_samples(xp, samples, cell_count):
    if not isinstance(samples, VisibilityRaySamples):
        raise TypeError("samples must be VisibilityRaySamples")
    cell_ids = xp.asarray(samples.cell_ids)
    heights = _as_calculation_float(xp, samples.heights, "sample heights")
    directions = _as_calculation_float(
        xp, samples.directions, "sample directions"
    )
    ray_indices = xp.asarray(samples.ray_indices)
    distances = _as_calculation_float(
        xp, samples.distances, "sample distances"
    )
    if cell_ids.ndim != 1 or cell_ids.dtype.kind != "i":
        raise TypeError("sample cell_ids must be a one-dimensional signed integer array")
    cell_ids = cell_ids.astype(xp.int64, copy=False)
    sample_count = cell_ids.size
    if heights.shape != (sample_count,):
        raise ValueError("sample heights must match cell_ids")
    if directions.shape != (sample_count, 3):
        raise ValueError("sample directions must have shape [N,3]")
    if ray_indices.shape != (sample_count,) or ray_indices.dtype.kind != "i":
        raise TypeError("sample ray_indices must be a matching signed integer array")
    ray_indices = ray_indices.astype(xp.int64, copy=False)
    if distances.shape != (sample_count,):
        raise ValueError("sample distances must match cell_ids")

    invalid_samples = _device_error_flags(
        xp,
        (~xp.isfinite(heights))
        | (~xp.all(xp.isfinite(directions), axis=1))
        | (~xp.isfinite(distances))
        | (distances < 0.0)
        | (ray_indices < 0)
        | (cell_ids < 0)
        | (cell_ids >= cell_count),
    )[0]
    if invalid_samples:
        raise ValueError("visibility samples contain invalid values")
    return cell_ids, heights, directions


def _validated_visibility_map(
    xp,
    elevation_mean,
    elevation_variance,
    elevation_valid,
    last_seen,
    cell_count,
):
    mean = _as_calculation_float(xp, elevation_mean, "elevation_mean")
    variance = _as_calculation_float(
        xp, elevation_variance, "elevation_variance"
    )
    valid = xp.asarray(elevation_valid)
    seen = _as_calculation_float(xp, last_seen, "last_seen")
    for name, values in (
        ("elevation_mean", mean),
        ("elevation_variance", variance),
        ("elevation_valid", valid),
        ("last_seen", seen),
    ):
        if values.shape != (cell_count,):
            raise ValueError("{} must have shape [{}]".format(name, cell_count))
    if valid.dtype.kind != "b":
        raise TypeError("elevation_valid must have a boolean dtype")

    invalid_state, invalid_variance, invalid_seen = _device_error_flags(
        xp,
        ~xp.isfinite(mean),
        (~xp.isfinite(variance)) | (variance < 0.0),
        xp.isnan(seen) | xp.isposinf(seen),
    )
    if invalid_state:
        raise ValueError("elevation_mean must be finite")
    if invalid_variance:
        raise ValueError("elevation_variance must be finite and nonnegative")
    if invalid_seen:
        raise ValueError("last_seen may contain finite values or -inf only")
    return mean, variance, valid, seen


def _minimum_upper_bound_candidates(xp, cell_ids, heights, cell_count):
    if cell_ids.size == 0:
        return cell_ids.astype(xp.int64, copy=False), heights
    minimum_by_cell = xp.full(cell_count, xp.inf, dtype=heights.dtype)
    xp.minimum.at(minimum_by_cell, cell_ids, heights)
    unique_cells = xp.flatnonzero(xp.isfinite(minimum_by_cell)).astype(
        xp.int64, copy=False
    )
    return unique_cells, minimum_by_cell[unique_cells]


def visibility_evidence(
    xp,
    samples,
    elevation_mean,
    elevation_variance,
    elevation_valid,
    last_seen,
    *,
    map_shape,
    resolution,
    timestamp,
    cfg,
):
    """Derive upper-bound and penetration evidence without mutating map state."""

    if not isinstance(cfg, ElevationMapLifecycleConfig):
        raise TypeError("cfg must be an ElevationMapLifecycleConfig")
    if not cfg.enabled or (
        not cfg.enable_visibility_cleanup and not cfg.enable_upper_bound
    ):
        return _empty_visibility_evidence(xp, xp.float32)

    row_count, column_count = _validated_map_shape(map_shape)
    cell_count = row_count * column_count
    cell_ids, heights, directions = _validated_visibility_samples(
        xp,
        samples,
        cell_count,
    )
    float_dtype = xp.result_type(heights.dtype, xp.float32)
    if cell_ids.size == 0:
        return _empty_visibility_evidence(xp, float_dtype)

    if cfg.enable_upper_bound:
        upper_cells, upper_heights = _minimum_upper_bound_candidates(
            xp, cell_ids, heights, cell_count
        )
    else:
        upper_cells = xp.empty((0,), dtype=xp.int64)
        upper_heights = xp.empty((0,), dtype=float_dtype)

    if not cfg.enable_visibility_cleanup:
        return VisibilityEvidence(
            upper_bound_cell_ids=upper_cells,
            upper_bound_candidates=upper_heights,
            penetrated_cell_ids=xp.empty((0,), dtype=xp.int64),
        )

    resolution = _validate_positive_real(resolution, "resolution")
    if isinstance(timestamp, bool) or not isinstance(timestamp, numbers.Real):
        raise TypeError("timestamp must be a finite real number")
    timestamp = float(timestamp)
    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")
    mean, variance, valid, seen = _validated_visibility_map(
        xp,
        elevation_mean,
        elevation_variance,
        elevation_valid,
        last_seen,
        cell_count,
    )

    rows = cell_ids // column_count
    columns = cell_ids % column_count
    center_mean = mean[cell_ids]

    row_minus_exists = rows > 0
    row_plus_exists = rows + 1 < row_count
    column_minus_exists = columns > 0
    column_plus_exists = columns + 1 < column_count
    row_minus_ids = xp.maximum(cell_ids - column_count, 0)
    row_plus_ids = xp.minimum(cell_ids + column_count, mean.size - 1)
    column_minus_ids = xp.maximum(cell_ids - 1, 0)
    column_plus_ids = xp.minimum(cell_ids + 1, mean.size - 1)

    row_minus_valid = row_minus_exists & valid[row_minus_ids]
    row_plus_valid = row_plus_exists & valid[row_plus_ids]
    column_minus_valid = column_minus_exists & valid[column_minus_ids]
    column_plus_valid = column_plus_exists & valid[column_plus_ids]
    neighbour_count = (
        row_minus_valid.astype(xp.int8)
        + row_plus_valid.astype(xp.int8)
        + column_minus_valid.astype(xp.int8)
        + column_plus_valid.astype(xp.int8)
    )

    row_both = row_minus_valid & row_plus_valid
    row_any = row_minus_valid | row_plus_valid
    row_derivative = xp.where(
        row_both,
        (mean[row_plus_ids] - mean[row_minus_ids]) / (2.0 * resolution),
        xp.where(
            row_plus_valid,
            (mean[row_plus_ids] - center_mean) / resolution,
            xp.where(
                row_minus_valid,
                (center_mean - mean[row_minus_ids]) / resolution,
                0.0,
            ),
        ),
    )
    column_both = column_minus_valid & column_plus_valid
    column_any = column_minus_valid | column_plus_valid
    column_derivative = xp.where(
        column_both,
        (mean[column_plus_ids] - mean[column_minus_ids]) / (2.0 * resolution),
        xp.where(
            column_plus_valid,
            (mean[column_plus_ids] - center_mean) / resolution,
            xp.where(
                column_minus_valid,
                (center_mean - mean[column_minus_ids]) / resolution,
                0.0,
            ),
        ),
    )
    # GpuRollingGrid rows/columns increase toward decreasing world x/y.  The
    # world-frame gradient is therefore the negative of the array gradient,
    # making (-dz/dx, -dz/dy, 1) equal to (+drow, +dcolumn, 1).
    normals = xp.stack(
        (row_derivative, column_derivative, xp.ones_like(row_derivative)),
        axis=1,
    )
    normal_norm = xp.sqrt(xp.sum(normals * normals, axis=1))
    direction_norm = xp.sqrt(xp.sum(directions * directions, axis=1))
    safe_normal_norm = xp.where(normal_norm > 0.0, normal_norm, 1.0)
    safe_direction_norm = xp.where(direction_norm > 0.0, direction_norm, 1.0)
    unit_normals = normals / safe_normal_norm[:, None]
    unit_directions = directions / safe_direction_norm[:, None]
    cosine = xp.abs(xp.sum(unit_normals * unit_directions, axis=1))
    geometry_finite = (
        xp.all(xp.isfinite(normals), axis=1)
        & xp.all(xp.isfinite(directions), axis=1)
        & xp.isfinite(cosine)
        & (normal_norm > 0.0)
        & (direction_norm > 0.0)
    )
    height_threshold = (
        center_mean
        - cfg.visibility_sigma_factor * xp.sqrt(variance[cell_ids])
        - cfg.visibility_height_margin_m
    )
    penetrated = (
        valid[cell_ids]
        & (neighbour_count >= 2)
        & (row_any | column_any)
        & geometry_finite
        & ((timestamp - seen[cell_ids]) > cfg.visibility_min_age_sec)
        & (heights < height_threshold)
        & (cosine >= cfg.visibility_cleanup_cos_threshold)
    )
    penetrated_by_cell = xp.zeros(cell_count, dtype=xp.bool_)
    penetrated_by_cell[cell_ids[penetrated]] = True
    penetrated_cells = xp.flatnonzero(penetrated_by_cell).astype(
        xp.int64, copy=False
    )
    return VisibilityEvidence(
        upper_bound_cell_ids=upper_cells,
        upper_bound_candidates=upper_heights,
        penetrated_cell_ids=penetrated_cells,
    )


def apply_visibility_and_upper_bounds(
    xp,
    evidence,
    accepted_cell_ids,
    elevation_mean,
    elevation_variance,
    elevation_valid,
    elevation_validity,
    upper_bound,
    upper_bound_valid,
    upper_bound_last_update,
    *,
    timestamp,
    max_variance,
    outlier_variance,
    cfg,
    clear_layers=(),
):
    """Commit visibility evidence after callers have backed up touched cells."""

    if not isinstance(cfg, ElevationMapLifecycleConfig):
        raise TypeError("cfg must be an ElevationMapLifecycleConfig")
    if not cfg.enabled:
        return _empty_visibility_update_result(xp)
    if not isinstance(evidence, VisibilityEvidence):
        raise TypeError("evidence must be VisibilityEvidence")
    if isinstance(timestamp, bool) or not isinstance(timestamp, numbers.Real):
        raise TypeError("timestamp must be a finite real number")
    timestamp = float(timestamp)
    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite")
    max_variance = _validate_positive_real(max_variance, "max_variance")
    if isinstance(outlier_variance, bool) or not isinstance(
        outlier_variance, numbers.Real
    ):
        raise TypeError("outlier_variance must be a finite nonnegative real number")
    outlier_variance = float(outlier_variance)
    if not math.isfinite(outlier_variance) or outlier_variance < 0.0:
        raise ValueError("outlier_variance must be finite and nonnegative")

    mean = _require_mutable_backend_array(
        xp, elevation_mean, "elevation_mean"
    )
    variance = _require_mutable_backend_array(
        xp, elevation_variance, "elevation_variance"
    )
    valid = _require_mutable_backend_array(
        xp, elevation_valid, "elevation_valid"
    )
    validity = _require_mutable_backend_array(
        xp, elevation_validity, "elevation_validity"
    )
    upper = _require_mutable_backend_array(xp, upper_bound, "upper_bound")
    upper_valid = _require_mutable_backend_array(
        xp, upper_bound_valid, "upper_bound_valid"
    )
    upper_time = _require_mutable_backend_array(
        xp, upper_bound_last_update, "upper_bound_last_update"
    )
    state_arrays = (
        ("elevation_mean", mean),
        ("elevation_variance", variance),
        ("elevation_valid", valid),
        ("elevation_validity", validity),
        ("upper_bound", upper),
        ("upper_bound_valid", upper_valid),
        ("upper_bound_last_update", upper_time),
    )
    if mean.ndim != 1:
        raise ValueError("visibility state arrays must be one-dimensional")
    cell_count = mean.size
    for name, values in state_arrays:
        if values.shape != (cell_count,):
            raise ValueError("{} must match elevation_mean".format(name))
    for name, values in (
        ("elevation_mean", mean),
        ("elevation_variance", variance),
        ("elevation_validity", validity),
        ("upper_bound", upper),
        ("upper_bound_last_update", upper_time),
    ):
        if values.dtype.kind not in "f":
            raise TypeError("{} must have a floating dtype".format(name))
    if valid.dtype.kind != "b" or upper_valid.dtype.kind != "b":
        raise TypeError("validity masks must have boolean dtypes")
    _require_float_scalar_representable(
        xp, timestamp, upper_time, "timestamp"
    )
    _require_float_scalar_representable(
        xp, max_variance, variance, "max_variance"
    )
    _require_float_scalar_representable(
        xp, outlier_variance, variance, "outlier_variance"
    )

    extra_layers = []
    if not isinstance(clear_layers, (tuple, list)):
        raise TypeError("clear_layers must be a sequence of (array, fill) pairs")
    for item in clear_layers:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("clear_layers entries must be (array, fill) pairs")
        layer, fill = item
        layer = _require_mutable_backend_array(xp, layer, "clear layer")
        if layer.shape != (cell_count,):
            raise ValueError("clear layer must match elevation_mean")
        try:
            fill_value = xp.asarray(fill, dtype=layer.dtype)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("clear layer fill must match its dtype") from error
        if fill_value.shape != ():
            raise TypeError("clear layer fill must be scalar")
        extra_layers.append((layer, fill_value))

    mutable_layers = [values for _, values in state_arrays]
    mutable_layers.extend(layer for layer, _ in extra_layers)
    for index, layer in enumerate(mutable_layers):
        for other in mutable_layers[index + 1 :]:
            if xp.shares_memory(layer, other):
                raise ValueError("mutable visibility layers must not overlap")

    upper_ids = xp.asarray(evidence.upper_bound_cell_ids)
    upper_candidates = xp.asarray(evidence.upper_bound_candidates)
    penetrated_ids = xp.asarray(evidence.penetrated_cell_ids)
    if upper_ids.ndim != 1 or upper_ids.dtype.kind != "i":
        raise TypeError("upper_bound_cell_ids must be a signed integer vector")
    if penetrated_ids.ndim != 1 or penetrated_ids.dtype.kind != "i":
        raise TypeError("penetrated_cell_ids must be a signed integer vector")
    if upper_candidates.shape != upper_ids.shape or upper_candidates.dtype.kind != "f":
        raise TypeError("upper_bound_candidates must be a matching float vector")
    upper_ids = upper_ids.astype(xp.int64, copy=False)
    penetrated_ids = penetrated_ids.astype(xp.int64, copy=False)
    accepted_ids = xp.asarray(accepted_cell_ids)
    if accepted_ids.ndim != 1 or accepted_ids.dtype.kind != "i":
        raise TypeError("accepted_cell_ids must be a signed integer vector")
    accepted_ids = accepted_ids.astype(xp.int64, copy=False)
    upper_info = xp.finfo(upper.dtype)
    upper_limit = float(upper_info.max)
    upper_smallest = float(upper_info.smallest_subnormal)
    if cell_count:
        safe_accepted_ids = xp.clip(accepted_ids, 0, cell_count - 1)
        accepted_mean_values = mean[safe_accepted_ids]
    else:
        accepted_mean_values = xp.zeros(
            accepted_ids.shape, dtype=mean.dtype
        )

    (
        invalid_state,
        invalid_upper_evidence,
        invalid_upper_ids,
        invalid_penetrated_ids,
        invalid_accepted_ids,
        unrepresentable_candidates,
        unrepresentable_accepted_mean,
        underflow_candidates,
        underflow_accepted_mean,
    ) = _device_error_flags(
        xp,
        (~xp.isfinite(mean))
        | (~xp.isfinite(variance))
        | (variance < 0.0)
        | (~xp.isfinite(validity))
        | (validity < 0.0)
        | (validity > 1.0)
        | (~xp.isfinite(upper))
        | xp.isnan(upper_time)
        | xp.isposinf(upper_time),
        ~xp.isfinite(upper_candidates),
        (upper_ids < 0) | (upper_ids >= cell_count),
        (penetrated_ids < 0) | (penetrated_ids >= cell_count),
        (accepted_ids < 0) | (accepted_ids >= cell_count),
        (
            xp.abs(upper_candidates) > upper_limit
            if cfg.enable_upper_bound
            else xp.zeros(upper_candidates.shape, dtype=xp.bool_)
        ),
        (
            xp.abs(accepted_mean_values) > upper_limit
            if cfg.enable_upper_bound
            else xp.zeros(accepted_ids.shape, dtype=xp.bool_)
        ),
        (
            (upper_candidates != 0.0)
            & (xp.abs(upper_candidates) < upper_smallest)
            if cfg.enable_upper_bound
            else xp.zeros(upper_candidates.shape, dtype=xp.bool_)
        ),
        (
            (accepted_mean_values != 0.0)
            & (xp.abs(accepted_mean_values) < upper_smallest)
            if cfg.enable_upper_bound
            else xp.zeros(accepted_ids.shape, dtype=xp.bool_)
        ),
    )
    if invalid_state:
        raise ValueError("visibility state contains invalid values")
    if invalid_upper_evidence:
        raise ValueError("upper-bound evidence must be finite")
    if invalid_upper_ids or invalid_penetrated_ids or invalid_accepted_ids:
        raise ValueError("visibility update contains an out-of-range cell")
    if (
        unrepresentable_candidates
        or unrepresentable_accepted_mean
        or underflow_candidates
        or underflow_accepted_mean
    ):
        raise ValueError("upper-bound values must be representable by target dtype")

    accepted_mask = xp.zeros(cell_count, dtype=xp.bool_)
    accepted_mask[accepted_ids] = True
    accepted_ids = xp.flatnonzero(accepted_mask).astype(xp.int64, copy=False)
    penetration_mask = xp.zeros(cell_count, dtype=xp.bool_)
    penetration_mask[penetrated_ids] = True
    penetration_mask &= (
        valid & ~accepted_mask & cfg.enable_visibility_cleanup
    )
    processed_ids = xp.flatnonzero(penetration_mask).astype(
        xp.int64, copy=False
    )

    cleared_mask = xp.zeros(cell_count, dtype=xp.bool_)
    new_validity = xp.empty((0,), dtype=validity.dtype)
    new_variance = xp.empty((0,), dtype=variance.dtype)
    processed_changed = xp.empty((0,), dtype=xp.bool_)
    if processed_ids.size:
        if cfg.enable_continuous_validity:
            new_validity = xp.maximum(
                0.0,
                validity[processed_ids] - cfg.visibility_cleanup_step,
            ).astype(validity.dtype, copy=False)
            variance_increment = min(outlier_variance, max_variance)
            new_variance = (
                xp.minimum(
                    variance[processed_ids],
                    max_variance - variance_increment,
                )
                + variance_increment
            ).astype(variance.dtype, copy=False)
            cleared_mask[processed_ids] = (
                new_validity < cfg.elevation_validity_threshold
            )
            processed_changed = (
                (new_validity != validity[processed_ids])
                | (new_variance != variance[processed_ids])
                | cleared_mask[processed_ids]
            )
        else:
            cleared_mask[processed_ids] = True
            processed_changed = xp.ones(
                processed_ids.shape, dtype=xp.bool_
            )
        cleared_ids = xp.flatnonzero(cleared_mask).astype(
            xp.int64, copy=False
        )
    else:
        cleared_ids = xp.empty((0,), dtype=xp.int64)

    post_cleanup_valid = valid.copy()
    post_cleanup_valid[cleared_ids] = False
    upper_updated_mask = xp.zeros(cell_count, dtype=xp.bool_)
    tighten = xp.zeros(cell_count, dtype=xp.bool_)
    candidate_by_cell = xp.empty((0,), dtype=upper.dtype)
    if cfg.enable_upper_bound and upper_ids.size:
        candidate_by_cell = xp.full(cell_count, xp.inf, dtype=upper.dtype)
        xp.minimum.at(candidate_by_cell, upper_ids, upper_candidates)
        candidate_mask = (
            xp.isfinite(candidate_by_cell)
            & ~post_cleanup_valid
            & ~accepted_mask
        )
        tighten = candidate_mask & (
            ~upper_valid | (candidate_by_cell < upper)
        )
        upper_updated_mask |= tighten

    accepted_validity_changed = xp.zeros(accepted_ids.shape, dtype=xp.bool_)
    accepted_upper_changed = xp.zeros(accepted_ids.shape, dtype=xp.bool_)
    accepted_upper_values = xp.empty((0,), dtype=upper.dtype)
    if accepted_ids.size:
        accepted_validity_changed = (
            ~valid[accepted_ids] | (validity[accepted_ids] != 1.0)
        )
        if cfg.enable_upper_bound:
            accepted_upper_values = mean[accepted_ids].astype(
                upper.dtype, copy=False
            )
            accepted_upper_changed = (
                (upper[accepted_ids] != accepted_upper_values)
                | upper_valid[accepted_ids]
                | (upper_time[accepted_ids] != -xp.inf)
            )
            upper_updated_mask[accepted_ids] = accepted_upper_changed

    validity_updated_mask = xp.zeros(cell_count, dtype=xp.bool_)
    validity_updated_mask[processed_ids] = processed_changed
    validity_updated_mask[accepted_ids] = accepted_validity_changed
    result = VisibilityUpdateResult(
        validity_updated_cell_ids=xp.flatnonzero(validity_updated_mask).astype(
            xp.int64, copy=False
        ),
        cleared_cell_ids=cleared_ids,
        upper_bound_updated_cell_ids=xp.flatnonzero(
            upper_updated_mask
        ).astype(xp.int64, copy=False),
        accepted_cell_ids=accepted_ids,
    )

    # Everything above is validation or plan construction.  The caller has
    # already backed up these indices; the commit phase below performs only
    # shape-validated assignments and contains no result allocation.
    if processed_ids.size and cfg.enable_continuous_validity:
        validity[processed_ids] = new_validity
        variance[processed_ids] = new_variance
    if cleared_ids.size:
        valid[cleared_ids] = False
        validity[cleared_ids] = 0.0
        mean[cleared_ids] = 0.0
        variance[cleared_ids] = max_variance
        for layer, fill in extra_layers:
            layer[cleared_ids] = fill
    if cfg.enable_upper_bound and candidate_by_cell.size:
        upper[tighten] = candidate_by_cell[tighten]
        upper_valid[tighten] = True
        upper_time[tighten] = timestamp
    if accepted_ids.size:
        valid[accepted_ids] = True
        validity[accepted_ids] = 1.0
        if cfg.enable_upper_bound:
            upper[accepted_ids] = accepted_upper_values
            upper_valid[accepted_ids] = False
            upper_time[accepted_ids] = -xp.inf
    return result


def _validate_height_drift_inputs(
    xp,
    cell_ids,
    heights,
    measurement_variances,
    mean,
    map_variance,
    validity,
    valid,
    accepted_frame_count,
    slope,
    current_bias,
    cfg,
):
    if not isinstance(cfg, ElevationMapLifecycleConfig):
        raise TypeError("cfg must be an ElevationMapLifecycleConfig")
    if isinstance(current_bias, bool) or not isinstance(current_bias, numbers.Real):
        raise TypeError("current_bias must be a finite real number")
    if not math.isfinite(float(current_bias)):
        raise ValueError("current_bias must be finite")

    cell_ids = xp.asarray(cell_ids)
    heights = _as_calculation_float(xp, heights, "heights")
    measurement_variances = _as_calculation_float(
        xp,
        measurement_variances,
        "measurement_variances",
    )
    if cell_ids.dtype.kind != "i":
        raise TypeError("cell_ids must have a signed integer dtype")
    if not (
        cell_ids.ndim == heights.ndim == measurement_variances.ndim == 1
    ):
        raise ValueError("point inputs must be one-dimensional")
    if not (cell_ids.shape == heights.shape == measurement_variances.shape):
        raise ValueError("point inputs must have equal shape")

    state = {
        "mean": xp.asarray(mean),
        "map_variance": xp.asarray(map_variance),
        "validity": xp.asarray(validity),
        "valid": xp.asarray(valid),
        "accepted_frame_count": xp.asarray(accepted_frame_count),
        "slope": xp.asarray(slope),
    }
    for name, values in state.items():
        if values.ndim != 1:
            raise ValueError("map state {} must be one-dimensional".format(name))
    state_size = int(state["mean"].size)
    for name, values in state.items():
        if int(values.size) != state_size:
            raise ValueError("map state {} has an incompatible shape".format(name))
    for name in ("mean", "map_variance", "validity", "slope"):
        values = state[name]
        if values.dtype.kind != "f" or values.dtype.itemsize not in (4, 8):
            raise TypeError(
                "map state {} must be float32 or float64".format(name)
            )
    if state["valid"].dtype.kind != "b":
        raise TypeError("map state valid must have boolean dtype")
    counts = state["accepted_frame_count"]
    if counts.dtype.kind != "i" or counts.dtype.itemsize < 4:
        raise TypeError(
            "map state accepted_frame_count must be a signed integer "
            "of at least 32 bits"
        )

    (
        height_nonfinite,
        mean_nonfinite,
        map_variance_nonfinite,
        validity_nonfinite,
        slope_nonfinite,
        cell_outside,
    ) = _device_error_flags(
        xp,
        ~xp.isfinite(heights),
        ~xp.isfinite(state["mean"]),
        ~xp.isfinite(state["map_variance"]),
        ~xp.isfinite(state["validity"]),
        ~xp.isfinite(state["slope"]),
        (cell_ids < 0) | (cell_ids >= state_size),
    )
    if height_nonfinite:
        raise ValueError("heights must be finite")
    for name, nonfinite in zip(
        ("mean", "map_variance", "validity", "slope"),
        (
            mean_nonfinite,
            map_variance_nonfinite,
            validity_nonfinite,
            slope_nonfinite,
        ),
    ):
        if nonfinite:
            raise ValueError("map state {} must be finite".format(name))
    if cell_outside:
        raise ValueError("cell_ids are outside the flat map state arrays")

    return (
        cell_ids,
        heights,
        measurement_variances,
        state["mean"],
        state["map_variance"],
        state["validity"],
        state["valid"],
        state["accepted_frame_count"],
        state["slope"],
        float(current_bias),
    )


def estimate_height_drift(
    xp,
    cell_ids,
    heights,
    measurement_variances,
    mean,
    map_variance,
    validity,
    valid,
    accepted_frame_count,
    slope,
    current_bias,
    cfg: ElevationMapLifecycleConfig,
):
    """Estimate a bounded height-bias correction from stable map cells."""

    (
        cell_ids,
        heights,
        measurement_variances,
        map_mean,
        map_variance,
        validity,
        valid,
        accepted_frame_count,
        slope,
        current_bias,
    ) = _validate_height_drift_inputs(
        xp,
        cell_ids,
        heights,
        measurement_variances,
        mean,
        map_variance,
        validity,
        valid,
        accepted_frame_count,
        slope,
        current_bias,
        cfg,
    )
    if not cfg.enabled or not cfg.enable_height_drift_compensation:
        return HeightDriftEstimate(current_bias, 0.0, 0)

    residuals = heights - (map_mean[cell_ids] + current_bias)
    candidate = (
        valid[cell_ids]
        & (validity[cell_ids] >= cfg.drift_validity_min)
        & (map_variance[cell_ids] <= cfg.drift_variance_max)
        & (accepted_frame_count[cell_ids] >= cfg.drift_min_frames)
        & (slope[cell_ids] <= math.radians(cfg.drift_slope_max_deg))
        & (xp.abs(residuals) <= cfg.drift_residual_inlier_m)
        & xp.isfinite(measurement_variances)
        & (measurement_variances > 0.0)
    )
    candidate_cells = cell_ids[candidate]
    candidate_residuals = residuals[candidate]
    candidate_variances = measurement_variances[candidate]
    sort_keys = xp.stack((candidate_variances, candidate_cells), axis=0)
    order = xp.lexsort(sort_keys)
    sorted_cells = candidate_cells[order]
    sorted_residuals = candidate_residuals[order]
    sorted_variances = candidate_variances[order]
    unique_cells, group_starts = xp.unique(
        sorted_cells,
        return_index=True,
    )
    sample_count = int(unique_cells.size)
    if sample_count < cfg.drift_min_cells:
        return HeightDriftEstimate(current_bias, 0.0, sample_count)

    minimum_variance = sorted_variances[group_starts]
    group_boundary = xp.empty(sorted_cells.shape, dtype=xp.bool_)
    group_boundary[0] = True
    group_boundary[1:] = sorted_cells[1:] != sorted_cells[:-1]
    group_ids = xp.cumsum(group_boundary, dtype=xp.int64) - 1
    sorted_precision = minimum_variance[group_ids] / sorted_variances
    precision_sum = xp.add.reduceat(sorted_precision, group_starts)
    weighted_residual_sum = xp.add.reduceat(
        sorted_residuals * sorted_precision,
        group_starts,
    )
    cell_residuals = weighted_residual_sum / precision_sum
    median_residual = xp.median(cell_residuals)
    bounded_residual = xp.clip(
        median_residual,
        -cfg.drift_max_correction_per_frame_m,
        cfg.drift_max_correction_per_frame_m,
    )
    correction = cfg.drift_alpha * float(bounded_residual.item())
    return HeightDriftEstimate(
        new_bias=current_bias + correction,
        correction=correction,
        sample_count=sample_count,
    )


def _validate_wall_edge_inputs(xp, cell_ids, heights, mean, valid, cfg):
    if not isinstance(cfg, ElevationMapLifecycleConfig):
        raise TypeError("cfg must be an ElevationMapLifecycleConfig")

    cell_ids = xp.asarray(cell_ids)
    heights = _as_calculation_float(xp, heights, "heights")
    if cell_ids.dtype.kind != "i":
        raise TypeError("cell_ids must have a signed integer dtype")
    if cell_ids.ndim != 1 or heights.ndim != 1:
        raise ValueError("point inputs must be one-dimensional")
    if cell_ids.shape != heights.shape:
        raise ValueError("point inputs must have equal shape")

    mean = xp.asarray(mean)
    valid = xp.asarray(valid)
    if mean.ndim != 1 or valid.ndim != 1:
        raise ValueError("map state arrays must be one-dimensional")
    if mean.shape != valid.shape:
        raise ValueError("map state arrays must have equal shape")
    if mean.dtype.kind != "f" or mean.dtype.itemsize not in (4, 8):
        raise TypeError("map state mean must be float32 or float64")
    if valid.dtype.kind != "b":
        raise TypeError("map state valid must have boolean dtype")
    height_nonfinite, mean_nonfinite, cell_outside = _device_error_flags(
        xp,
        ~xp.isfinite(heights),
        ~xp.isfinite(mean),
        (cell_ids < 0) | (cell_ids >= mean.size),
    )
    if height_nonfinite:
        raise ValueError("heights must be finite")
    if mean_nonfinite:
        raise ValueError("map state mean must be finite")
    if cell_outside:
        raise ValueError("cell_ids are outside the flat map state arrays")
    return cell_ids, heights, mean, valid


def wall_edge_keep_mask(
    xp,
    cell_ids,
    heights,
    mean,
    valid,
    cfg: ElevationMapLifecycleConfig,
):
    """Ignore low wall-edge points before Gaussian rejection accounting."""

    cell_ids, heights, mean, valid = _validate_wall_edge_inputs(
        xp,
        cell_ids,
        heights,
        mean,
        valid,
        cfg,
    )
    keep_mask = xp.ones(cell_ids.shape, dtype=xp.bool_)
    ignored_mask = xp.zeros(cell_ids.shape, dtype=xp.bool_)
    empty_cells = xp.empty((0,), dtype=cell_ids.dtype)
    if (
        cell_ids.size == 0
        or not cfg.enabled
        or not cfg.enable_wall_edge_sharpening
    ):
        return WallEdgeSelection(keep_mask, ignored_mask, empty_cells)

    unique_cells, inverse, counts = xp.unique(
        cell_ids,
        return_inverse=True,
        return_counts=True,
    )
    minimum = xp.full(unique_cells.shape, xp.inf, dtype=heights.dtype)
    maximum = xp.full(unique_cells.shape, -xp.inf, dtype=heights.dtype)
    xp.minimum.at(minimum, inverse, heights)
    xp.maximum.at(maximum, inverse, heights)
    sharpened_groups = (
        (counts >= cfg.wall_point_count_threshold)
        & ((maximum - minimum) >= cfg.wall_min_vertical_span_m)
        & valid[unique_cells]
    )
    ignored_mask = sharpened_groups[inverse] & (
        heights < mean[cell_ids] - cfg.wall_lower_point_tolerance_m
    )
    keep_mask = ~ignored_mask
    return WallEdgeSelection(
        keep_mask=keep_mask,
        ignored_mask=ignored_mask,
        sharpened_cell_ids=unique_cells[sharpened_groups],
    )


def overhead_keep_mask(xp, relative_points, cfg: ElevationMapLifecycleConfig):
    """Return points at or below the distance-dependent overhead limit."""

    if not isinstance(cfg, ElevationMapLifecycleConfig):
        raise TypeError("cfg must be an ElevationMapLifecycleConfig")

    points = xp.asarray(relative_points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("relative_points must have shape [N,3]")
    if points.dtype.kind not in "fiu" or points.dtype.kind == "b":
        raise TypeError("relative_points must have a real numeric dtype")

    points = points.astype(xp.result_type(points.dtype, xp.float32), copy=False)
    if points.shape[0] == 0:
        return xp.empty((0,), dtype=xp.bool_)
    if bool(xp.any(~xp.isfinite(points)).item()):
        raise ValueError("relative_points must be finite")
    if not cfg.enabled or not cfg.enable_overhead_filter:
        return xp.ones((points.shape[0],), dtype=xp.bool_)

    horizontal_distance = xp.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    if cfg.overhead_slope > 0.0:
        cap_transition = cfg.overhead_distance_offset_m + (
            cfg.overhead_height_cap_m - cfg.overhead_height_intercept_m
        ) / cfg.overhead_slope
        ramp_limit = (
            xp.maximum(horizontal_distance - cfg.overhead_distance_offset_m, 0.0)
            * cfg.overhead_slope
            + cfg.overhead_height_intercept_m
        )
        height_limit = xp.where(
            horizontal_distance >= cap_transition,
            cfg.overhead_height_cap_m,
            ramp_limit,
        )
    else:
        height_limit = xp.full_like(
            horizontal_distance,
            cfg.overhead_height_intercept_m,
        )
    return points[:, 2] <= height_limit
