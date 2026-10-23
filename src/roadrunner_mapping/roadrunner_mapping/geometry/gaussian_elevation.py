"""Backend-independent per-cell Gaussian elevation fusion primitives.

The numerical functions in this module accept an ``xp`` array namespace so
the same implementation can be used with NumPy and CuPy.  CuPy is deliberately
not imported here.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GaussianElevationConfig:
    """Validated numerical parameters for Gaussian elevation fusion."""

    sensor_std_m: float = 0.02
    range_variance_factor: float = 1e-5
    measurement_variance_min: float = 0.0004
    measurement_variance_max: float = 0.25
    pose_vertical_variance: float = 0.0009
    frame_variance_min: float = 0.0004
    posterior_variance_min: float = 0.0004
    single_point_initial_variance: float = 0.01
    multi_point_initial_variance: float = 0.005
    max_variance: float = 0.25
    process_variance_per_sec: float = 0.0004
    outlier_variance_per_frame: float = 0.01
    mahalanobis_threshold: float = 2.5
    max_effective_points_per_frame: int = 8
    initial_cluster_gap_m: float = 0.08
    initial_min_cluster_points: int = 2
    conflict_tau_sec: float = 2.0
    confidence_variance_good: float = 0.0025
    confidence_variance_bad: float = 0.04
    uncertainty_std_good: float = 0.04
    uncertainty_std_bad: float = 0.12
    local_coverage_low: float = 0.15
    local_coverage_high: float = 0.60

    def __post_init__(self) -> None:
        integer_names = (
            "max_effective_points_per_frame",
            "initial_min_cluster_points",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError("{} must be a positive integer".format(name))
            if value <= 0:
                raise ValueError("{} must be a positive integer".format(name))

        float_names = tuple(
            name
            for name in self.__dataclass_fields__
            if name not in integer_names
        )
        for name in float_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError("{} must be a finite real number".format(name))
            if not math.isfinite(float(value)):
                raise ValueError("{} must be finite".format(name))

        nonnegative = (
            "range_variance_factor",
            "process_variance_per_sec",
            "outlier_variance_per_frame",
        )
        for name in nonnegative:
            if getattr(self, name) < 0.0:
                raise ValueError("{} must be nonnegative".format(name))

        strictly_positive = (
            "sensor_std_m",
            "measurement_variance_min",
            "measurement_variance_max",
            "pose_vertical_variance",
            "frame_variance_min",
            "posterior_variance_min",
            "single_point_initial_variance",
            "multi_point_initial_variance",
            "max_variance",
            "mahalanobis_threshold",
            "initial_cluster_gap_m",
            "conflict_tau_sec",
            "confidence_variance_good",
            "confidence_variance_bad",
            "uncertainty_std_good",
            "uncertainty_std_bad",
        )
        for name in strictly_positive:
            if getattr(self, name) <= 0.0:
                raise ValueError("{} must be positive".format(name))

        if self.measurement_variance_min > self.measurement_variance_max:
            raise ValueError("measurement variance minimum exceeds maximum")
        if not (
            self.posterior_variance_min
            <= self.frame_variance_min
            <= self.max_variance
        ):
            raise ValueError(
                "posterior_variance_min <= frame_variance_min <= max_variance "
                "is required"
            )
        for name in (
            "single_point_initial_variance",
            "multi_point_initial_variance",
        ):
            value = getattr(self, name)
            if not self.posterior_variance_min <= value <= self.max_variance:
                raise ValueError("{} is outside posterior variance bounds".format(name))
        if self.multi_point_initial_variance > self.single_point_initial_variance:
            raise ValueError(
                "multi-point initial variance must not exceed single-point variance"
            )
        if self.confidence_variance_good >= self.confidence_variance_bad:
            raise ValueError("confidence variance bounds must increase")
        if self.uncertainty_std_good >= self.uncertainty_std_bad:
            raise ValueError("uncertainty standard-deviation bounds must increase")
        if not (
            0.0 <= self.local_coverage_low < self.local_coverage_high <= 1.0
        ):
            raise ValueError("local coverage bounds must increase within [0, 1]")


def point_height_variance(xp, ranges, cfg: GaussianElevationConfig):
    """Return independent per-point height variance from sensor-frame range.

    Shared vertical pose uncertainty is intentionally excluded.  It is added
    once when a complete frame observation is formed.
    """

    ranges = _as_calculation_float(xp, ranges, "ranges")
    nonfinite, negative = _device_error_flags(
        xp,
        ~xp.isfinite(ranges),
        ranges < 0.0,
    )
    if nonfinite:
        raise ValueError("ranges must be finite")
    if negative:
        raise ValueError("ranges must be nonnegative")
    variance = cfg.sensor_std_m**2 + cfg.range_variance_factor * ranges * ranges
    return xp.clip(
        variance,
        cfg.measurement_variance_min,
        cfg.measurement_variance_max,
    )


@dataclass(frozen=True)
class FrameObservationAggregation:
    """One precision-weighted observation for each accepted cell."""

    cell_ids: Any
    count: Any
    mean: Any
    variance: Any
    scatter: Any
    precision: Any


@dataclass(frozen=True)
class GaussianElevationUpdate:
    """Point- and cell-level information produced by one frame update."""

    touched_cell_ids: Any
    accepted_cell_ids: Any
    accepted_point_mask: Any
    rejected_point_mask: Any
    reject_fraction: Any


def _as_calculation_float(xp, values, name):
    values = xp.asarray(values)
    if values.dtype.kind not in "fiu" or values.dtype.kind == "b":
        raise TypeError("{} must have a real numeric dtype".format(name))
    dtype = xp.result_type(values.dtype, xp.float32)
    return values.astype(dtype, copy=False)


def _device_error_flags(xp, *conditions):
    """Return several device predicates with one device-to-host synchronization."""

    flags = xp.stack(tuple(xp.any(condition) for condition in conditions))
    return tuple(bool(value) for value in flags.tolist())


def _validate_point_arrays(xp, cell_ids, heights, variances=None):
    cell_ids = xp.asarray(cell_ids)
    heights = _as_calculation_float(xp, heights, "heights")
    if cell_ids.ndim != 1 or heights.ndim != 1:
        raise ValueError("point inputs must be one-dimensional")
    if cell_ids.shape != heights.shape:
        raise ValueError("cell IDs and heights must have equal shape")
    height_nonfinite, cell_negative = _device_error_flags(
        xp,
        ~xp.isfinite(heights),
        cell_ids < 0,
    )
    if height_nonfinite:
        raise ValueError("heights must be finite")
    if cell_negative:
        raise ValueError("cell IDs must be nonnegative")
    if cell_ids.dtype.kind != "i":
        raise TypeError("cell IDs must have a signed integer dtype")
    if variances is None:
        return cell_ids, heights
    variances = _as_calculation_float(xp, variances, "variances")
    if variances.ndim != 1 or variances.shape != heights.shape:
        raise ValueError("point variances must match heights")
    variance_nonfinite, variance_nonpositive = _device_error_flags(
        xp,
        ~xp.isfinite(variances),
        variances <= 0.0,
    )
    if variance_nonfinite or variance_nonpositive:
        raise ValueError("point variances must be finite and positive")
    return cell_ids, heights, variances


def select_initial_clusters(
    xp,
    cell_ids,
    heights,
    cfg: GaussianElevationConfig,
):
    """Select the robust initialization height cluster in every unknown cell.

    Adjacent sorted heights split only when their difference is strictly larger
    than ``initial_cluster_gap_m``.  The largest cluster wins; because clusters
    are visited from low to high, retaining the first equal-size cluster also
    implements the lowest-mean tie break deterministically.
    """

    cell_ids, heights = _validate_point_arrays(xp, cell_ids, heights)
    selected = xp.zeros(cell_ids.shape, dtype=xp.bool_)
    if cell_ids.size == 0:
        return selected

    sort_keys = xp.stack((heights, cell_ids), axis=0)
    order = xp.lexsort(sort_keys)
    sorted_cells = cell_ids[order]
    sorted_heights = heights[order]
    cluster_boundary = xp.empty(cell_ids.shape, dtype=xp.bool_)
    cluster_boundary[0] = True
    cluster_boundary[1:] = (sorted_cells[1:] != sorted_cells[:-1]) | (
        sorted_heights[1:] - sorted_heights[:-1] > cfg.initial_cluster_gap_m
    )
    cluster_ids = xp.cumsum(cluster_boundary, dtype=xp.int64) - 1
    # CuPy 12.3 does not implement maximum.at for signed int64 data. Counts
    # are nonnegative, so uint64 keeps full capacity while using a supported
    # device atomic dtype. Group/index IDs remain signed int64 for indexing.
    cluster_count = xp.bincount(cluster_ids).astype(xp.uint64, copy=False)
    calculation_dtype = sorted_heights.dtype
    cluster_sum = xp.bincount(
        cluster_ids, weights=sorted_heights
    ).astype(calculation_dtype, copy=False)
    cluster_mean = cluster_sum / cluster_count
    cluster_cells = sorted_cells[cluster_boundary]

    cell_boundary = xp.empty(cluster_cells.shape, dtype=xp.bool_)
    cell_boundary[0] = True
    cell_boundary[1:] = cluster_cells[1:] != cluster_cells[:-1]
    cell_starts = xp.nonzero(cell_boundary)[0]
    cluster_cell_ids = xp.cumsum(cell_boundary, dtype=xp.int64) - 1
    cell_count = int(cell_starts.size)
    maximum_count = xp.zeros((cell_count,), dtype=cluster_count.dtype)
    xp.maximum.at(maximum_count, cluster_cell_ids, cluster_count)
    count_winner = cluster_count == maximum_count[cluster_cell_ids]
    # Clusters are already ordered by (cell, height), hence cluster means are
    # increasing within each cell. Select the first maximum-count candidate in
    # every cell instead of round-tripping a float64 mean through CuPy 12.3's
    # lossy minimum.at implementation.
    candidate_indices = xp.nonzero(count_winner)[0]
    candidate_cells = cluster_cell_ids[candidate_indices]
    _, first_candidate = xp.unique(candidate_cells, return_index=True)
    selected_cluster = xp.zeros(cluster_count.shape, dtype=xp.bool_)
    selected_cluster[candidate_indices[first_candidate]] = True
    selected[order] = selected_cluster[cluster_ids]
    return selected


def aggregate_frame_observations(
    xp,
    cell_ids,
    heights,
    variances,
    accepted,
    cfg: GaussianElevationConfig,
) -> FrameObservationAggregation:
    """Aggregate accepted heteroscedastic points into one observation per cell."""

    cell_ids, heights, variances = _validate_point_arrays(
        xp, cell_ids, heights, variances
    )
    accepted = xp.asarray(accepted)
    if accepted.ndim != 1 or accepted.shape != heights.shape:
        raise ValueError("accepted mask must match heights")
    if accepted.dtype.kind != "b":
        raise TypeError("accepted mask must have boolean dtype")

    accepted_cells = cell_ids[accepted]
    accepted_heights = heights[accepted]
    accepted_variances = variances[accepted]
    if accepted_cells.size == 0:
        empty_float = xp.empty((0,), dtype=heights.dtype)
        return FrameObservationAggregation(
            cell_ids=xp.empty((0,), dtype=cell_ids.dtype),
            count=xp.empty((0,), dtype=xp.int64),
            mean=empty_float,
            variance=empty_float.copy(),
            scatter=empty_float.copy(),
            precision=empty_float.copy(),
        )

    unique_cells, inverse = xp.unique(accepted_cells, return_inverse=True)
    group_count = int(unique_cells.size)
    count = xp.bincount(inverse, minlength=group_count).astype(xp.int64)
    point_precision = 1.0 / accepted_variances
    calculation_dtype = xp.result_type(heights.dtype, variances.dtype, xp.float32)
    point_precision = point_precision.astype(calculation_dtype, copy=False)
    precision = xp.bincount(
        inverse, weights=point_precision, minlength=group_count
    ).astype(calculation_dtype, copy=False)
    weighted_sum = xp.bincount(
        inverse,
        weights=point_precision * accepted_heights,
        minlength=group_count,
    ).astype(calculation_dtype, copy=False)
    mean = weighted_sum / precision
    residual = accepted_heights - mean[inverse]
    scatter = (
        xp.bincount(
            inverse,
            weights=point_precision * residual * residual,
            minlength=group_count,
        ).astype(calculation_dtype, copy=False)
        / precision
    )
    effective_count = xp.minimum(count, cfg.max_effective_points_per_frame)
    precision_scale = effective_count / count
    effective_precision = precision * precision_scale
    frame_variance = xp.maximum(
        1.0 / effective_precision
        + scatter / effective_count
        + cfg.pose_vertical_variance,
        cfg.frame_variance_min,
    ).astype(calculation_dtype, copy=False)
    return FrameObservationAggregation(
        cell_ids=unique_cells,
        count=count,
        mean=mean,
        variance=frame_variance,
        scatter=scatter,
        precision=precision,
    )


_FLOAT_STATE_NAMES = (
    "mean",
    "variance",
    "last_seen",
    "variance_updated_at",
    "conflict",
    "conflict_updated_at",
    "ground_sum",
    "min_z",
    "max_z",
)
_COUNT_STATE_NAMES = ("support_count", "accepted_frame_count")


def _validate_state_arrays(xp, state, cfg, *, validate_values):
    """Validate the complete mutable state before any fusion write."""

    for name, values in state.items():
        if not isinstance(values, xp.ndarray):
            raise TypeError("state array {} must belong to the xp backend".format(name))
        if values.ndim != 1:
            raise ValueError("state array {} must be one-dimensional".format(name))
        writable = getattr(getattr(values, "flags", None), "writeable", True)
        if not writable:
            raise ValueError("state array {} must be writable".format(name))

    state_size = int(state["mean"].size)
    for name, values in state.items():
        if int(values.size) != state_size:
            raise ValueError("state array {} has an incompatible shape".format(name))
    for name in _FLOAT_STATE_NAMES:
        values = state[name]
        if values.dtype.kind != "f" or values.dtype.itemsize not in (4, 8):
            raise TypeError("state array {} must be float32 or float64".format(name))
    if state["valid"].dtype.kind != "b":
        raise TypeError("state array valid must have boolean dtype")
    for name in _COUNT_STATE_NAMES:
        values = state[name]
        if values.dtype.kind != "i" or values.dtype.itemsize < 4:
            raise TypeError(
                "state array {} must be a signed integer of at least 32 bits".format(
                    name
                )
            )

    if not validate_values:
        return state_size

    timestamp_legal = (
        (xp.isfinite(state["last_seen"]) | xp.isneginf(state["last_seen"]))
        & (
            xp.isfinite(state["variance_updated_at"])
            | xp.isneginf(state["variance_updated_at"])
        )
        & (
            xp.isfinite(state["conflict_updated_at"])
            | xp.isneginf(state["conflict_updated_at"])
        )
    )
    extrema_legal = (
        (xp.isfinite(state["min_z"]) & xp.isfinite(state["max_z"]))
        | (xp.isposinf(state["min_z"]) & xp.isneginf(state["max_z"]))
    )
    extrema_ordered = (state["min_z"] <= state["max_z"]) | (
        xp.isposinf(state["min_z"]) & xp.isneginf(state["max_z"])
    )
    legal = (
        xp.isfinite(state["mean"])
        & xp.isfinite(state["variance"])
        & (state["variance"] >= cfg.posterior_variance_min)
        & (state["variance"] <= cfg.max_variance)
        & timestamp_legal
        & xp.isfinite(state["conflict"])
        & (state["conflict"] >= 0.0)
        & (state["conflict"] <= 1.0)
        & xp.isfinite(state["ground_sum"])
        & (state["ground_sum"] >= 0.0)
        & extrema_legal
        & extrema_ordered
        & (state["support_count"] >= 0)
        & (state["accepted_frame_count"] >= 0)
    )
    if bool(xp.any(~legal)):
        raise ValueError("state arrays contain an illegal value")
    return state_size


def update_gaussian_elevation(
    xp,
    cell_ids,
    heights,
    variances,
    ground_weights,
    timestamp: float,
    *,
    mean,
    variance,
    valid,
    support_count,
    accepted_frame_count,
    last_seen,
    variance_updated_at,
    conflict,
    conflict_updated_at,
    ground_sum,
    min_z,
    max_z,
    cfg: GaussianElevationConfig,
    validate_state_values: bool = True,
) -> GaussianElevationUpdate:
    """Fuse one frame of explicit flat-cell height observations in place.

    Frame identity/de-duplication belongs to the caller.  For any already valid
    touched cell, this numerical core requires a strictly newer timestamp.
    """

    state = {
        "mean": mean,
        "variance": variance,
        "valid": valid,
        "support_count": support_count,
        "accepted_frame_count": accepted_frame_count,
        "last_seen": last_seen,
        "variance_updated_at": variance_updated_at,
        "conflict": conflict,
        "conflict_updated_at": conflict_updated_at,
        "ground_sum": ground_sum,
        "min_z": min_z,
        "max_z": max_z,
    }
    if not isinstance(validate_state_values, bool):
        raise TypeError("validate_state_values must be boolean")
    state_size = _validate_state_arrays(
        xp,
        state,
        cfg,
        validate_values=validate_state_values,
    )
    if not isinstance(timestamp, numbers.Real) or not math.isfinite(float(timestamp)):
        raise ValueError("timestamp must be finite")
    timestamp = float(timestamp)
    cell_ids = xp.asarray(cell_ids)
    heights = _as_calculation_float(xp, heights, "heights")
    variances = _as_calculation_float(xp, variances, "variances")
    ground_weights = _as_calculation_float(xp, ground_weights, "ground weights")
    if cell_ids.ndim != 1 or cell_ids.dtype.kind != "i":
        raise TypeError("cell IDs must be a one-dimensional signed integer array")
    if not (
        heights.ndim == variances.ndim == ground_weights.ndim == 1
        and cell_ids.shape == heights.shape == variances.shape == ground_weights.shape
    ):
        raise ValueError("all point inputs must be one-dimensional with equal shape")
    if bool(xp.any((cell_ids < 0) | (cell_ids >= state_size))):
        raise ValueError("cell IDs are outside the flat state arrays")

    finite_point = (
        xp.isfinite(heights)
        & xp.isfinite(variances)
        & (variances > 0.0)
        & xp.isfinite(ground_weights)
    )
    accepted_point_mask = xp.zeros(cell_ids.shape, dtype=xp.bool_)
    rejected_point_mask = xp.zeros(cell_ids.shape, dtype=xp.bool_)
    finite_indices = xp.nonzero(finite_point)[0]
    finite_cells = cell_ids[finite_point]
    finite_heights = heights[finite_point]
    finite_variances = variances[finite_point]
    touched_cells, touched_inverse = xp.unique(finite_cells, return_inverse=True)
    touched_was_valid = valid[touched_cells]
    known_cells = touched_cells[touched_was_valid]

    if known_cells.size:
        known_times = variance_updated_at[known_cells]
        if bool(xp.any(timestamp <= known_times)):
            raise ValueError(
                "timestamp must be strictly increasing for known touched cells"
            )

    finite_known = valid[finite_cells]
    finite_accepted = xp.zeros(finite_cells.shape, dtype=xp.bool_)
    known_point_cells = finite_cells[finite_known]
    elapsed = timestamp - variance_updated_at[known_point_cells]
    predicted = xp.minimum(
        variance[known_point_cells] + cfg.process_variance_per_sec * elapsed,
        cfg.max_variance,
    )
    denominator = xp.sqrt(
        predicted + finite_variances[finite_known] + cfg.pose_vertical_variance
    )
    innovation = xp.abs(finite_heights[finite_known] - mean[known_point_cells])
    known_indices = xp.nonzero(finite_known)[0]
    finite_accepted[known_indices] = (
        innovation / denominator <= cfg.mahalanobis_threshold
    )

    finite_unknown = ~finite_known
    unknown_indices = xp.nonzero(finite_unknown)[0]
    unknown_selected = select_initial_clusters(
        xp,
        finite_cells[finite_unknown],
        finite_heights[finite_unknown],
        cfg,
    )
    finite_accepted[unknown_indices] = unknown_selected

    accepted_point_mask[finite_indices] = finite_accepted
    rejected_point_mask[finite_indices] = ~finite_accepted
    touched_count = int(touched_cells.size)
    raw_count = xp.zeros((touched_count,), dtype=xp.uint64)
    accepted_count = xp.zeros((touched_count,), dtype=xp.uint64)
    xp.add.at(raw_count, touched_inverse, xp.uint64(1))
    xp.add.at(
        accepted_count,
        touched_inverse[finite_accepted],
        xp.uint64(1),
    )
    reject_fraction = (1.0 - accepted_count / raw_count).astype(
        variance.dtype, copy=False
    )

    aggregation = aggregate_frame_observations(
        xp,
        finite_cells,
        finite_heights,
        finite_variances,
        finite_accepted,
        cfg,
    )
    accepted_cells = aggregation.cell_ids
    accepted_positions = xp.searchsorted(touched_cells, accepted_cells)

    new_touched_mean = mean[touched_cells].copy()
    new_touched_variance = variance[touched_cells].copy()
    new_touched_valid = valid[touched_cells].copy()
    new_touched_variance[touched_was_valid] = xp.minimum(
        new_touched_variance[touched_was_valid]
        + cfg.process_variance_per_sec
        * (timestamp - variance_updated_at[known_cells]),
        cfg.max_variance,
    )
    if accepted_cells.size:
        accepted_was_valid = valid[accepted_cells]
        known_observation_mean = aggregation.mean[accepted_was_valid]
        known_observation_variance = aggregation.variance[accepted_was_valid]
        known_accepted_positions = accepted_positions[accepted_was_valid]
        if known_accepted_positions.size:
            prior_variance = new_touched_variance[known_accepted_positions]
            gain = prior_variance / (
                prior_variance + known_observation_variance
            )
            new_touched_mean[known_accepted_positions] = new_touched_mean[
                known_accepted_positions
            ] + gain * (
                known_observation_mean
                - new_touched_mean[known_accepted_positions]
            )
            new_touched_variance[known_accepted_positions] = xp.maximum(
                (1.0 - gain) * prior_variance,
                cfg.posterior_variance_min,
            )

        unknown_accepted_positions = accepted_positions[~accepted_was_valid]
        if unknown_accepted_positions.size:
            unknown_count = aggregation.count[~accepted_was_valid]
            initial_floor = xp.where(
                unknown_count >= cfg.initial_min_cluster_points,
                cfg.multi_point_initial_variance,
                cfg.single_point_initial_variance,
            )
            new_touched_mean[unknown_accepted_positions] = aggregation.mean[
                ~accepted_was_valid
            ]
            new_touched_variance[unknown_accepted_positions] = xp.clip(
                xp.maximum(
                    aggregation.variance[~accepted_was_valid], initial_floor
                ),
                cfg.posterior_variance_min,
                cfg.max_variance,
            )
            new_touched_valid[unknown_accepted_positions] = True

        support_limit = xp.iinfo(support_count.dtype).max
        frame_limit = xp.iinfo(accepted_frame_count.dtype).max
        overflow = xp.any(
            support_count[accepted_cells]
            > support_limit - aggregation.count
        ) | xp.any(accepted_frame_count[accepted_cells] >= frame_limit)
        if bool(overflow):
            raise OverflowError("state counter capacity would be exceeded")

        accepted_input_cells = finite_cells[finite_accepted]
        accepted_heights = finite_heights[finite_accepted]
        accepted_ground = ground_weights[finite_point][finite_accepted]
        accepted_sort_keys = xp.stack(
            (accepted_heights, accepted_input_cells), axis=0
        )
        accepted_order = xp.lexsort(accepted_sort_keys)
        sorted_accepted_cells = accepted_input_cells[accepted_order]
        sorted_accepted_heights = accepted_heights[accepted_order]
        sorted_accepted_ground = accepted_ground[accepted_order]
        group_boundary = xp.empty(sorted_accepted_cells.shape, dtype=xp.bool_)
        group_boundary[0] = True
        group_boundary[1:] = (
            sorted_accepted_cells[1:] != sorted_accepted_cells[:-1]
        )
        group_starts = xp.nonzero(group_boundary)[0]
        accepted_ground_sum = xp.add.reduceat(
            sorted_accepted_ground, group_starts
        )
        accepted_min_z = sorted_accepted_heights[group_starts]
        group_ends = group_starts + aggregation.count - 1
        accepted_max_z = sorted_accepted_heights[group_ends]

    new_touched_variance = xp.minimum(
        new_touched_variance + cfg.outlier_variance_per_frame * reject_fraction,
        cfg.max_variance,
    )

    # Commit only after every validation and all potentially failing grouped
    # computation has completed, preserving exception atomicity.
    mean[touched_cells] = new_touched_mean
    variance[touched_cells] = new_touched_variance
    valid[touched_cells] = new_touched_valid
    variance_updated_at[touched_cells] = timestamp
    conflict[touched_cells] = reject_fraction
    conflict_updated_at[touched_cells] = timestamp
    if accepted_cells.size:
        support_count[accepted_cells] = (
            support_count[accepted_cells] + aggregation.count
        )
        accepted_frame_count[accepted_cells] = (
            accepted_frame_count[accepted_cells] + 1
        )
        last_seen[accepted_cells] = timestamp
        ground_sum[accepted_cells] = (
            ground_sum[accepted_cells] + accepted_ground_sum
        )
        min_z[accepted_cells] = xp.minimum(
            min_z[accepted_cells], accepted_min_z
        )
        max_z[accepted_cells] = xp.maximum(
            max_z[accepted_cells], accepted_max_z
        )

    return GaussianElevationUpdate(
        touched_cell_ids=touched_cells,
        accepted_cell_ids=accepted_cells,
        accepted_point_mask=accepted_point_mask,
        rejected_point_mask=rejected_point_mask,
        reject_fraction=reject_fraction,
    )
