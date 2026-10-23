"""Persistent GPU elevation grid for chronological GOOSE teacher targets."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterable

import numpy as np

from .coordinates import validate_transform
from .gaussian_elevation import (
    GaussianElevationConfig,
    point_height_variance,
    update_gaussian_elevation,
)
from .elevation_map_lifecycle import (
    ElevationMapLifecycleConfig,
    VisibilityEvidence,
    apply_visibility_and_upper_bounds,
    estimate_height_drift,
    overhead_keep_mask,
    raycast_grid_samples,
    select_visibility_endpoints,
    visibility_evidence,
    wall_edge_keep_mask,
)
from .gpu_rolling_grid import (
    GpuRollingGrid,
    bilinear_world_xy_samples,
    weighted_neighbor_sample,
)
from .schema import MapSpec


_GEOMETRY_TOP_LEVEL_TIMING_FIELDS = (
    "geometry_prepare_sec",
    "geometry_drift_sec",
    "geometry_visibility_sec",
    "geometry_map_update_sec",
    "geometry_world_fields_sec",
    "geometry_risk_sec",
    "geometry_sample_sec",
    "geometry_transfer_sec",
)
_GEOMETRY_MAP_TIMING_FIELDS = (
    "geometry_map_center_sec",
    "geometry_map_maintenance_sec",
    "geometry_map_backup_sec",
    "geometry_map_clear_sec",
    "geometry_map_visibility_apply_sec",
    "geometry_map_fusion_sec",
    "geometry_map_commit_sec",
)
_GEOMETRY_TIMING_FIELDS = (
    _GEOMETRY_TOP_LEVEL_TIMING_FIELDS + _GEOMETRY_MAP_TIMING_FIELDS
)


class _GeometryStageTimer:
    """Measure non-overlapping geometry stages without per-stage GPU syncs."""

    def __init__(self, *, cp, device):
        self._device = device
        self._durations = {name: 0.0 for name in _GEOMETRY_TIMING_FIELDS}
        self._event_segments = []
        self._active = False
        self._active_name = None
        self._active_started = None

        cuda = getattr(cp, "cuda", None)
        self._event_factory = getattr(cuda, "Event", None)
        self._get_elapsed_time = getattr(cuda, "get_elapsed_time", None)
        get_current_stream = getattr(cuda, "get_current_stream", None)
        self._uses_cuda_events = all(
            callable(api)
            for api in (
                self._event_factory,
                self._get_elapsed_time,
                get_current_stream,
            )
        )
        self._stream = None
        if self._uses_cuda_events:
            with self._device:
                self._stream = get_current_stream()

    @staticmethod
    def _finite_nonnegative(value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            return 0.0
        return value

    @classmethod
    def _accumulated_duration(cls, current: float, elapsed: float) -> float:
        current = cls._finite_nonnegative(current)
        elapsed = cls._finite_nonnegative(elapsed)
        if elapsed > sys.float_info.max - current:
            return sys.float_info.max
        return current + elapsed

    def _add_duration(self, name: str, elapsed: float) -> None:
        self._durations[name] = self._accumulated_duration(
            self._durations[name], elapsed
        )

    def _map_update_duration(self, durations) -> float:
        total = 0.0
        for name in _GEOMETRY_MAP_TIMING_FIELDS:
            total = self._accumulated_duration(total, durations[name])
        return total

    def _record_event(self):
        with self._device:
            event = self._event_factory()
            event.record(self._stream)
        return event

    @property
    def active(self):
        return self._active_name if self._active else None

    def start(self, name: str) -> None:
        if name not in self._durations:
            raise ValueError("unknown geometry timing field: {}".format(name))
        if name == "geometry_map_update_sec":
            raise ValueError("aggregate geometry timing field cannot be recorded")
        if self._active:
            raise RuntimeError("nested geometry timing segments are not supported")
        self._active = True
        self._active_name = name
        try:
            self._active_started = (
                self._record_event()
                if self._uses_cuda_events
                else time.perf_counter()
            )
        except Exception:
            self._active = False
            self._active_name = None
            raise

    def stop(self) -> None:
        if not self._active:
            raise RuntimeError("no active geometry timing segment")
        name = self._active_name
        started = self._active_started
        try:
            if self._uses_cuda_events:
                ended = self._record_event()
                self._event_segments.append((name, started, ended))
            else:
                self._add_duration(name, time.perf_counter() - started)
        finally:
            self._active = False
            self._active_name = None
            self._active_started = None

    @contextmanager
    def segment(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop()

    def finish(self) -> dict[str, float]:
        if self._active:
            raise RuntimeError("cannot finish an active geometry timing segment")
        durations = dict(self._durations)
        if self._event_segments:
            with self._device:
                self._event_segments[-1][2].synchronize()
                for name, started, ended in self._event_segments:
                    durations[name] = self._accumulated_duration(
                        durations[name],
                        self._get_elapsed_time(started, ended) * 1e-3,
                    )
        durations["geometry_map_update_sec"] = self._map_update_duration(
            durations
        )
        self._durations = durations
        if self._event_segments:
            self._event_segments.clear()
        return dict(self._durations)


@contextmanager
def _geometry_timing_segment(timer, name: str):
    if timer is None:
        yield
        return
    with timer.segment(name):
        yield


def _start_geometry_timing(timer, name: str) -> None:
    if timer is not None:
        timer.start(name)


def _stop_geometry_timing(timer) -> None:
    if timer is not None:
        timer.stop()


def _switch_geometry_timing(timer, name: str) -> None:
    if timer is not None:
        timer.stop()
        timer.start(name)


def _close_geometry_timing_on_error(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        timer = kwargs.get("timer")
        try:
            return method(*args, **kwargs)
        except Exception:
            if timer is not None and timer.active is not None:
                timer.stop()
            raise

    return wrapped


def _empty_geometry_timing() -> dict[str, float]:
    return {name: 0.0 for name in _GEOMETRY_TIMING_FIELDS}


def _smoothstep(cp, edge0: float, edge1: float, values):
    scaled = cp.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _plane_fit_is_valid(cp, local_count, window_area: float, determinant):
    """Return the independent support, coverage and rank validity contract."""

    return (
        (local_count >= 4.0)
        & (local_count / float(window_area) >= 0.25)
        & (cp.abs(determinant) > cp.float64(1e-12))
    )


def _update_risk_distribution(
    *,
    cp,
    risk_sum,
    risk_sq_sum,
    risk_weight,
    risk_histogram,
    risk_last_update,
    risk,
    confidence,
    valid,
    timestamp: float,
    decay_tau_sec: float,
) -> None:
    """Add qualified geometry-risk observations to weighted GPU statistics.

    Invalid or low-confidence geometry is deliberately excluded here.  The
    accumulated elevation state still receives those points, but transient
    sparse-cloud risk estimates must not become persistent risk evidence.
    """

    risk = cp.asarray(risk, dtype=cp.float32)
    confidence = cp.asarray(confidence, dtype=cp.float32)
    valid = cp.asarray(valid, dtype=cp.bool_)
    sample_valid = (
        valid
        & cp.isfinite(risk)
        & cp.isfinite(confidence)
        & (confidence > 0.0)
    )
    finite_update = cp.isfinite(risk_last_update)
    age = cp.maximum(0.0, np.float64(timestamp) - risk_last_update)
    decay = cp.where(
        finite_update,
        cp.exp(-age / float(decay_tau_sec)).astype(cp.float32),
        cp.float32(0.0),
    )
    sample_risk = cp.clip(risk, 0.0, 1.0)
    sample_weight = cp.clip(confidence, 0.0, 1.0)

    old_weight = risk_weight[sample_valid] * decay[sample_valid]
    new_weight = sample_weight[sample_valid]
    new_risk = sample_risk[sample_valid]
    risk_weight[sample_valid] = old_weight + new_weight
    risk_sum[sample_valid] = (
        risk_sum[sample_valid] * decay[sample_valid]
        + new_weight * new_risk
    )
    risk_sq_sum[sample_valid] = (
        risk_sq_sum[sample_valid] * decay[sample_valid]
        + new_weight * new_risk * new_risk
    )
    risk_last_update[sample_valid] = np.float64(timestamp)

    bin_count = len(risk_histogram)
    bin_index = cp.minimum(
        (sample_risk * float(bin_count)).astype(cp.int32),
        bin_count - 1,
    )
    for index, histogram in enumerate(risk_histogram):
        histogram[sample_valid] = (
            histogram[sample_valid] * decay[sample_valid]
            + new_weight * (bin_index[sample_valid] == index)
        )


def _compute_risk_distribution_fields(
    *,
    cp,
    risk_sum,
    risk_sq_sum,
    risk_weight,
    risk_histogram,
    cvar_alpha: float,
) -> dict[str, object]:
    """Convert weighted moments and a risk histogram into map layers.

    The histogram CVaR uses the upper edge of each occupied bin.  This is a
    conservative approximation suitable for a safety-oriented risk layer.
    """

    safe_weight = cp.maximum(risk_weight, cp.float32(1e-12))
    mean = risk_sum / safe_weight
    variance = cp.maximum(risk_sq_sum / safe_weight - mean * mean, 0.0)
    std = cp.sqrt(variance)
    tail_mass = (1.0 - float(cvar_alpha)) * risk_weight
    remaining = tail_mass.copy()
    tail_sum = cp.zeros_like(risk_weight, dtype=cp.float32)
    bin_count = len(risk_histogram)
    for index in range(bin_count - 1, -1, -1):
        take = cp.minimum(risk_histogram[index], remaining)
        upper_edge = float(index + 1) / float(bin_count)
        tail_sum += take * upper_edge
        remaining = cp.maximum(remaining - take, 0.0)
    cvar = cp.where(
        tail_mass > 1e-12,
        tail_sum / cp.maximum(tail_mass, 1e-12),
        0.0,
    )
    valid = risk_weight > 1e-6
    return {
        "geometry_risk_mean": cp.where(valid, cp.clip(mean, 0.0, 1.0), 0.0).astype(
            cp.float32
        ),
        "geometry_risk_std": cp.where(valid, std, 0.0).astype(cp.float32),
        "geometry_risk_cvar": cp.where(
            valid, cp.clip(cvar, 0.0, 1.0), 0.0
        ).astype(cp.float32),
        "geometry_risk_distribution_weight": risk_weight.astype(cp.float32),
        "geometry_risk_distribution_valid": valid,
    }


def _compute_world_slope_fields(
    *,
    cp,
    ndimage,
    elevation_mean,
    elevation_valid,
    last_seen,
    latest_timestamp: float,
    time_origin: float,
    max_geometry_age_sec: float,
    resolution: float,
):
    """Derive the shared lightweight world-slope preprocessing fields."""

    relative_latest = float(latest_timestamp) - float(time_origin)
    age = cp.maximum(0.0, relative_latest - last_seen)
    observed = elevation_valid & (age <= float(max_geometry_age_sec))
    mask = observed.astype(cp.float32)
    elevation_raw = cp.where(observed, elevation_mean, cp.float32(0.0))

    kernel = max(3, int(round(1.4 / float(resolution))))
    if kernel % 2 == 0:
        kernel += 1
    window_area = float(kernel * kernel)
    local_count = (
        ndimage.uniform_filter(mask, size=kernel, mode="constant", cval=0.0)
        * window_area
    )
    local_sum = (
        ndimage.uniform_filter(
            elevation_raw * mask, size=kernel, mode="constant", cval=0.0
        )
        * window_area
    )
    local_mean = cp.where(
        local_count > 0, local_sum / cp.maximum(local_count, 1.0), 0.0
    )
    filled = cp.where(observed, elevation_raw, local_mean)
    ground = ndimage.median_filter(filled, size=5, mode="nearest")
    longitudinal_gradient = ndimage.sobel(
        ground, axis=0, mode="nearest"
    ) / (8.0 * float(resolution))
    lateral_gradient = ndimage.sobel(
        ground, axis=1, mode="nearest"
    ) / (8.0 * float(resolution))
    longitudinal_slope = cp.arctan(cp.abs(longitudinal_gradient))
    lateral_slope = cp.arctan(cp.abs(lateral_gradient))
    slope = cp.arctan(cp.hypot(longitudinal_gradient, lateral_gradient))
    valid = observed & (local_count >= 4.0)
    for layer in (slope, longitudinal_slope, lateral_slope):
        layer[~valid] = 0.0
    return {
        "relative_latest": relative_latest,
        "age": age,
        "observed": observed,
        "mask": mask,
        "elevation_raw": elevation_raw,
        "kernel": kernel,
        "window_area": window_area,
        "local_count": local_count,
        "ground": ground,
        "valid": valid,
        "slope": slope,
        "longitudinal_slope": longitudinal_slope,
        "lateral_slope": lateral_slope,
    }


def _compute_world_geometry_fields(
    *,
    cp,
    ndimage,
    elevation_mean,
    elevation_variance,
    elevation_valid,
    geometry_support_count,
    accepted_frame_count,
    ground_sum,
    min_z,
    max_z,
    last_seen,
    variance_updated_at,
    conflict_score,
    conflict_updated_at,
    latest_timestamp: float,
    time_origin: float,
    max_geometry_age_sec: float,
    resolution: float,
    gaussian_config: GaussianElevationConfig,
    elevation_validity=None,
    upper_bound=None,
    upper_bound_valid=None,
    height_drift_bias_z: float = 0.0,
):
    """Derive fixed-world geometry from the per-cell Gaussian posterior."""

    del min_z, max_z  # Retained state diagnostics; mean drives spatial geometry.
    slope_fields = _compute_world_slope_fields(
        cp=cp,
        ndimage=ndimage,
        elevation_mean=elevation_mean,
        elevation_valid=elevation_valid,
        last_seen=last_seen,
        latest_timestamp=latest_timestamp,
        time_origin=time_origin,
        max_geometry_age_sec=max_geometry_age_sec,
        resolution=resolution,
    )
    relative_latest = slope_fields["relative_latest"]
    age = slope_fields["age"]
    observed = slope_fields["observed"]
    if elevation_validity is None:
        elevation_validity = elevation_valid.astype(cp.float32)
    else:
        elevation_validity = cp.asarray(
            elevation_validity, dtype=cp.float32
        )
    if upper_bound is None:
        upper_bound = cp.zeros_like(elevation_mean, dtype=cp.float32)
    else:
        upper_bound = cp.asarray(upper_bound, dtype=cp.float32)
    if upper_bound_valid is None:
        upper_bound_valid = cp.zeros_like(elevation_valid, dtype=cp.bool_)
    else:
        upper_bound_valid = cp.asarray(upper_bound_valid, dtype=cp.bool_)
    for name, values in (
        ("elevation_validity", elevation_validity),
        ("upper_bound", upper_bound),
        ("upper_bound_valid", upper_bound_valid),
    ):
        if values.shape != elevation_mean.shape:
            raise ValueError("{} must match elevation_mean".format(name))
    if not np.isfinite(height_drift_bias_z):
        raise ValueError("height_drift_bias_z must be finite")
    pure_upper_valid = upper_bound_valid & ~observed
    mask = slope_fields["mask"]
    safe_count = cp.maximum(geometry_support_count, 1)
    elevation_raw = slope_fields["elevation_raw"]
    variance_age = cp.maximum(0.0, relative_latest - variance_updated_at)
    render_variance = cp.minimum(
        elevation_variance
        + gaussian_config.process_variance_per_sec * variance_age,
        gaussian_config.max_variance,
    )
    render_variance = cp.where(
        observed, render_variance, gaussian_config.max_variance
    )
    elevation_std = cp.sqrt(render_variance)

    kernel = slope_fields["kernel"]
    window_area = slope_fields["window_area"]
    local_count = slope_fields["local_count"]
    ground = slope_fields["ground"]
    longitudinal_slope = slope_fields["longitudinal_slope"]
    lateral_slope = slope_fields["lateral_slope"]
    slope = slope_fields["slope"]
    local_max = ndimage.maximum_filter(
        cp.where(observed, ground, -1e6), size=3, mode="nearest"
    )
    positive_step = cp.maximum(local_max - ground, 0.0)
    negative_reference = cp.where(
        observed,
        ground,
        cp.where(pure_upper_valid, upper_bound, 1e6),
    )
    local_min_bound = ndimage.minimum_filter(
        negative_reference, size=3, mode="nearest"
    )
    negative_step = cp.maximum(ground - local_min_bound, 0.0)

    half = kernel // 2
    # Accumulate normal equations in float64: subtracting the fitted plane
    # energy from sum(z²) is cancellation-prone for elevated flat surfaces.
    fit_mask = mask.astype(cp.float64)
    fit_elevation = elevation_raw.astype(cp.float64)
    offsets = cp.arange(-half, half + 1, dtype=cp.float64) * float(resolution)
    x_kernel = cp.broadcast_to(offsets[:, None], (kernel, kernel))
    y_kernel = cp.broadcast_to(offsets[None, :], (kernel, kernel))

    def convolve(values, weights):
        return ndimage.convolve(values, weights, mode="constant", cval=0.0)

    ones = cp.ones((kernel, kernel), dtype=cp.float64)
    z_masked = fit_elevation * fit_mask
    s_x = convolve(fit_mask, x_kernel)
    s_y = convolve(fit_mask, y_kernel)
    s_xx = convolve(fit_mask, x_kernel * x_kernel)
    s_yy = convolve(fit_mask, y_kernel * y_kernel)
    s_xy = convolve(fit_mask, x_kernel * y_kernel)
    s_z = convolve(z_masked, ones)
    s_xz = convolve(z_masked, x_kernel)
    s_yz = convolve(z_masked, y_kernel)
    s_zz = convolve(z_masked * fit_elevation, ones)
    determinant = (
        s_xx * (s_yy * local_count - s_y * s_y)
        - s_xy * (s_xy * local_count - s_y * s_x)
        + s_x * (s_xy * s_y - s_yy * s_x)
    )
    rank_valid = cp.abs(determinant) > cp.float64(1e-12)
    safe_det = cp.where(rank_valid, determinant, cp.float64(1.0))
    plane_a = (
        s_xz * (s_yy * local_count - s_y * s_y)
        - s_xy * (s_yz * local_count - s_y * s_z)
        + s_x * (s_yz * s_y - s_yy * s_z)
    ) / safe_det
    plane_b = (
        s_xx * (s_yz * local_count - s_y * s_z)
        - s_xz * (s_xy * local_count - s_y * s_x)
        + s_x * (s_xy * s_z - s_yz * s_x)
    ) / safe_det
    plane_c = (
        s_xx * (s_yy * s_z - s_yz * s_y)
        - s_xy * (s_xy * s_z - s_yz * s_x)
        + s_xz * (s_xy * s_y - s_yy * s_x)
    ) / safe_det
    coverage = local_count / window_area
    ground_fit_valid = (
        observed
        & _plane_fit_is_valid(cp, local_count, window_area, determinant)
    )
    squared_error = cp.maximum(
        s_zz - (plane_a * s_xz + plane_b * s_yz + plane_c * s_z), 0.0
    )
    roughness = cp.where(
        ground_fit_valid,
        cp.sqrt(squared_error / cp.maximum(local_count, 1.0)),
        0.0,
    )
    residual = cp.abs(elevation_raw - ground)
    slope_risk = cp.maximum(
        _smoothstep(cp, np.deg2rad(8.0), np.deg2rad(20.0), slope),
        _smoothstep(cp, np.deg2rad(6.0), np.deg2rad(15.0), lateral_slope),
    )
    roughness_risk = _smoothstep(cp, 0.03, 0.12, roughness)
    positive_step_risk = _smoothstep(cp, 0.05, 0.25, positive_step)
    negative_step_risk = _smoothstep(cp, 0.05, 0.25, negative_step)
    uncertainty_risk = _smoothstep(
        cp,
        gaussian_config.uncertainty_std_good,
        gaussian_config.uncertainty_std_bad,
        elevation_std,
    )
    valid = slope_fields["valid"]
    support_count = cp.where(
        observed, geometry_support_count, 0
    ).astype(cp.int32)
    freshness = cp.exp(-age / float(max_geometry_age_sec))
    variance_confidence = 1.0 - _smoothstep(
        cp,
        gaussian_config.confidence_variance_good,
        gaussian_config.confidence_variance_bad,
        render_variance,
    )
    local_confidence = _smoothstep(
        cp,
        gaussian_config.local_coverage_low,
        gaussian_config.local_coverage_high,
        coverage,
    )
    conflict_age = cp.maximum(0.0, relative_latest - conflict_updated_at)
    decayed_conflict = cp.where(
        cp.isfinite(conflict_updated_at),
        conflict_score * cp.exp(-conflict_age / gaussian_config.conflict_tau_sec),
        0.0,
    )
    frame_confidence = 0.25 + 0.75 * _smoothstep(
        cp, 1.0, 3.0, accepted_frame_count
    )
    confidence = cp.where(
        valid,
        cp.clip(
            variance_confidence
            * freshness
            * local_confidence
            * cp.clip(1.0 - decayed_conflict, 0.0, 1.0)
            * frame_confidence,
            0.0,
            1.0,
        ),
        0.0,
    )
    non_ground_ratio = cp.where(
        observed,
        cp.clip(1.0 - ground_sum / safe_count, 0.0, 1.0),
        0.0,
    )
    layers = (
        slope,
        longitudinal_slope,
        lateral_slope,
        roughness,
        positive_step,
        negative_step,
        slope_risk,
        roughness_risk,
        positive_step_risk,
        negative_step_risk,
        uncertainty_risk,
        residual,
        non_ground_ratio,
    )
    for layer in layers:
        layer[~valid] = 0.0
    risk = cp.maximum(
        cp.maximum(cp.maximum(slope_risk, positive_step_risk), negative_step_risk),
        uncertainty_risk,
    )
    return {
        "observed": observed,
        "geometry_risk": risk.astype(cp.float32),
        "geometry_confidence": confidence.astype(cp.float32),
        "geometry_valid_mask": valid,
        "geometry_support_count": support_count,
        "elevation": elevation_raw.astype(cp.float32),
        "elevation_validity": cp.where(
            observed, elevation_validity, 0.0
        ).astype(cp.float32),
        "elevation_upper_bound": cp.where(
            pure_upper_valid, upper_bound, 0.0
        ).astype(cp.float32),
        "elevation_upper_bound_valid": pure_upper_valid,
        "height_drift_bias_z": cp.full_like(
            elevation_raw,
            float(height_drift_bias_z),
            dtype=cp.float32,
        ),
        "elevation_variance": cp.where(observed, render_variance, 0.0).astype(cp.float32),
        "elevation_std": cp.where(observed, elevation_std, 0.0).astype(cp.float32),
        "slope": slope.astype(cp.float32),
        "longitudinal_slope": longitudinal_slope.astype(cp.float32),
        "lateral_slope": lateral_slope.astype(cp.float32),
        "roughness": roughness.astype(cp.float32),
        "positive_step_height": positive_step.astype(cp.float32),
        "negative_step_height": negative_step.astype(cp.float32),
        "non_ground_ratio": non_ground_ratio.astype(cp.float32),
        "ground_fit_residual": residual.astype(cp.float32),
        "slope_risk": slope_risk.astype(cp.float32),
        "roughness_risk": roughness_risk.astype(cp.float32),
        "positive_step_risk": positive_step_risk.astype(cp.float32),
        "negative_step_risk": negative_step_risk.astype(cp.float32),
        "uncertainty_risk": uncertainty_risk.astype(cp.float32),
        "geometry_age": cp.where(observed, age, 0.0).astype(cp.float32),
        "geometry_conflict_score": cp.where(
            observed, decayed_conflict, 0.0
        ).astype(cp.float32),
        "ground_fit_valid": ground_fit_valid,
    }


@dataclass(frozen=True)
class _GpuModules:
    cp: object
    ndimage: object


def _load_gpu_modules() -> _GpuModules:
    try:
        import cupy as cp
        from cupyx.scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "gpu-rolling-grid requires cupy-cuda11x in the teacher runtime"
        ) from exc
    return _GpuModules(cp=cp, ndimage=ndimage)


def configure_memory_pool_limit(cp, limit_bytes):
    """Optionally cap CuPy's process-global default device memory pool."""

    pool = cp.get_default_memory_pool()
    if limit_bytes is not None:
        limit_bytes = int(limit_bytes)
        if limit_bytes <= 0:
            raise ValueError("memory_pool_limit_bytes must be positive")
        pool.set_limit(size=limit_bytes)
    return pool


def _empty_lifecycle_stats() -> dict[str, object]:
    return {
        "visibility_input_rays": 0,
        "visibility_cleaned_cells": 0,
        "upper_bound_cells": 0,
        "geometry_valid_cells": 0,
        "height_drift_bias_z": 0.0,
        "height_drift_sample_cells": 0,
    }


class GpuRollingElevationMap:
    """Fuse each observation once into a spatially rolling GPU height map."""

    def __init__(
        self,
        *,
        map_spec: MapSpec,
        world_map_length: float = 80.0,
        world_resolution: float | None = None,
        device: str = "cuda:0",
        max_geometry_age_sec: float = 120.0,
        memory_pool_limit_bytes: int | None = None,
        sample_mode: str = "nearest",
        sample_min_valid_weight: float = 0.5,
        risk_histogram_bins: int = 16,
        risk_cvar_alpha: float = 0.8,
        risk_distribution_decay_sec: float | None = None,
        gaussian_config: GaussianElevationConfig | None = None,
        lifecycle_config: ElevationMapLifecycleConfig | None = None,
    ) -> None:
        if world_map_length <= 0.0:
            raise ValueError("world_map_length must be positive")
        if world_map_length < np.hypot(
            map_spec.x_max - map_spec.x_min,
            map_spec.y_max - map_spec.y_min,
        ):
            raise ValueError("world_map_length must contain the rotated output map")
        if max_geometry_age_sec <= 0.0:
            raise ValueError("max_geometry_age_sec must be positive")
        if world_resolution is None:
            world_resolution = map_spec.resolution
        if world_resolution <= 0.0:
            raise ValueError("world_resolution must be positive")
        if world_resolution > map_spec.resolution:
            raise ValueError(
                "world_resolution must not be coarser than output resolution"
            )
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
        if risk_distribution_decay_sec is None:
            risk_distribution_decay_sec = max_geometry_age_sec
        if (
            not np.isfinite(risk_distribution_decay_sec)
            or risk_distribution_decay_sec <= 0.0
        ):
            raise ValueError(
                "risk_distribution_decay_sec must be finite and positive"
            )
        modules = _load_gpu_modules()
        self.cp = modules.cp
        self.ndimage = modules.ndimage
        self.memory_pool = configure_memory_pool_limit(
            self.cp,
            memory_pool_limit_bytes,
        )
        self.map_spec = map_spec
        self.resolution = float(world_resolution)
        self.output_resolution = float(map_spec.resolution)
        self.world_map_length = float(world_map_length)
        self.cell_n = int(np.ceil(world_map_length / self.resolution))
        self.max_geometry_age_sec = float(max_geometry_age_sec)
        self.sample_mode = str(sample_mode)
        self.sample_min_valid_weight = float(sample_min_valid_weight)
        self.risk_histogram_bins = int(risk_histogram_bins)
        self.risk_cvar_alpha = float(risk_cvar_alpha)
        self.risk_distribution_decay_sec = float(risk_distribution_decay_sec)
        self.gaussian_config = (
            GaussianElevationConfig()
            if gaussian_config is None
            else gaussian_config
        )
        if not isinstance(self.gaussian_config, GaussianElevationConfig):
            raise TypeError("gaussian_config must be GaussianElevationConfig")
        self.lifecycle_config = (
            ElevationMapLifecycleConfig()
            if lifecycle_config is None
            else lifecycle_config
        )
        if not isinstance(self.lifecycle_config, ElevationMapLifecycleConfig):
            raise TypeError(
                "lifecycle_config must be ElevationMapLifecycleConfig"
            )
        if (
            self.lifecycle_config.enabled
            and self.lifecycle_config.drift_variance_max
            > self.gaussian_config.max_variance
        ):
            raise ValueError(
                "drift_variance_max must not exceed Gaussian max_variance"
            )
        device_index = int(str(device).split(":")[-1])
        self.device = self.cp.cuda.Device(device_index)
        self.rolling_grid = GpuRollingGrid(
            world_map_length=self.world_map_length,
            resolution=self.resolution,
            device=self.device,
            cp=self.cp,
        )
        self._time_origin: float | None = None
        self._latest_timestamp = -np.inf
        self._ingested_ids: list[str] = []
        self._ingested_timestamps: list[float] = []
        self._ingested_set: set[str] = set()
        self._last_world_geometry_fields = None
        self.height_drift_bias_z = 0.0
        self.height_drift_last_update = -np.inf
        self.last_lifecycle_stats = _empty_lifecycle_stats()
        self.last_geometry_timing = _empty_geometry_timing()
        with self.device:
            shape = (self.cell_n, self.cell_n)
            self.elevation_mean = self.cp.zeros(shape, dtype=self.cp.float32)
            self.elevation_variance = self.cp.full(
                shape, self.gaussian_config.max_variance, dtype=self.cp.float32
            )
            self.elevation_valid = self.cp.zeros(shape, dtype=self.cp.bool_)
            self.geometry_support_count = self.cp.zeros(shape, dtype=self.cp.int32)
            self.accepted_frame_count = self.cp.zeros(shape, dtype=self.cp.int32)
            self.min_z = self.cp.full(shape, self.cp.inf, dtype=self.cp.float32)
            self.max_z = self.cp.full(shape, -self.cp.inf, dtype=self.cp.float32)
            self.ground_sum = self.cp.zeros(shape, dtype=self.cp.float32)
            self.last_seen = self.cp.full(shape, -self.cp.inf, dtype=self.cp.float32)
            self.variance_updated_at = self.cp.full(
                shape, -self.cp.inf, dtype=self.cp.float32
            )
            self.geometry_conflict_score = self.cp.zeros(
                shape, dtype=self.cp.float32
            )
            self.conflict_updated_at = self.cp.full(
                shape, -self.cp.inf, dtype=self.cp.float32
            )
            self.risk_pending_observation = self.cp.zeros(shape, dtype=self.cp.bool_)
            self.risk_sum = self.cp.zeros(shape, dtype=self.cp.float32)
            self.risk_sq_sum = self.cp.zeros(shape, dtype=self.cp.float32)
            self.risk_weight = self.cp.zeros(shape, dtype=self.cp.float32)
            self.risk_last_update = self.cp.full(
                shape, -self.cp.inf, dtype=self.cp.float64
            )
            self.elevation_validity = self.cp.zeros(
                shape, dtype=self.cp.float32
            )
            self.upper_bound = self.cp.zeros(shape, dtype=self.cp.float32)
            self.upper_bound_valid = self.cp.zeros(
                shape, dtype=self.cp.bool_
            )
            self.upper_bound_last_update = self.cp.full(
                shape, -self.cp.inf, dtype=self.cp.float64
            )
            self.risk_histogram = [
                self.cp.zeros(shape, dtype=self.cp.float32)
                for _ in range(self.risk_histogram_bins)
            ]
        geometry_layer_contract = (
            (self.elevation_mean, 0.0),
            (self.elevation_variance, self.gaussian_config.max_variance),
            (self.elevation_valid, False),
            (self.geometry_support_count, 0),
            (self.accepted_frame_count, 0),
            (self.min_z, self.cp.inf),
            (self.max_z, -self.cp.inf),
            (self.ground_sum, 0.0),
            (self.last_seen, -self.cp.inf),
            (self.variance_updated_at, -self.cp.inf),
            (self.geometry_conflict_score, 0.0),
            (self.conflict_updated_at, -self.cp.inf),
            (self.risk_pending_observation, False),
            (self.risk_sum, 0.0),
            (self.risk_sq_sum, 0.0),
            (self.risk_weight, 0.0),
            (self.risk_last_update, -self.cp.inf),
            (self.elevation_validity, 0.0),
            (self.upper_bound, 0.0),
            (self.upper_bound_valid, False),
            (self.upper_bound_last_update, -self.cp.inf),
        )
        self._geometry_mutated_layers = tuple(
            layer for layer, _ in geometry_layer_contract
        ) + tuple(self.risk_histogram)
        for layer, fill in geometry_layer_contract:
            self.rolling_grid.register_layer(layer, fill)
        for histogram in self.risk_histogram:
            self.rolling_grid.register_layer(histogram, 0.0)

    @property
    def _center(self) -> np.ndarray:
        """兼容旧调用方，中心实际由共享滚动栅格维护。"""

        return self.rolling_grid.center

    @property
    def _initialized(self) -> bool:
        return self.rolling_grid.initialized

    @property
    def memory_pool_used_bytes(self) -> int:
        return int(self.memory_pool.used_bytes())

    @property
    def memory_pool_total_bytes(self) -> int:
        return int(self.memory_pool.total_bytes())

    @property
    def ingested_frame_ids(self) -> tuple[str, ...]:
        return tuple(self._ingested_ids)

    @property
    def ingested_timestamps(self) -> tuple[float, ...]:
        return tuple(self._ingested_timestamps)

    def clear(self) -> None:
        # 共享栅格会同时清空已注册的语义图层，保证两种图层不会残留
        # 不同世界坐标系的历史状态。
        self.rolling_grid.clear()
        self._time_origin = None
        self._latest_timestamp = -np.inf
        self._ingested_ids.clear()
        self._ingested_timestamps.clear()
        self._ingested_set.clear()
        self._last_world_geometry_fields = None
        self.height_drift_bias_z = 0.0
        self.height_drift_last_update = -np.inf
        self.last_lifecycle_stats = _empty_lifecycle_stats()
        self.last_geometry_timing = _empty_geometry_timing()

    def world_geometry_snapshot(self) -> dict[str, object] | None:
        """Return the latest geometry fields in the rolling world grid.

        ``_geometry_maps`` computes fields before sampling them into the current
        body grid.  Retaining that GPU result lets the optional world-frame
        publisher export the same aligned state without recomputing geometry.
        The snapshot is intentionally limited to fields needed for risk fusion.
        """

        fields = self._last_world_geometry_fields
        if fields is None:
            return None
        names = (
            "geometry_risk",
            "geometry_confidence",
            "geometry_valid_mask",
            "geometry_support_count",
            "elevation",
            "elevation_validity",
            "elevation_upper_bound",
            "elevation_upper_bound_valid",
            "height_drift_bias_z",
            "elevation_variance",
            "elevation_std",
            "uncertainty_risk",
            "slope_risk",
            "positive_step_risk",
            "negative_step_risk",
            "non_ground_ratio",
            "geometry_risk_std",
            "geometry_risk_cvar",
            "geometry_risk_distribution_weight",
            "geometry_risk_distribution_valid",
        )
        with self.rolling_grid.lock:
            with self.device:
                return {name: self.cp.asnumpy(fields[name]) for name in names}

    def _clear_shifted_edges(self, array, row_shift: int, column_shift: int, value) -> None:
        if row_shift > 0:
            array[:row_shift, :] = value
        elif row_shift < 0:
            array[row_shift:, :] = value
        if column_shift > 0:
            array[:, :column_shift] = value
        elif column_shift < 0:
            array[:, column_shift:] = value

    def _move_center(self, target_xy: np.ndarray) -> None:
        self.rolling_grid.move_center(target_xy)

    def _clear_upper_bound_cells(self, mask) -> None:
        self.upper_bound[mask] = 0.0
        self.upper_bound_valid[mask] = False
        self.upper_bound_last_update[mask] = -self.cp.inf

    def _clear_geometry_cells(self, mask, *, preserve_upper_bound: bool) -> None:
        """Clear one set of Gaussian and risk cells with one shared contract."""

        for layer, fill in (
            (self.elevation_mean, 0.0),
            (self.elevation_variance, self.gaussian_config.max_variance),
            (self.elevation_valid, False),
            (self.elevation_validity, 0.0),
            (self.geometry_support_count, 0),
            (self.accepted_frame_count, 0),
            (self.ground_sum, 0.0),
            (self.min_z, self.cp.inf),
            (self.max_z, -self.cp.inf),
            (self.last_seen, -self.cp.inf),
            (self.variance_updated_at, -self.cp.inf),
            (self.geometry_conflict_score, 0.0),
            (self.conflict_updated_at, -self.cp.inf),
            (self.risk_pending_observation, False),
            (self.risk_sum, 0.0),
            (self.risk_sq_sum, 0.0),
            (self.risk_weight, 0.0),
            (self.risk_last_update, -self.cp.inf),
        ):
            layer[mask] = fill
        for histogram in self.risk_histogram:
            histogram[mask] = 0.0
        if not preserve_upper_bound:
            self._clear_upper_bound_cells(mask)

    def _lifecycle_maintenance_masks(
        self,
        relative_time: float,
        robot_position_world,
        *,
        height_drift_bias_z: float | None = None,
    ):
        shape = self.elevation_valid.shape
        geometry_clear = self.cp.zeros(shape, dtype=self.cp.bool_)
        upper_clear = self.cp.zeros(shape, dtype=self.cp.bool_)
        cfg = self.lifecycle_config
        if not cfg.enabled:
            return geometry_clear, upper_clear

        if cfg.enable_upper_bound:
            upper_clear |= self.upper_bound_valid & self.cp.isfinite(
                self.upper_bound_last_update
            ) & (
                (relative_time - self.upper_bound_last_update)
                > cfg.upper_bound_max_age_sec
            )

        if not cfg.enable_overlap_clearance or robot_position_world is None:
            return geometry_clear, upper_clear
        robot = np.asarray(robot_position_world, dtype=np.float64).reshape(3)
        if not np.isfinite(robot).all():
            raise ValueError("robot_position_world must be finite")
        indices = self.cp.arange(self.cell_n, dtype=self.cp.float32)
        world_axis = (
            self.cp.float32(0.5 * self.world_map_length)
            - (indices + self.cp.float32(0.5)) * self.cp.float32(self.resolution)
        )
        world_x = self.cp.float32(self.rolling_grid.center[0]) + world_axis[:, None]
        world_y = self.cp.float32(self.rolling_grid.center[1]) + world_axis[None, :]
        near_robot = (
            (world_x - self.cp.float32(robot[0])) ** 2
            + (world_y - self.cp.float32(robot[1])) ** 2
        ) <= self.cp.float32(cfg.overlap_clear_radius_m**2)
        drift_bias = (
            self.height_drift_bias_z
            if height_drift_bias_z is None
            else float(height_drift_bias_z)
        )
        if not np.isfinite(drift_bias):
            raise ValueError("height_drift_bias_z must be finite")
        robot_height = self.cp.float32(robot[2] - drift_bias)
        geometry_clear |= (
            near_robot
            & self.elevation_valid
            & (
                self.cp.abs(self.elevation_mean - robot_height)
                > cfg.overlap_clear_range_z_m
            )
        )
        if cfg.enable_upper_bound:
            upper_clear |= (
                near_robot
                & self.upper_bound_valid
                & (
                    self.cp.abs(self.upper_bound - robot_height)
                    > cfg.overlap_clear_range_z_m
                )
            )
        return geometry_clear, upper_clear

    def _metadata_snapshot(self, frame_id: str) -> dict[str, object]:
        return {
            "time_origin": self._time_origin,
            "latest_timestamp": self._latest_timestamp,
            "ingested_ids_length": len(self._ingested_ids),
            "ingested_timestamps_length": len(self._ingested_timestamps),
            "frame_id": frame_id,
            "frame_id_present": frame_id in self._ingested_set,
            "world_fields": self._last_world_geometry_fields,
            "height_drift_bias_z": self.height_drift_bias_z,
            "height_drift_last_update": self.height_drift_last_update,
            "last_lifecycle_stats": dict(self.last_lifecycle_stats),
        }

    def _restore_metadata(self, snapshot: dict[str, object]) -> None:
        self._time_origin = snapshot["time_origin"]
        self._latest_timestamp = snapshot["latest_timestamp"]
        del self._ingested_ids[snapshot["ingested_ids_length"] :]
        del self._ingested_timestamps[
            snapshot["ingested_timestamps_length"] :
        ]
        frame_id = snapshot["frame_id"]
        if snapshot["frame_id_present"]:
            self._ingested_set.add(frame_id)
        else:
            self._ingested_set.discard(frame_id)
        self._last_world_geometry_fields = snapshot["world_fields"]
        self.height_drift_bias_z = snapshot["height_drift_bias_z"]
        self.height_drift_last_update = snapshot["height_drift_last_update"]
        self.last_lifecycle_stats = dict(snapshot["last_lifecycle_stats"])

    def _begin_center_transaction(self, target_xy: np.ndarray) -> dict[str, object]:
        """Move the rolling window while retaining only rollback edge data."""

        grid = self.rolling_grid
        target = np.asarray(target_xy, dtype=np.float64).reshape(2)
        if not np.isfinite(target).all():
            raise ValueError("rolling-grid center must be finite")
        transaction = {
            "center": grid.center,
            "initialized": grid.initialized,
            "shift": (0, 0),
            "mode": "none",
            "layers": (),
        }
        if not grid.initialized:
            self._move_center(target)
            transaction["mode"] = "initialize"
            return transaction

        shift = np.rint((target - grid.center) / self.resolution).astype(np.int64)
        row_shift, column_shift = int(shift[0]), int(shift[1])
        transaction["shift"] = (row_shift, column_shift)
        if row_shift == 0 and column_shift == 0:
            return transaction

        if abs(row_shift) >= self.cell_n or abs(column_shift) >= self.cell_n:
            transaction["mode"] = "full"
            transaction["layers"] = tuple(
                layer.copy() for layer, _ in grid._layers
            )
        else:
            transaction["mode"] = "edges"
            edge_backups = []
            for layer, _ in grid._layers:
                if row_shift > 0:
                    row_edge = layer[-row_shift:, :].copy()
                elif row_shift < 0:
                    row_edge = layer[:-row_shift, :].copy()
                else:
                    row_edge = None
                if column_shift > 0:
                    column_edge = layer[:, -column_shift:].copy()
                elif column_shift < 0:
                    column_edge = layer[:, :-column_shift].copy()
                else:
                    column_edge = None
                edge_backups.append((row_edge, column_edge))
            transaction["layers"] = tuple(edge_backups)
        self._move_center(target)
        return transaction

    def _rollback_center_transaction(self, transaction: dict[str, object]) -> None:
        grid = self.rolling_grid
        mode = transaction["mode"]
        row_shift, column_shift = transaction["shift"]
        if mode == "full":
            for (layer, _), backup in zip(grid._layers, transaction["layers"]):
                layer[...] = backup
        elif mode == "edges":
            for (layer, _), (row_edge, column_edge) in zip(
                grid._layers, transaction["layers"]
            ):
                layer[...] = self.cp.roll(
                    layer,
                    (-row_shift, -column_shift),
                    axis=(0, 1),
                )
                if row_shift > 0:
                    layer[-row_shift:, :] = row_edge
                elif row_shift < 0:
                    layer[:-row_shift, :] = row_edge
                if column_shift > 0:
                    layer[:, -column_shift:] = column_edge
                elif column_shift < 0:
                    layer[:, :-column_shift] = column_edge
        grid._center = np.asarray(transaction["center"], dtype=np.float64).copy()
        grid._initialized = bool(transaction["initialized"])

    def _transactional_ingest_world_arrays(
        self,
        target_xyz: np.ndarray,
        frame_id: str,
        timestamp: float,
        points,
        ground_gpu,
        dynamic_gpu,
        *,
        point_variances=None,
        sensor_origin_world=None,
        robot_position_world=None,
        relative_points=None,
        timer=None,
    ):
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        if not np.isfinite(target).all():
            raise ValueError("target robot position must be finite")
        with _geometry_timing_segment(timer, "geometry_map_center_sec"):
            transaction = self._begin_center_transaction(target[:2])
        try:
            return self._ingest_world_arrays(
                frame_id,
                timestamp,
                points,
                ground_gpu,
                dynamic_gpu,
                point_variances=point_variances,
                sensor_origin_world=sensor_origin_world,
                relative_points=relative_points,
                robot_position_world=(
                    target
                    if robot_position_world is None
                    else robot_position_world
                ),
                timer=timer,
            )
        except Exception:
            self._rollback_center_transaction(transaction)
            raise

    def _validated_new_timestamp(
        self, timestamp: float, *, previous_timestamp: float | None = None
    ) -> float:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("frame timestamp must be finite")
        previous = (
            self._latest_timestamp
            if previous_timestamp is None
            else float(previous_timestamp)
        )
        if np.isfinite(previous) and timestamp <= previous:
            raise ValueError("new frame timestamps must be strictly increasing")
        return timestamp

    def _prepare_offline_frame(self, frame):
        timestamp = float(frame.timestamp)
        observation = frame.observation_world
        xyz = np.asarray(observation.xyz, dtype=np.float32)
        ground = np.asarray(observation.ground_probability, dtype=np.float32)
        dynamic = np.asarray(observation.dynamic, dtype=bool)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError("world point cloud must have shape [N,C>=3]")
        if ground.shape != (xyz.shape[0],) or dynamic.shape != (xyz.shape[0],):
            raise ValueError("world point-cloud metadata must match point count")
        cached_variance = getattr(
            frame,
            "height_variance",
            getattr(observation, "height_variance", None),
        )
        if cached_variance is not None:
            cached_variance = np.asarray(cached_variance, dtype=np.float32)
            if cached_variance.shape != (xyz.shape[0],):
                raise ValueError("point height variance must match point count")
        body_origin = validate_transform(frame.world_from_body)[:3, 3]
        lifecycle_enabled = bool(
            getattr(getattr(self, "lifecycle_config", None), "enabled", False)
        )
        if lifecycle_enabled:
            sensor_origin = getattr(
                frame,
                "sensor_origin_world",
                getattr(observation, "sensor_origin_world", None),
            )
            if sensor_origin is None:
                raise ValueError(
                    "sensor_origin_world is required when elevation-map "
                    "lifecycle is enabled"
                )
            sensor_origin = np.asarray(
                sensor_origin, dtype=np.float32
            ).reshape(3)
            if not np.isfinite(sensor_origin).all():
                raise ValueError("sensor_origin_world must be finite")
            relative_points = getattr(
                observation, "relative_points_lidar", None
            )
            if relative_points is None:
                relative_points = xyz[:, :3] - sensor_origin[None, :]
            relative_points = np.asarray(relative_points, dtype=np.float32)
            if relative_points.shape != (xyz.shape[0], 3):
                raise ValueError("relative LiDAR points must match point count")
            if not np.isfinite(relative_points).all():
                raise ValueError("relative LiDAR points must be finite")
        else:
            sensor_origin = body_origin
            relative_points = xyz[:, :3] - sensor_origin[None, :]
        return (
            timestamp,
            xyz,
            ground,
            dynamic,
            cached_variance,
            body_origin,
            sensor_origin,
            relative_points,
        )

    def _ingest(self, frame_id: str, frame) -> None:
        if frame_id in self._ingested_set:
            return
        (
            timestamp,
            xyz,
            ground,
            dynamic,
            cached_variance,
            body_origin,
            sensor_origin,
            relative_points,
        ) = self._prepare_offline_frame(frame)
        with self.device:
            self._ingest_world_arrays(
                frame_id,
                timestamp,
                self.cp.asarray(xyz),
                self.cp.asarray(ground),
                self.cp.asarray(dynamic),
                point_variances=(
                    None
                    if cached_variance is None
                    else self.cp.asarray(cached_variance, dtype=self.cp.float32)
                ),
                sensor_origin_world=sensor_origin,
                robot_position_world=body_origin,
                relative_points=relative_points,
            )

    def _ingest_world_arrays(
        self,
        frame_id: str,
        timestamp: float,
        points,
        ground_gpu,
        dynamic_gpu,
        point_variances=None,
        sensor_origin_world=None,
        robot_position_world=None,
        relative_points=None,
        timer=None,
    ):
        if self.lifecycle_config.enabled:
            return self._ingest_world_arrays_lifecycle(
                frame_id,
                timestamp,
                points,
                ground_gpu,
                dynamic_gpu,
                point_variances=point_variances,
                sensor_origin_world=sensor_origin_world,
                robot_position_world=robot_position_world,
                relative_points=relative_points,
                timer=timer,
            )
        return self._ingest_world_arrays_baseline(
            frame_id,
            timestamp,
            points,
            ground_gpu,
            dynamic_gpu,
            point_variances=point_variances,
            sensor_origin_world=sensor_origin_world,
            robot_position_world=robot_position_world,
            relative_points=relative_points,
            timer=timer,
        )

    @_close_geometry_timing_on_error
    def _ingest_world_arrays_baseline(
        self,
        frame_id: str,
        timestamp: float,
        points,
        ground_gpu,
        dynamic_gpu,
        point_variances=None,
        sensor_origin_world=None,
        robot_position_world=None,
        relative_points=None,
        timer=None,
    ):
        """Fuse one world-frame observation; caller holds ``self.device``."""

        if frame_id in self._ingested_set:
            return
        _start_geometry_timing(timer, "geometry_prepare_sec")
        timestamp = self._validated_new_timestamp(timestamp)
        candidate_origin = (
            timestamp if self._time_origin is None else self._time_origin
        )
        relative_time = timestamp - candidate_origin
        points = self.cp.asarray(points, dtype=self.cp.float32)
        ground_gpu = self.cp.asarray(ground_gpu, dtype=self.cp.float32)
        dynamic_gpu = self.cp.asarray(dynamic_gpu, dtype=self.cp.bool_)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("GPU point cloud must have shape [N,C>=3]")
        if any(
            array.shape != (points.shape[0],)
            for array in (ground_gpu, dynamic_gpu)
        ):
            raise ValueError("GPU point-cloud metadata must match point count")
        if point_variances is None:
            origin = (
                np.zeros(3, dtype=np.float32)
                if sensor_origin_world is None
                else np.asarray(sensor_origin_world, dtype=np.float32).reshape(3)
            )
            if not np.isfinite(origin).all():
                raise ValueError("sensor_origin_world must be finite")
            delta = points[:, :3] - self.cp.asarray(origin, dtype=self.cp.float32)
            ranges = self.cp.sqrt(self.cp.sum(delta * delta, axis=1))
            point_variances = point_height_variance(
                self.cp, ranges, self.gaussian_config
            )
        else:
            point_variances = self.cp.asarray(
                point_variances, dtype=self.cp.float32
            )
            if point_variances.shape != (points.shape[0],):
                raise ValueError("point height variance must match point count")
        rows, columns, valid = self.rolling_grid.world_indices(points)
        valid &= (
            ~dynamic_gpu
            & self.cp.isfinite(ground_gpu)
            & self.cp.isfinite(point_variances)
            & (point_variances > 0.0)
        )
        flat = rows[valid] * self.cell_n + columns[valid]
        _stop_geometry_timing(timer)
        _start_geometry_timing(timer, "geometry_map_backup_sec")
        metadata_snapshot = self._metadata_snapshot(frame_id)
        touched_flat = None
        touched_backups = ()
        try:
            if flat.size:
                z = points[valid, 2]
                accepted_ground = ground_gpu[valid]
                accepted_variance = point_variances[valid]
                touched_flat = self.cp.unique(flat)
                touched_backups = tuple(
                    layer.ravel()[touched_flat].copy()
                    for layer in self._geometry_mutated_layers
                )
                _switch_geometry_timing(
                    timer, "geometry_map_maintenance_sec"
                )
                touched = self.cp.zeros(
                    self.cell_n * self.cell_n, dtype=self.cp.bool_
                )
                touched[touched_flat] = True
                touched = touched.reshape(self.elevation_valid.shape)
                expired = touched & self.elevation_valid & (
                    (relative_time - self.last_seen) > self.max_geometry_age_sec
                )
                _switch_geometry_timing(timer, "geometry_map_clear_sec")
                for layer, fill in (
                    (self.elevation_mean, 0.0),
                    (self.elevation_variance, self.gaussian_config.max_variance),
                    (self.elevation_valid, False),
                    (self.geometry_support_count, 0),
                    (self.accepted_frame_count, 0),
                    (self.ground_sum, 0.0),
                    (self.min_z, self.cp.inf),
                    (self.max_z, -self.cp.inf),
                    (self.last_seen, -self.cp.inf),
                    (self.variance_updated_at, -self.cp.inf),
                    (self.geometry_conflict_score, 0.0),
                    (self.conflict_updated_at, -self.cp.inf),
                    (self.risk_pending_observation, False),
                    (self.risk_sum, 0.0),
                    (self.risk_sq_sum, 0.0),
                    (self.risk_weight, 0.0),
                    (self.risk_last_update, -self.cp.inf),
                ):
                    layer[expired] = fill
                for histogram in self.risk_histogram:
                    histogram[expired] = 0.0

                _switch_geometry_timing(timer, "geometry_map_fusion_sec")
                update = update_gaussian_elevation(
                    self.cp,
                    flat.astype(self.cp.int64, copy=False),
                    z,
                    accepted_variance,
                    accepted_ground,
                    relative_time,
                    mean=self.elevation_mean.ravel(),
                    variance=self.elevation_variance.ravel(),
                    valid=self.elevation_valid.ravel(),
                    support_count=self.geometry_support_count.ravel(),
                    accepted_frame_count=self.accepted_frame_count.ravel(),
                    last_seen=self.last_seen.ravel(),
                    variance_updated_at=self.variance_updated_at.ravel(),
                    conflict=self.geometry_conflict_score.ravel(),
                    conflict_updated_at=self.conflict_updated_at.ravel(),
                    ground_sum=self.ground_sum.ravel(),
                    min_z=self.min_z.ravel(),
                    max_z=self.max_z.ravel(),
                    cfg=self.gaussian_config,
                    validate_state_values=False,
                )
                self.risk_pending_observation.ravel()[update.touched_cell_ids] = True
            _switch_geometry_timing(timer, "geometry_map_commit_sec")
            self._time_origin = candidate_origin
            self._latest_timestamp = timestamp
            self._ingested_ids.append(frame_id)
            self._ingested_timestamps.append(timestamp)
            self._ingested_set.add(frame_id)
            _stop_geometry_timing(timer)
            return self._consume_pending_risk(timestamp, timer=timer)
        except Exception:
            if touched_flat is not None:
                for layer, backup in zip(
                    self._geometry_mutated_layers, touched_backups
                ):
                    layer.ravel()[touched_flat] = backup
            self._restore_metadata(metadata_snapshot)
            raise

    @_close_geometry_timing_on_error
    def _ingest_world_arrays_lifecycle(
        self,
        frame_id: str,
        timestamp: float,
        points,
        ground_gpu,
        dynamic_gpu,
        point_variances=None,
        sensor_origin_world=None,
        robot_position_world=None,
        relative_points=None,
        timer=None,
    ):
        """Fuse one frame with the full elevation-map lifecycle enabled."""

        if frame_id in self._ingested_set:
            return
        _start_geometry_timing(timer, "geometry_prepare_sec")
        timestamp = self._validated_new_timestamp(timestamp)
        candidate_origin = (
            timestamp if self._time_origin is None else self._time_origin
        )
        relative_time = timestamp - candidate_origin
        cp = self.cp
        cfg = self.lifecycle_config
        points = cp.asarray(points, dtype=cp.float32)
        ground_gpu = cp.asarray(ground_gpu, dtype=cp.float32)
        dynamic_gpu = cp.asarray(dynamic_gpu, dtype=cp.bool_)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("GPU point cloud must have shape [N,C>=3]")
        if any(
            array.shape != (points.shape[0],)
            for array in (ground_gpu, dynamic_gpu)
        ):
            raise ValueError("GPU point-cloud metadata must match point count")

        sensor_origin = None
        if sensor_origin_world is not None:
            sensor_origin = np.asarray(
                sensor_origin_world, dtype=np.float32
            ).reshape(3)
            if not np.isfinite(sensor_origin).all():
                raise ValueError("sensor_origin_world must be finite")
        if points.shape[0] and sensor_origin is None:
            raise ValueError(
                "sensor_origin_world is required when elevation-map lifecycle "
                "is enabled"
            )
        if relative_points is None:
            if points.shape[0]:
                relative_points = points[:, :3] - cp.asarray(
                    sensor_origin, dtype=cp.float32
                )
            else:
                relative_points = cp.empty((0, 3), dtype=cp.float32)
        else:
            relative_points = cp.asarray(relative_points, dtype=cp.float32)
        if relative_points.shape != (points.shape[0], 3):
            raise ValueError("relative LiDAR points must match point count")

        if point_variances is None:
            ranges = cp.sqrt(cp.sum(relative_points * relative_points, axis=1))
            point_variances = point_height_variance(
                cp, ranges, self.gaussian_config
            )
        else:
            point_variances = cp.asarray(point_variances, dtype=cp.float32)
            if point_variances.shape != (points.shape[0],):
                raise ValueError("point height variance must match point count")

        overhead_keep = overhead_keep_mask(cp, relative_points, cfg)
        points = points[overhead_keep]
        relative_points = relative_points[overhead_keep]
        ground_gpu = ground_gpu[overhead_keep]
        dynamic_gpu = dynamic_gpu[overhead_keep]
        point_variances = point_variances[overhead_keep]

        rows, columns, inside_map = self.rolling_grid.world_indices(points)
        measurement_valid = (
            ~dynamic_gpu
            & cp.isfinite(points[:, :3]).all(axis=1)
            & cp.isfinite(ground_gpu)
            & cp.isfinite(point_variances)
            & (point_variances > 0.0)
        )
        map_valid = inside_map & measurement_valid
        map_points = points[map_valid, :3]
        valid_ground = ground_gpu[map_valid]
        valid_variance = point_variances[map_valid]
        valid_flat = (
            rows[map_valid] * self.cell_n + columns[map_valid]
        ).astype(cp.int64, copy=False)

        _stop_geometry_timing(timer)
        _start_geometry_timing(timer, "geometry_drift_sec")
        drift_estimate = None
        if valid_flat.size:
            if (
                self._time_origin is None
                or not cfg.enable_height_drift_compensation
            ):
                prior_slope = cp.zeros_like(
                    self.elevation_mean, dtype=cp.float32
                )
            else:
                prior_slope = self._compute_current_world_slope()
            drift_estimate = estimate_height_drift(
                cp,
                valid_flat,
                map_points[:, 2],
                valid_variance,
                self.elevation_mean.ravel(),
                self.elevation_variance.ravel(),
                self.elevation_validity.ravel(),
                self.elevation_valid.ravel(),
                self.accepted_frame_count.ravel(),
                prior_slope.ravel(),
                self.height_drift_bias_z,
                cfg,
            )
            corrected_bias = drift_estimate.new_bias
        else:
            corrected_bias = self.height_drift_bias_z
        corrected_points = points[:, :3].copy()
        corrected_points[:, 2] -= cp.float32(corrected_bias)
        valid_points = corrected_points[map_valid]

        _stop_geometry_timing(timer)
        _start_geometry_timing(timer, "geometry_visibility_sec")
        wall_selection = wall_edge_keep_mask(
            cp,
            valid_flat,
            valid_points[:, 2],
            self.elevation_mean.ravel(),
            self.elevation_valid.ravel(),
            cfg,
        )
        point_keep = wall_selection.keep_mask
        accepted_flat = valid_flat[point_keep]
        accepted_points = valid_points[point_keep]
        accepted_ground = valid_ground[point_keep]
        accepted_variance = valid_variance[point_keep]
        endpoint_cells = cp.unique(accepted_flat)
        visibility_enabled = cfg.enable_visibility_cleanup or cfg.enable_upper_bound

        empty_ids = cp.empty((0,), dtype=cp.int64)
        empty_values = cp.empty((0,), dtype=cp.float32)
        evidence = VisibilityEvidence(
            upper_bound_cell_ids=empty_ids,
            upper_bound_candidates=empty_values,
            penetrated_cell_ids=empty_ids.copy(),
        )
        visibility_input_rays = 0
        if visibility_enabled:
            ray_keep = measurement_valid.copy()
            ray_keep[cp.flatnonzero(map_valid)] = point_keep
            ray_endpoints = corrected_points[ray_keep]
        if visibility_enabled and ray_endpoints.size:
            corrected_sensor_origin = cp.asarray(
                sensor_origin, dtype=cp.float32
            ).copy()
            corrected_sensor_origin[2] -= cp.float32(corrected_bias)
            endpoints = select_visibility_endpoints(
                cp,
                ray_endpoints,
                cfg.max_visibility_rays_per_frame,
            )
            visibility_input_rays = int(endpoints.shape[0])

            def world_xy_to_flat_ids(world_x, world_y):
                ray_rows, ray_columns, in_map = (
                    self.rolling_grid.world_xy_indices(world_x, world_y)
                )
                return (
                    ray_rows * self.cell_n + ray_columns,
                    in_map,
                )

            samples = raycast_grid_samples(
                cp,
                corrected_sensor_origin,
                endpoints,
                self.resolution,
                cfg.visibility_max_ray_length_m,
                cfg.visibility_endpoint_margin_m,
                world_xy_to_flat_ids,
            )
            evidence = visibility_evidence(
                cp,
                samples,
                self.elevation_mean.ravel(),
                self.elevation_variance.ravel(),
                self.elevation_valid.ravel(),
                self.last_seen.ravel(),
                map_shape=self.elevation_mean.shape,
                resolution=self.resolution,
                timestamp=relative_time,
                cfg=cfg,
            )
            if evidence.penetrated_cell_ids.size and endpoint_cells.size:
                penetration_keep = ~cp.isin(
                    evidence.penetrated_cell_ids,
                    endpoint_cells,
                )
                evidence = VisibilityEvidence(
                    upper_bound_cell_ids=evidence.upper_bound_cell_ids,
                    upper_bound_candidates=evidence.upper_bound_candidates,
                    penetrated_cell_ids=evidence.penetrated_cell_ids[
                        penetration_keep
                    ],
                )

        _stop_geometry_timing(timer)
        _start_geometry_timing(timer, "geometry_map_maintenance_sec")
        geometry_clear, upper_clear = self._lifecycle_maintenance_masks(
            relative_time,
            robot_position_world,
            height_drift_bias_z=corrected_bias,
        )
        if endpoint_cells.size:
            endpoint_mask = cp.zeros(
                self.cell_n * self.cell_n, dtype=cp.bool_
            )
            endpoint_mask[endpoint_cells] = True
            endpoint_mask = endpoint_mask.reshape(self.elevation_valid.shape)
            geometry_clear |= (
                endpoint_mask
                & self.elevation_valid
                & (
                    (relative_time - self.last_seen)
                    > self.max_geometry_age_sec
                )
            )

        _switch_geometry_timing(timer, "geometry_map_backup_sec")
        changed_parts = (
            accepted_flat,
            cp.flatnonzero(geometry_clear | upper_clear).astype(
                cp.int64, copy=False
            ),
            evidence.upper_bound_cell_ids.astype(cp.int64, copy=False),
            evidence.penetrated_cell_ids.astype(cp.int64, copy=False),
        )
        nonempty_parts = tuple(part for part in changed_parts if part.size)
        touched_flat = (
            cp.unique(cp.concatenate(nonempty_parts))
            if nonempty_parts
            else empty_ids
        )
        metadata_snapshot = self._metadata_snapshot(frame_id)
        touched_backups = tuple(
            layer.ravel()[touched_flat].copy()
            for layer in self._geometry_mutated_layers
        )
        clear_layers = (
            (self.geometry_support_count.ravel(), 0),
            (self.accepted_frame_count.ravel(), 0),
            (self.last_seen.ravel(), -cp.inf),
            (self.variance_updated_at.ravel(), -cp.inf),
            (self.geometry_conflict_score.ravel(), 0.0),
            (self.conflict_updated_at.ravel(), -cp.inf),
            (self.ground_sum.ravel(), 0.0),
            (self.min_z.ravel(), cp.inf),
            (self.max_z.ravel(), -cp.inf),
            (self.risk_pending_observation.ravel(), False),
            (self.risk_sum.ravel(), 0.0),
            (self.risk_sq_sum.ravel(), 0.0),
            (self.risk_weight.ravel(), 0.0),
            (self.risk_last_update.ravel(), -cp.inf),
        ) + tuple((histogram.ravel(), 0.0) for histogram in self.risk_histogram)
        try:
            _switch_geometry_timing(timer, "geometry_map_clear_sec")
            self._clear_geometry_cells(
                geometry_clear,
                preserve_upper_bound=False,
            )
            upper_clear &= ~geometry_clear
            self._clear_upper_bound_cells(upper_clear)

            _switch_geometry_timing(
                timer, "geometry_map_visibility_apply_sec"
            )
            visibility_cleaned_cells = 0
            if visibility_enabled:
                visibility_update = apply_visibility_and_upper_bounds(
                    cp,
                    evidence,
                    empty_ids,
                    self.elevation_mean.ravel(),
                    self.elevation_variance.ravel(),
                    self.elevation_valid.ravel(),
                    self.elevation_validity.ravel(),
                    self.upper_bound.ravel(),
                    self.upper_bound_valid.ravel(),
                    self.upper_bound_last_update.ravel(),
                    timestamp=relative_time,
                    max_variance=self.gaussian_config.max_variance,
                    outlier_variance=(
                        self.gaussian_config.outlier_variance_per_frame
                    ),
                    cfg=cfg,
                    clear_layers=clear_layers,
                )
                visibility_cleaned_cells = int(
                    visibility_update.cleared_cell_ids.size
                )

            _switch_geometry_timing(
                timer, "geometry_map_maintenance_sec"
            )
            _, post_visibility_upper_clear = (
                self._lifecycle_maintenance_masks(
                    relative_time,
                    robot_position_world,
                    height_drift_bias_z=corrected_bias,
                )
            )
            _switch_geometry_timing(timer, "geometry_map_clear_sec")
            self._clear_upper_bound_cells(post_visibility_upper_clear)

            if accepted_flat.size:
                _switch_geometry_timing(timer, "geometry_map_fusion_sec")
                update = update_gaussian_elevation(
                    cp,
                    accepted_flat,
                    accepted_points[:, 2],
                    accepted_variance,
                    accepted_ground,
                    relative_time,
                    mean=self.elevation_mean.ravel(),
                    variance=self.elevation_variance.ravel(),
                    valid=self.elevation_valid.ravel(),
                    support_count=self.geometry_support_count.ravel(),
                    accepted_frame_count=self.accepted_frame_count.ravel(),
                    last_seen=self.last_seen.ravel(),
                    variance_updated_at=self.variance_updated_at.ravel(),
                    conflict=self.geometry_conflict_score.ravel(),
                    conflict_updated_at=self.conflict_updated_at.ravel(),
                    ground_sum=self.ground_sum.ravel(),
                    min_z=self.min_z.ravel(),
                    max_z=self.max_z.ravel(),
                    cfg=self.gaussian_config,
                    validate_state_values=False,
                )
                accepted_evidence = VisibilityEvidence(
                    upper_bound_cell_ids=empty_ids,
                    upper_bound_candidates=empty_values,
                    penetrated_cell_ids=empty_ids,
                )
                apply_visibility_and_upper_bounds(
                    cp,
                    accepted_evidence,
                    update.accepted_cell_ids,
                    self.elevation_mean.ravel(),
                    self.elevation_variance.ravel(),
                    self.elevation_valid.ravel(),
                    self.elevation_validity.ravel(),
                    self.upper_bound.ravel(),
                    self.upper_bound_valid.ravel(),
                    self.upper_bound_last_update.ravel(),
                    timestamp=relative_time,
                    max_variance=self.gaussian_config.max_variance,
                    outlier_variance=(
                        self.gaussian_config.outlier_variance_per_frame
                    ),
                    cfg=cfg,
                )
                self.risk_pending_observation.ravel()[
                    update.touched_cell_ids
                ] = True

            _switch_geometry_timing(timer, "geometry_map_commit_sec")
            if drift_estimate is not None:
                self.height_drift_bias_z = drift_estimate.new_bias
                if drift_estimate.sample_count >= cfg.drift_min_cells:
                    self.height_drift_last_update = relative_time
            upper_count = cp.count_nonzero(self.upper_bound_valid)
            geometry_count = cp.count_nonzero(self.elevation_valid)
            self.last_lifecycle_stats = {
                "visibility_input_rays": visibility_input_rays,
                "visibility_cleaned_cells": visibility_cleaned_cells,
                "upper_bound_cells": int(
                    upper_count.item()
                    if hasattr(upper_count, "item")
                    else upper_count
                ),
                "geometry_valid_cells": int(
                    geometry_count.item()
                    if hasattr(geometry_count, "item")
                    else geometry_count
                ),
                "height_drift_bias_z": float(self.height_drift_bias_z),
                "height_drift_sample_cells": (
                    0 if drift_estimate is None else drift_estimate.sample_count
                ),
            }
            self._time_origin = candidate_origin
            self._latest_timestamp = timestamp
            self._ingested_ids.append(frame_id)
            self._ingested_timestamps.append(timestamp)
            self._ingested_set.add(frame_id)
            _stop_geometry_timing(timer)
            return self._consume_pending_risk(timestamp, timer=timer)
        except Exception:
            for layer, backup in zip(
                self._geometry_mutated_layers, touched_backups
            ):
                layer.ravel()[touched_flat] = backup
            self._restore_metadata(metadata_snapshot)
            raise

    def _prepare_lidar_arrays(
        self,
        points_lidar,
        intensity,
        *,
        world_from_lidar: np.ndarray,
        minimum_range: float,
        maximum_range: float,
        minimum_forward_x: float | None = None,
        row_world_from_lidar=None,
        point_rows=None,
    ):
        """Validate, filter and transform LiDAR arrays without map mutation.

        ``row_world_from_lidar`` is optional and contains one pose per organized
        VLS row.  When present, the point-row IDs select these transforms on the
        GPU, allowing scan-internal motion compensation without a CPU point copy.
        """

        if not 0.0 <= minimum_range < maximum_range:
            raise ValueError("point-cloud range limits are invalid")
        if minimum_forward_x is not None and not np.isfinite(minimum_forward_x):
            raise ValueError("minimum_forward_x must be finite when provided")
        points = self.cp.asarray(points_lidar, dtype=self.cp.float32)
        intensity = self.cp.asarray(intensity, dtype=self.cp.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("GPU LiDAR points must have shape [N,3]")
        if intensity.shape != (points.shape[0],):
            raise ValueError("GPU intensity must match LiDAR point count")
        if (row_world_from_lidar is None) != (point_rows is None):
            raise ValueError(
                "row_world_from_lidar and point_rows must be provided together"
            )
        if point_rows is not None:
            point_rows = self.cp.asarray(point_rows, dtype=self.cp.int32)
            row_world_from_lidar = self.cp.asarray(
                row_world_from_lidar, dtype=self.cp.float32
            )
            if point_rows.shape != (points.shape[0],):
                raise ValueError("point_rows must match the point count")
            if row_world_from_lidar.ndim != 3 or row_world_from_lidar.shape[1:] != (
                4,
                4,
            ):
                raise ValueError("row_world_from_lidar must have shape [H,4,4]")
        finite = self.cp.isfinite(points).all(axis=1) & self.cp.isfinite(intensity)
        range_squared = self.cp.sum(points * points, axis=1)
        valid = (
            finite
            & (range_squared >= self.cp.float32(minimum_range * minimum_range))
            & (range_squared <= self.cp.float32(maximum_range * maximum_range))
        )
        if minimum_forward_x is not None:
            valid &= points[:, 0] >= self.cp.float32(minimum_forward_x)
        point_variances = point_height_variance(
            self.cp,
            self.cp.sqrt(range_squared[valid]),
            self.gaussian_config,
        )
        relative_points = points[valid]
        if point_rows is not None:
            point_rows = point_rows[valid]
        points = relative_points
        if points.size:
            if point_rows is None:
                transform = self.cp.asarray(world_from_lidar, dtype=self.cp.float32)
                points = points @ transform[:3, :3].T + transform[:3, 3]
            else:
                transform = row_world_from_lidar[point_rows]
                points = self.cp.einsum(
                    "nij,nj->ni", transform[:, :3, :3], points
                ) + transform[:, :3, 3]
        ground = self.cp.ones(points.shape[0], dtype=self.cp.float32)
        dynamic = self.cp.zeros(points.shape[0], dtype=self.cp.bool_)
        sensor_origin = np.asarray(world_from_lidar, dtype=np.float32)[:3, 3]
        return (
            points,
            ground,
            dynamic,
            point_variances,
            relative_points,
            sensor_origin,
        )

    def _ingest_lidar_arrays(
        self,
        frame_id: str,
        timestamp: float,
        points_lidar,
        intensity,
        *,
        world_from_lidar: np.ndarray,
        minimum_range: float,
        maximum_range: float,
        minimum_forward_x: float | None = None,
        row_world_from_lidar=None,
        point_rows=None,
    ) -> None:
        if frame_id in self._ingested_set:
            return
        timestamp = self._validated_new_timestamp(timestamp)
        (
            points,
            ground,
            dynamic,
            point_variances,
            relative_points,
            sensor_origin,
        ) = self._prepare_lidar_arrays(
            points_lidar,
            intensity,
            world_from_lidar=world_from_lidar,
            minimum_range=minimum_range,
            maximum_range=maximum_range,
            minimum_forward_x=minimum_forward_x,
            row_world_from_lidar=row_world_from_lidar,
            point_rows=point_rows,
        )
        self._ingest_world_arrays(
            frame_id,
            timestamp,
            points,
            ground,
            dynamic,
            point_variances=point_variances,
            sensor_origin_world=sensor_origin,
            relative_points=relative_points,
        )

    def _body_world_coordinates(self, target_world_from_body: np.ndarray):
        cp = self.cp
        transform = validate_transform(target_world_from_body)
        height, width = self.map_spec.shape
        rows, columns = cp.meshgrid(
            cp.arange(height, dtype=cp.float32),
            cp.arange(width, dtype=cp.float32),
            indexing="ij",
        )
        body_x = self.map_spec.x_max - (rows + 0.5) * self.output_resolution
        body_y = self.map_spec.y_max - (columns + 0.5) * self.output_resolution
        rotation = cp.asarray(transform[:3, :3], dtype=cp.float32)
        translation = cp.asarray(transform[:3, 3], dtype=cp.float32)
        world_x = (
            rotation[0, 0] * body_x
            + rotation[0, 1] * body_y
            + translation[0]
        )
        world_y = (
            rotation[1, 0] * body_x
            + rotation[1, 1] * body_y
            + translation[1]
        )
        return transform, world_x, world_y, rotation, translation

    def _sample_current_body(
        self,
        world_fields: dict[str, object],
        target_world_from_body: np.ndarray,
    ) -> dict[str, object]:
        """Sample pose-independent world geometry layers into the body grid."""

        cp = self.cp
        transform, world_x, world_y, _, translation = (
            self._body_world_coordinates(target_world_from_body)
        )
        if self.sample_mode == "nearest":
            map_rows, map_columns, inside = self.rolling_grid.world_xy_indices(
                world_x,
                world_y,
            )
            safe_rows = cp.clip(map_rows, 0, self.cell_n - 1)
            safe_columns = cp.clip(map_columns, 0, self.cell_n - 1)
            sample_context = (
                map_rows,
                map_columns,
                inside,
                safe_rows,
                safe_columns,
            )
        else:
            map_rows, map_columns, weights, _ = (
                bilinear_world_xy_samples(
                    world_x,
                    world_y,
                    center=self.rolling_grid.center,
                    world_map_length=self.world_map_length,
                    resolution=self.resolution,
                    cell_n=self.cell_n,
                    xp=cp,
                )
            )
            sample_context = (map_rows, map_columns, weights)

        def sample(values, support, fill_value=0.0):
            if self.sample_mode == "nearest":
                (
                    map_rows,
                    map_columns,
                    inside,
                    safe_rows,
                    safe_columns,
                ) = sample_context
                supported = inside & support[safe_rows, safe_columns]
                return cp.where(
                    supported,
                    values[safe_rows, safe_columns],
                    fill_value,
                )
            map_rows, map_columns, weights = sample_context
            return weighted_neighbor_sample(
                values,
                map_rows,
                map_columns,
                weights,
                support=support,
                min_valid_weight=self.sample_min_valid_weight,
                xp=cp,
                fill_value=fill_value,
            )[0]

        def sample_mask(support):
            if self.sample_mode == "nearest":
                (
                    map_rows,
                    map_columns,
                    inside,
                    safe_rows,
                    safe_columns,
                ) = sample_context
                return inside & support[safe_rows, safe_columns]
            values = cp.ones_like(support, dtype=cp.float32)
            map_rows, map_columns, weights = sample_context
            _, _, valid = weighted_neighbor_sample(
                values,
                map_rows,
                map_columns,
                weights,
                support=support,
                min_valid_weight=self.sample_min_valid_weight,
                xp=cp,
            )
            return valid

        observed = world_fields["observed"]
        valid = world_fields["geometry_valid_mask"]
        distribution_valid = world_fields.get(
            "geometry_risk_distribution_valid", valid
        )
        sampled_observed = sample_mask(observed)
        sampled_valid = sample_mask(valid)
        sampled_distribution_valid = sample_mask(distribution_valid)
        sampled_elevation_world = sample(
            world_fields["elevation"],
            observed,
            fill_value=cp.float32(0.0),
        )
        sampled_support_count = sample(
            world_fields["geometry_support_count"],
            observed,
            fill_value=cp.float32(0.0),
        ).astype(cp.int32)
        sampled_variance = sample(
            world_fields["elevation_variance"],
            observed,
            fill_value=cp.float32(0.0),
        ).astype(cp.float32)
        sampled_std = cp.sqrt(cp.maximum(sampled_variance, 0.0)).astype(
            cp.float32
        )
        sampled_conflict = sample(
            world_fields["geometry_conflict_score"],
            observed,
            fill_value=cp.float32(0.0),
        ).astype(cp.float32)
        sampled_ground_fit_valid = sample_mask(world_fields["ground_fit_valid"])
        upper_valid = (
            world_fields["elevation_upper_bound_valid"]
            if "elevation_upper_bound_valid" in world_fields
            else cp.zeros_like(observed, dtype=cp.bool_)
        )
        sampled_upper_valid = sample_mask(upper_valid)
        sampled_upper_world = sample(
            (
                world_fields["elevation_upper_bound"]
                if "elevation_upper_bound" in world_fields
                else cp.zeros_like(
                    world_fields["elevation"], dtype=cp.float32
                )
            ),
            upper_valid,
            fill_value=cp.float32(0.0),
        ).astype(cp.float32)
        sampled_elevation_validity = sample(
            (
                world_fields["elevation_validity"]
                if "elevation_validity" in world_fields
                else observed.astype(cp.float32)
            ),
            observed,
            fill_value=cp.float32(0.0),
        ).astype(cp.float32)
        sampled_drift_bias = sample(
            (
                world_fields["height_drift_bias_z"]
                if "height_drift_bias_z" in world_fields
                else cp.zeros_like(
                    world_fields["elevation"], dtype=cp.float32
                )
            ),
            cp.ones_like(observed, dtype=cp.bool_),
            fill_value=cp.float32(0.0),
        ).astype(cp.float32)

        sampled = {}
        for name, values in world_fields.items():
            if name in {
                "observed",
                "elevation",
                "elevation_variance",
                "elevation_std",
                "geometry_support_count",
                "geometry_conflict_score",
                "ground_fit_valid",
                "geometry_risk_distribution_valid",
                "elevation_validity",
                "elevation_upper_bound",
                "elevation_upper_bound_valid",
                "height_drift_bias_z",
            }:
                continue
            support = (
                distribution_valid
                if name
                in {
                    "geometry_risk_std",
                    "geometry_risk_cvar",
                    "geometry_risk_distribution_weight",
                }
                else valid
            )
            sampled[name] = sample(
                values,
                support,
                fill_value=cp.float32(0.0),
            ).astype(cp.float32)
        target_from_world = cp.asarray(
            np.linalg.inv(transform), dtype=cp.float32
        )

        def body_height(world_z):
            dx = world_x - translation[0]
            dy = world_y - translation[1]
            # Lifecycle observations and ray bounds live in the Z-corrected
            # world frame.  Apply the same correction to the body origin
            # before converting heights back to body coordinates; otherwise
            # the reported ground height is lowered once more by the full
            # accumulated drift bias.
            dz = world_z - (translation[2] - sampled_drift_bias)
            return (
                target_from_world[2, 0] * dx
                + target_from_world[2, 1] * dy
                + target_from_world[2, 2] * dz
            )

        sampled["elevation"] = cp.where(
            sampled_observed,
            body_height(sampled_elevation_world),
            cp.float32(0.0),
        ).astype(cp.float32)
        sampled["elevation_validity"] = sampled_elevation_validity
        sampled["elevation_upper_bound"] = cp.where(
            sampled_upper_valid,
            body_height(sampled_upper_world),
            cp.float32(0.0),
        ).astype(cp.float32)
        sampled["elevation_upper_bound_valid"] = sampled_upper_valid
        sampled["height_drift_bias_z"] = sampled_drift_bias
        sampled["geometry_valid_mask"] = sampled_valid
        sampled["geometry_risk_distribution_valid"] = sampled_distribution_valid
        sampled["geometry_support_count"] = sampled_support_count
        sampled["elevation_variance"] = sampled_variance
        sampled["elevation_std"] = sampled_std
        sampled["geometry_age"] = sample(
            world_fields["geometry_age"],
            observed,
            fill_value=cp.float32(0.0),
        ).astype(cp.float32)
        sampled["geometry_conflict_score"] = sampled_conflict
        sampled["ground_fit_valid"] = sampled_ground_fit_valid
        return sampled

    def _compute_current_world_fields(self):
        if self._time_origin is None:
            raise ValueError("GPU rolling elevation map has no time origin")
        return _compute_world_geometry_fields(
            cp=self.cp,
            ndimage=self.ndimage,
            elevation_mean=self.elevation_mean,
            elevation_variance=self.elevation_variance,
            elevation_valid=self.elevation_valid,
            geometry_support_count=self.geometry_support_count,
            accepted_frame_count=self.accepted_frame_count,
            ground_sum=self.ground_sum,
            min_z=self.min_z,
            max_z=self.max_z,
            last_seen=self.last_seen,
            variance_updated_at=self.variance_updated_at,
            conflict_score=self.geometry_conflict_score,
            conflict_updated_at=self.conflict_updated_at,
            latest_timestamp=self._latest_timestamp,
            time_origin=self._time_origin,
            max_geometry_age_sec=self.max_geometry_age_sec,
            resolution=self.resolution,
            gaussian_config=self.gaussian_config,
            elevation_validity=self.elevation_validity,
            upper_bound=self.upper_bound,
            upper_bound_valid=self.upper_bound_valid,
            height_drift_bias_z=self.height_drift_bias_z,
        )

    def _compute_current_world_slope(self):
        if self._time_origin is None:
            raise ValueError("GPU rolling elevation map has no time origin")
        return _compute_world_slope_fields(
            cp=self.cp,
            ndimage=self.ndimage,
            elevation_mean=self.elevation_mean,
            elevation_valid=self.elevation_valid,
            last_seen=self.last_seen,
            latest_timestamp=self._latest_timestamp,
            time_origin=self._time_origin,
            max_geometry_age_sec=self.max_geometry_age_sec,
            resolution=self.resolution,
        )["slope"]

    def _consume_pending_risk(self, timestamp: float, *, timer=None):
        """Consume exactly one geometry-risk observation for the current frame."""

        with _geometry_timing_segment(timer, "geometry_world_fields_sec"):
            world_fields = self._compute_current_world_fields()
        with _geometry_timing_segment(timer, "geometry_risk_sec"):
            _update_risk_distribution(
                cp=self.cp,
                risk_sum=self.risk_sum,
                risk_sq_sum=self.risk_sq_sum,
                risk_weight=self.risk_weight,
                risk_histogram=self.risk_histogram,
                risk_last_update=self.risk_last_update,
                risk=world_fields["geometry_risk"],
                confidence=world_fields["geometry_confidence"],
                valid=(
                    world_fields["geometry_valid_mask"]
                    & self.risk_pending_observation
                ),
                timestamp=float(timestamp),
                decay_tau_sec=self.risk_distribution_decay_sec,
            )
        self.risk_pending_observation.fill(False)
        return world_fields

    def _geometry_maps(
        self,
        target_world_from_body: np.ndarray,
        world_fields=None,
        *,
        timer=None,
    ) -> dict[str, np.ndarray]:
        cp = self.cp
        if world_fields is None:
            with _geometry_timing_segment(timer, "geometry_world_fields_sec"):
                world_fields = self._compute_current_world_fields()
        with _geometry_timing_segment(timer, "geometry_risk_sec"):
            distribution_fields = _compute_risk_distribution_fields(
                cp=cp,
                risk_sum=self.risk_sum,
                risk_sq_sum=self.risk_sq_sum,
                risk_weight=self.risk_weight,
                risk_histogram=self.risk_histogram,
                cvar_alpha=self.risk_cvar_alpha,
            )
            distribution_valid = (
                distribution_fields["geometry_risk_distribution_valid"]
                & world_fields["geometry_valid_mask"]
            )
            distribution_fields["geometry_risk_distribution_valid"] = (
                distribution_valid
            )
            for name in (
                "geometry_risk_mean",
                "geometry_risk_std",
                "geometry_risk_cvar",
                "geometry_risk_distribution_weight",
            ):
                distribution_fields[name] = cp.where(
                    distribution_valid,
                    distribution_fields[name],
                    cp.float32(0.0),
                )
        world_fields.update(distribution_fields)
        self._last_world_geometry_fields = world_fields
        with _geometry_timing_segment(timer, "geometry_sample_sec"):
            result_gpu = self._sample_current_body(
                world_fields,
                target_world_from_body,
            )
        with _geometry_timing_segment(timer, "geometry_transfer_sec"):
            return {
                name: cp.asnumpy(values)
                for name, values in result_gpu.items()
            }

    def advance(
        self,
        frame_ids: Iterable[str],
        frame_loader: Callable[[str], object],
        *,
        target_world_from_body: np.ndarray,
    ) -> dict[str, np.ndarray]:
        timer = _GeometryStageTimer(cp=self.cp, device=self.device)
        desired = tuple(frame_ids)
        if not desired:
            raise ValueError("GPU rolling elevation target has no frames")
        transform = validate_transform(target_world_from_body)
        with self.rolling_grid.lock:
            latest_world_fields = None
            for frame_id in desired:
                if frame_id in self._ingested_set:
                    continue
                frame = frame_loader(frame_id)
                with _geometry_timing_segment(timer, "geometry_prepare_sec"):
                    (
                        timestamp,
                        xyz,
                        ground,
                        dynamic,
                        cached_variance,
                        body_origin,
                        sensor_origin,
                        relative_points,
                    ) = self._prepare_offline_frame(frame)
                timestamp = self._validated_new_timestamp(timestamp)
                with self.device:
                    with _geometry_timing_segment(
                        timer, "geometry_prepare_sec"
                    ):
                        points_gpu = self.cp.asarray(xyz)
                        ground_gpu = self.cp.asarray(ground)
                        dynamic_gpu = self.cp.asarray(dynamic)
                        variance_gpu = (
                            None
                            if cached_variance is None
                            else self.cp.asarray(
                                cached_variance, dtype=self.cp.float32
                            )
                        )
                        relative_points_gpu = self.cp.asarray(
                            relative_points, dtype=self.cp.float32
                        )
                    latest_world_fields = self._transactional_ingest_world_arrays(
                        # 多楼层清理始终围绕当前输出目标位姿；历史帧的
                        # sensor_origin 仍保留各历史帧真实射线起点。
                        transform[:3, 3],
                        frame_id,
                        timestamp,
                        points_gpu,
                        ground_gpu,
                        dynamic_gpu,
                        point_variances=variance_gpu,
                        sensor_origin_world=sensor_origin,
                        relative_points=relative_points_gpu,
                        timer=timer,
                    )
                del (
                    frame,
                    xyz,
                    ground,
                    dynamic,
                    cached_variance,
                    body_origin,
                    sensor_origin,
                    relative_points,
                )
            if not self._ingested_ids:
                raise ValueError("GPU rolling elevation map contains no observations")
            with self.device:
                result = self._geometry_maps(
                    transform, latest_world_fields, timer=timer
                )
            timing = timer.finish()
            self.last_geometry_timing = timing
            return result

    def advance_pointcloud(
        self,
        frame_id: str,
        timestamp: float,
        points_lidar,
        intensity,
        *,
        world_from_lidar: np.ndarray,
        minimum_range: float,
        maximum_range: float,
        minimum_forward_x: float | None = None,
        target_world_from_body: np.ndarray,
        row_world_from_lidar=None,
        point_rows=None,
    ) -> dict[str, np.ndarray]:
        """Update from GPU-decoded LiDAR arrays without a CPU point copy."""

        timer = _GeometryStageTimer(cp=self.cp, device=self.device)
        transform = validate_transform(target_world_from_body)
        with self.rolling_grid.lock:
            if frame_id in self._ingested_set:
                with self.device:
                    result = self._geometry_maps(transform, timer=timer)
                timing = timer.finish()
                self.last_geometry_timing = timing
                return result
            timestamp = self._validated_new_timestamp(timestamp)
            with self.device:
                with _geometry_timing_segment(timer, "geometry_prepare_sec"):
                    (
                        points,
                        ground,
                        dynamic,
                        point_variances,
                        relative_points,
                        sensor_origin,
                    ) = (
                    self._prepare_lidar_arrays(
                        points_lidar,
                        intensity,
                        world_from_lidar=world_from_lidar,
                        minimum_range=minimum_range,
                        maximum_range=maximum_range,
                        minimum_forward_x=minimum_forward_x,
                        row_world_from_lidar=row_world_from_lidar,
                        point_rows=point_rows,
                    )
                    )
            with self.device:
                world_fields = self._transactional_ingest_world_arrays(
                    transform[:3, 3],
                    frame_id,
                    timestamp,
                    points,
                    ground,
                    dynamic,
                    point_variances=point_variances,
                    sensor_origin_world=sensor_origin,
                    relative_points=relative_points,
                    timer=timer,
                )
                result = self._geometry_maps(
                    transform, world_fields, timer=timer
                )
            timing = timer.finish()
            self.last_geometry_timing = timing
            return result
