from dataclasses import FrozenInstanceError, fields, replace
import math
from typing import get_type_hints

import numpy as np
import pytest

from roadrunner_mapping.geometry.elevation_map_lifecycle import (
    ElevationMapLifecycleConfig,
    HeightDriftEstimate,
    VisibilityEvidence,
    VisibilityRaySamples,
    VisibilityUpdateResult,
    WallEdgeSelection,
    apply_visibility_and_upper_bounds,
    estimate_height_drift,
    overhead_keep_mask,
    raycast_grid_samples,
    select_visibility_endpoints,
    visibility_evidence,
    wall_edge_keep_mask,
)


EXPECTED_DEFAULTS = {
    "enabled": False,
    "enable_visibility_cleanup": True,
    "enable_upper_bound": True,
    "enable_height_drift_compensation": True,
    "enable_overhead_filter": True,
    "enable_wall_edge_sharpening": True,
    "enable_overlap_clearance": True,
    "enable_continuous_validity": True,
    "overhead_slope": 0.30,
    "overhead_distance_offset_m": 1.0,
    "overhead_height_intercept_m": 0.20,
    "overhead_height_cap_m": 2.0,
    "drift_validity_min": 0.8,
    "drift_variance_max": 0.01,
    "drift_min_frames": 3,
    "drift_slope_max_deg": 8.0,
    "drift_residual_inlier_m": 0.15,
    "drift_min_cells": 30,
    "drift_alpha": 0.2,
    "drift_max_correction_per_frame_m": 0.05,
    "wall_point_count_threshold": 20,
    "wall_min_vertical_span_m": 0.30,
    "wall_lower_point_tolerance_m": 0.05,
    "max_visibility_rays_per_frame": 4096,
    "visibility_max_ray_length_m": 40.0,
    "visibility_endpoint_margin_m": 0.30,
    "visibility_sigma_factor": 1.0,
    "visibility_height_margin_m": 0.01,
    "visibility_min_age_sec": 0.5,
    "visibility_cleanup_step": 0.2,
    "visibility_cleanup_cos_threshold": 0.5,
    "elevation_validity_threshold": 0.5,
    "upper_bound_max_age_sec": 10.0,
    "overlap_clear_radius_m": 2.0,
    "overlap_clear_range_z_m": 2.0,
}

BOOL_FIELDS = tuple(
    name for name, value in EXPECTED_DEFAULTS.items() if isinstance(value, bool)
)
INTEGER_FIELDS = (
    "drift_min_frames",
    "drift_min_cells",
    "wall_point_count_threshold",
    "max_visibility_rays_per_frame",
)
REAL_FIELDS = tuple(
    name
    for name, value in EXPECTED_DEFAULTS.items()
    if isinstance(value, float)
)


class _TrackedReduction:
    def __init__(self, value, backend):
        self.value = bool(value)
        self.backend = backend

    def item(self):
        self.backend.host_sync_count += 1
        return self.value


class _TrackedFlags:
    def __init__(self, values, backend):
        self.values = values
        self.backend = backend

    def tolist(self):
        self.backend.host_sync_count += 1
        return list(self.values)


class _TrackingNumpyBackend:
    ndarray = np.ndarray

    def __init__(self):
        self.host_sync_count = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def any(self, condition):
        return _TrackedReduction(np.any(condition), self)

    def stack(self, values):
        return _TrackedFlags([value.value for value in values], self)


def test_config_defaults_types_and_frozen_contract():
    cfg = ElevationMapLifecycleConfig()

    actual = {field.name: getattr(cfg, field.name) for field in fields(cfg)}
    assert actual == EXPECTED_DEFAULTS
    assert get_type_hints(ElevationMapLifecycleConfig) == {
        name: bool if name in BOOL_FIELDS else int if name in INTEGER_FIELDS else float
        for name in EXPECTED_DEFAULTS
    }
    with pytest.raises(FrozenInstanceError):
        cfg.enabled = True


@pytest.mark.parametrize("name", BOOL_FIELDS)
def test_config_requires_boolean_switches(name):
    with pytest.raises(TypeError):
        replace(ElevationMapLifecycleConfig(), **{name: 1})


@pytest.mark.parametrize("name", REAL_FIELDS + INTEGER_FIELDS)
def test_config_rejects_bool_as_number(name):
    with pytest.raises(TypeError):
        replace(ElevationMapLifecycleConfig(), **{name: True})


@pytest.mark.parametrize("name", INTEGER_FIELDS)
@pytest.mark.parametrize("value", [1.5, np.nan, np.inf])
def test_config_requires_integral_count_types(name, value):
    with pytest.raises(TypeError):
        replace(ElevationMapLifecycleConfig(), **{name: value})


@pytest.mark.parametrize("name", REAL_FIELDS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_config_rejects_nonfinite_real_values(name, value):
    with pytest.raises(ValueError):
        replace(ElevationMapLifecycleConfig(), **{name: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"overhead_slope": -0.1},
        {"overhead_distance_offset_m": -0.1},
        {"overhead_height_intercept_m": 0.0},
        {"overhead_height_cap_m": 0.19},
        {"drift_validity_min": -0.1},
        {"drift_validity_min": 1.1},
        {"drift_variance_max": 0.0},
        {"drift_min_frames": 0},
        {"drift_slope_max_deg": 0.0},
        {"drift_slope_max_deg": 90.0},
        {"drift_residual_inlier_m": 0.0},
        {"drift_min_cells": 0},
        {"drift_alpha": 0.0},
        {"drift_alpha": 1.1},
        {"drift_max_correction_per_frame_m": 0.0},
        {"wall_point_count_threshold": 0},
        {"wall_min_vertical_span_m": 0.0},
        {"wall_lower_point_tolerance_m": -0.1},
        {"max_visibility_rays_per_frame": 0},
        {"visibility_max_ray_length_m": 0.0},
        {"visibility_endpoint_margin_m": -0.1},
        {"visibility_endpoint_margin_m": 40.0},
        {"visibility_sigma_factor": -0.1},
        {"visibility_height_margin_m": -0.1},
        {"visibility_min_age_sec": -0.1},
        {"visibility_cleanup_step": 0.0},
        {"visibility_cleanup_step": 1.1},
        {"visibility_cleanup_cos_threshold": -0.1},
        {"visibility_cleanup_cos_threshold": 1.1},
        {"elevation_validity_threshold": 0.0},
        {"elevation_validity_threshold": 1.0},
        {"upper_bound_max_age_sec": 0.0},
        {"overlap_clear_radius_m": -0.1},
        {"overlap_clear_range_z_m": 0.0},
    ],
)
def test_config_rejects_values_outside_declared_bounds(changes):
    with pytest.raises(ValueError):
        replace(ElevationMapLifecycleConfig(), **changes)


def test_drift_variance_upper_bound_is_deferred_to_gpu_map_construction():
    cfg = ElevationMapLifecycleConfig(drift_variance_max=1.0)

    assert cfg.drift_variance_max == 1.0


def test_overhead_filter_uses_default_sloped_height_limit():
    relative = np.array(
        [[0.0, 0.0, 0.2], [2.0, 0.0, 0.4], [2.0, 0.0, 1.5]]
    )

    keep = overhead_keep_mask(
        np,
        relative,
        ElevationMapLifecycleConfig(enabled=True),
    )

    np.testing.assert_array_equal(keep, [True, True, False])


def test_overhead_filter_keeps_points_on_intercept_and_cap_boundaries():
    relative = np.array([[0.0, 0.0, 0.2], [8.0, 0.0, 2.0]])

    keep = overhead_keep_mask(
        np,
        relative,
        ElevationMapLifecycleConfig(enabled=True),
    )

    np.testing.assert_array_equal(keep, [True, True])


def test_overhead_filter_uses_exact_cap_at_theoretical_transition():
    cap = 2.0
    relative = np.array(
        [[7.0, 0.0, cap], [7.0, 0.0, np.nextafter(cap, np.inf)]]
    )

    keep = overhead_keep_mask(
        np,
        relative,
        ElevationMapLifecycleConfig(enabled=True),
    )

    np.testing.assert_array_equal(keep, [True, False])


def test_overhead_filter_excludes_point_one_ulp_above_exact_boundary():
    cfg = ElevationMapLifecycleConfig(enabled=True)
    height_limit = cfg.overhead_height_intercept_m
    above_intercept = np.nextafter(height_limit, np.inf)
    relative = np.array(
        [[0.0, 0.0, height_limit], [0.0, 0.0, above_intercept]]
    )

    keep = overhead_keep_mask(np, relative, cfg)

    np.testing.assert_array_equal(keep, [True, False])


def test_overhead_filter_with_zero_slope_uses_constant_intercept():
    cfg = ElevationMapLifecycleConfig(enabled=True, overhead_slope=0.0)
    intercept = cfg.overhead_height_intercept_m
    relative = np.array(
        [
            [0.0, 0.0, intercept],
            [100.0, 0.0, intercept],
            [100.0, 0.0, np.nextafter(intercept, np.inf)],
        ]
    )

    keep = overhead_keep_mask(np, relative, cfg)

    np.testing.assert_array_equal(keep, [True, True, False])


@pytest.mark.parametrize(
    "cfg",
    [
        ElevationMapLifecycleConfig(enabled=False),
        ElevationMapLifecycleConfig(enabled=True, enable_overhead_filter=False),
    ],
)
def test_overhead_filter_is_bypassed_when_disabled(cfg):
    relative = np.array(
        [[0.0, 0.0, 100.0], [1.0, 1.0, np.finfo(float).max]]
    )

    keep = overhead_keep_mask(np, relative, cfg)

    np.testing.assert_array_equal(keep, [True, True])


@pytest.mark.parametrize(
    "relative",
    [
        np.array([0.0, 0.0, 0.0]),
        np.empty((2, 2)),
        np.empty((2, 4)),
    ],
)
def test_overhead_filter_rejects_invalid_shapes(relative):
    with pytest.raises(ValueError, match=r"\[N,3\]"):
        overhead_keep_mask(np, relative, ElevationMapLifecycleConfig())


@pytest.mark.parametrize(
    "relative",
    [
        np.array([[True, False, True]]),
        np.array([[1.0 + 0.0j, 0.0, 0.0]]),
        np.array([["1", "2", "3"]]),
    ],
)
def test_overhead_filter_rejects_non_real_numeric_types(relative):
    with pytest.raises(TypeError, match="real numeric dtype"):
        overhead_keep_mask(np, relative, ElevationMapLifecycleConfig())


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_overhead_filter_rejects_nonfinite_points(value):
    relative = np.array([[0.0, value, 0.0]])

    with pytest.raises(ValueError, match="finite"):
        overhead_keep_mask(np, relative, ElevationMapLifecycleConfig())


def test_overhead_filter_rejects_invalid_config_type():
    with pytest.raises(TypeError, match="ElevationMapLifecycleConfig"):
        overhead_keep_mask(np, np.empty((0, 3)), object())


def test_overhead_filter_accepts_empty_numeric_point_array(monkeypatch):
    def fail_finite_reduction(*args, **kwargs):
        pytest.fail("empty input must return before finite reduction")

    monkeypatch.setattr(np, "any", fail_finite_reduction)
    keep = overhead_keep_mask(
        np,
        np.empty((0, 3), dtype=np.int64),
        ElevationMapLifecycleConfig(enabled=True),
    )

    assert keep.shape == (0,)
    assert keep.dtype == np.bool_


def test_overhead_filter_numpy_cupy_parity():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:  # pragma: no cover - depends on host CUDA runtime
        pytest.skip("CUDA is unavailable: {}".format(error))
    if device_count < 1:  # pragma: no cover - depends on host CUDA runtime
        pytest.skip("CUDA is unavailable: no CUDA device")

    point_sets = (
        np.array(
            [
                [0.0, 0.0, 0.2],
                [7.0, 0.0, 2.0],
                [7.0, 0.0, np.nextafter(2.0, np.inf)],
            ]
        ),
        np.empty((0, 3), dtype=np.float64),
    )
    cfg = ElevationMapLifecycleConfig(enabled=True)
    for points in point_sets:
        numpy_keep = overhead_keep_mask(np, points, cfg)
        cupy_keep = overhead_keep_mask(cp, cp.asarray(points), cfg)

        assert isinstance(numpy_keep, np.ndarray)
        assert isinstance(cupy_keep, cp.ndarray)
        assert numpy_keep.dtype == np.bool_
        assert cupy_keep.dtype == cp.bool_
        np.testing.assert_array_equal(numpy_keep, cp.asnumpy(cupy_keep))


def test_height_drift_estimate_is_frozen():
    estimate = HeightDriftEstimate(
        new_bias=0.02,
        correction=0.02,
        sample_count=40,
    )

    with pytest.raises(FrozenInstanceError):
        estimate.new_bias = 0.0


def test_height_drift_estimates_default_correction_from_stable_cells():
    cell_ids = np.arange(40, dtype=np.int64)
    estimate = estimate_height_drift(
        np,
        cell_ids,
        np.full(40, 0.10),
        np.full(40, 0.0025),
        np.zeros(40),
        np.full(40, 0.005),
        np.ones(40),
        np.ones(40, dtype=np.bool_),
        np.full(40, 3, dtype=np.int32),
        np.zeros(40),
        current_bias=0.0,
        cfg=ElevationMapLifecycleConfig(enabled=True),
    )

    assert estimate.sample_count == 40
    assert estimate.correction == pytest.approx(0.01)
    assert estimate.new_bias == pytest.approx(0.01)


def _estimate_stable_height_drift(
    point_count=40,
    cfg=None,
    current_bias=0.25,
    **changes
):
    map_size = 40
    inputs = {
        "cell_ids": np.arange(point_count, dtype=np.int64),
        "heights": np.full(point_count, 0.10),
        "measurement_variances": np.full(point_count, 0.0025),
        "mean": np.zeros(map_size),
        "map_variance": np.full(map_size, 0.005),
        "validity": np.ones(map_size),
        "valid": np.ones(map_size, dtype=np.bool_),
        "accepted_frame_count": np.full(map_size, 3, dtype=np.int32),
        "slope": np.zeros(map_size),
    }
    inputs.update(changes)
    return estimate_height_drift(
        np,
        inputs["cell_ids"],
        inputs["heights"],
        inputs["measurement_variances"],
        inputs["mean"],
        inputs["map_variance"],
        inputs["validity"],
        inputs["valid"],
        inputs["accepted_frame_count"],
        inputs["slope"],
        current_bias=current_bias,
        cfg=cfg or ElevationMapLifecycleConfig(enabled=True),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"valid": np.zeros(40, dtype=np.bool_)},
        {"validity": np.full(40, np.nextafter(0.8, -np.inf))},
        {"map_variance": np.full(40, np.nextafter(0.01, np.inf))},
        {"accepted_frame_count": np.full(40, 2, dtype=np.int32)},
        {
            "slope": np.full(
                40,
                np.nextafter(np.deg2rad(8.0), np.inf),
            )
        },
        {"heights": np.full(40, 0.25 + np.nextafter(0.15, np.inf))},
        {"measurement_variances": np.zeros(40)},
        {"measurement_variances": np.full(40, np.inf)},
    ],
)
def test_height_drift_excludes_points_outside_stable_candidate_rules(changes):
    estimate = _estimate_stable_height_drift(**changes)

    assert estimate == HeightDriftEstimate(
        new_bias=0.25,
        correction=0.0,
        sample_count=0,
    )


def test_height_drift_candidate_thresholds_are_inclusive():
    estimate = _estimate_stable_height_drift(
        current_bias=0.0,
        heights=np.full(40, 0.15),
        map_variance=np.full(40, 0.01),
        validity=np.full(40, 0.8),
        accepted_frame_count=np.full(40, 3, dtype=np.int32),
        slope=np.full(40, np.deg2rad(8.0)),
    )

    assert estimate.sample_count == 40
    assert estimate.correction == pytest.approx(0.01)


def test_height_drift_uses_actual_candidate_count_for_minimum():
    valid = np.ones(40, dtype=np.bool_)
    valid[29] = False
    estimate = _estimate_stable_height_drift(point_count=30, valid=valid)

    assert estimate == HeightDriftEstimate(
        new_bias=0.25,
        correction=0.0,
        sample_count=29,
    )


@pytest.mark.parametrize(
    "cfg",
    [
        ElevationMapLifecycleConfig(enabled=False),
        ElevationMapLifecycleConfig(
            enabled=True,
            enable_height_drift_compensation=False,
        ),
    ],
)
def test_height_drift_is_bypassed_when_disabled(cfg):
    estimate = _estimate_stable_height_drift(cfg=cfg)

    assert estimate == HeightDriftEstimate(
        new_bias=0.25,
        correction=0.0,
        sample_count=0,
    )


def test_height_drift_precision_aggregates_each_cell_once():
    cell_ids = np.repeat(np.arange(30, dtype=np.int64), 2)
    residuals = np.tile([0.10, -0.10], 30)
    measurement_variances = np.tile([0.001, 0.009], 30)
    cfg = replace(
        ElevationMapLifecycleConfig(enabled=True),
        drift_max_correction_per_frame_m=0.20,
    )

    estimate = _estimate_stable_height_drift(
        point_count=60,
        cfg=cfg,
        current_bias=0.0,
        cell_ids=cell_ids,
        heights=residuals,
        measurement_variances=measurement_variances,
    )

    assert estimate.sample_count == 30
    assert estimate.correction == pytest.approx(0.2 * 0.08)
    assert estimate.new_bias == pytest.approx(0.2 * 0.08)


def test_height_drift_precision_aggregation_avoids_reciprocal_overflow():
    smallest = np.nextafter(0.0, 1.0)
    cell_ids = np.repeat(np.arange(30, dtype=np.int64), 2)
    estimate = _estimate_stable_height_drift(
        point_count=60,
        cfg=replace(
            ElevationMapLifecycleConfig(enabled=True),
            drift_max_correction_per_frame_m=0.20,
        ),
        current_bias=0.0,
        cell_ids=cell_ids,
        heights=np.tile([0.10, -0.10], 30),
        measurement_variances=np.tile([smallest, 2.0 * smallest], 30),
    )

    assert estimate.sample_count == 30
    assert estimate.correction == pytest.approx(0.2 * (0.10 / 3.0))
    assert math.isfinite(estimate.new_bias)


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_ids": np.arange(40, dtype=np.int64).reshape(20, 2)},
        {"heights": np.full((20, 2), 0.10)},
        {"measurement_variances": np.full((20, 2), 0.0025)},
        {"heights": np.full(39, 0.10)},
    ],
)
def test_height_drift_requires_equal_one_dimensional_point_arrays(changes):
    with pytest.raises(ValueError, match="one-dimensional|equal shape"):
        _estimate_stable_height_drift(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"cell_ids": np.arange(40, dtype=np.uint64)},
        {"cell_ids": np.zeros(40, dtype=np.bool_)},
        {"heights": np.ones(40, dtype=np.bool_)},
        {"heights": np.ones(40, dtype=np.complex128)},
        {"measurement_variances": np.ones(40, dtype=np.bool_)},
        {"measurement_variances": np.ones(40, dtype=np.complex128)},
    ],
)
def test_height_drift_rejects_invalid_point_dtypes(changes):
    with pytest.raises(TypeError):
        _estimate_stable_height_drift(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"mean": np.zeros((8, 5))},
        {"map_variance": np.full(39, 0.005)},
        {"validity": np.ones(40, dtype=np.int32)},
        {"valid": np.ones(40, dtype=np.int8)},
        {"accepted_frame_count": np.full(40, 3.0)},
        {"slope": np.zeros(40, dtype=np.int32)},
    ],
)
def test_height_drift_requires_strict_flat_map_state_arrays(changes):
    expected_error = TypeError if any(
        name in changes
        for name in ("validity", "valid", "accepted_frame_count", "slope")
    ) else ValueError
    with pytest.raises(expected_error):
        _estimate_stable_height_drift(**changes)


@pytest.mark.parametrize(
    "cell_ids",
    [
        np.concatenate((np.arange(39, dtype=np.int64), [-1])),
        np.concatenate((np.arange(39, dtype=np.int64), [40])),
    ],
)
def test_height_drift_rejects_cell_ids_outside_map(cell_ids):
    with pytest.raises(ValueError, match="outside"):
        _estimate_stable_height_drift(cell_ids=cell_ids)


@pytest.mark.parametrize(
    "changes",
    [
        {"heights": np.full(40, np.nan)},
        {"mean": np.full(40, np.inf)},
        {"map_variance": np.full(40, np.nan)},
        {"validity": np.full(40, np.inf)},
        {"slope": np.full(40, np.nan)},
    ],
)
def test_height_drift_rejects_nonfinite_required_values(changes):
    with pytest.raises(ValueError, match="finite"):
        _estimate_stable_height_drift(**changes)


@pytest.mark.parametrize("current_bias", [True, np.nan, np.inf, -np.inf])
def test_height_drift_requires_finite_real_current_bias(current_bias):
    expected_error = TypeError if isinstance(current_bias, bool) else ValueError
    with pytest.raises(expected_error):
        _estimate_stable_height_drift(current_bias=current_bias)


def test_height_drift_rejects_invalid_config_type():
    with pytest.raises(TypeError, match="ElevationMapLifecycleConfig"):
        _estimate_stable_height_drift(cfg=object())


def test_height_drift_accepts_empty_point_arrays():
    estimate = _estimate_stable_height_drift(
        point_count=0,
        cell_ids=np.empty((0,), dtype=np.int64),
        heights=np.empty((0,), dtype=np.float64),
        measurement_variances=np.empty((0,), dtype=np.float64),
    )

    assert estimate == HeightDriftEstimate(0.25, 0.0, 0)


def test_height_drift_validation_uses_one_host_sync_before_bypass():
    xp = _TrackingNumpyBackend()
    cell_ids = np.arange(40, dtype=np.int64)

    estimate = estimate_height_drift(
        xp,
        cell_ids,
        np.full(40, 0.10),
        np.full(40, 0.0025),
        np.zeros(40),
        np.full(40, 0.005),
        np.ones(40),
        np.ones(40, dtype=np.bool_),
        np.full(40, 3, dtype=np.int32),
        np.zeros(40),
        current_bias=0.0,
        cfg=ElevationMapLifecycleConfig(enabled=False),
    )

    assert estimate == HeightDriftEstimate(0.0, 0.0, 0)
    assert xp.host_sync_count == 1


def _select_wall_edges(cell_ids, heights, mean=None, valid=None, cfg=None):
    map_mean = np.zeros(8) if mean is None else mean
    map_valid = np.ones(8, dtype=np.bool_) if valid is None else valid
    return wall_edge_keep_mask(
        np,
        cell_ids,
        heights,
        map_mean,
        map_valid,
        cfg or ElevationMapLifecycleConfig(enabled=True),
    )


def test_wall_edge_selection_is_frozen():
    selection = WallEdgeSelection(
        keep_mask=np.ones(1, dtype=np.bool_),
        ignored_mask=np.zeros(1, dtype=np.bool_),
        sharpened_cell_ids=np.empty((0,), dtype=np.int64),
    )

    with pytest.raises(FrozenInstanceError):
        selection.keep_mask = np.zeros(1, dtype=np.bool_)


def test_wall_edge_sharpens_count_span_and_tolerance_boundaries():
    heights = np.array([-0.10, -0.05] + [0.20] * 18)
    selection = _select_wall_edges(
        np.full(20, 3, dtype=np.int64),
        heights,
    )

    np.testing.assert_array_equal(
        selection.keep_mask,
        [False, True] + [True] * 18,
    )
    np.testing.assert_array_equal(selection.ignored_mask, ~selection.keep_mask)
    np.testing.assert_array_equal(selection.sharpened_cell_ids, [3])


@pytest.mark.parametrize(
    ("cell_ids", "heights", "valid"),
    [
        (
            np.full(19, 2, dtype=np.int64),
            np.linspace(-0.10, 0.20, 19),
            np.ones(8, dtype=np.bool_),
        ),
        (
            np.full(20, 2, dtype=np.int64),
            np.linspace(-0.10, 0.19, 20),
            np.ones(8, dtype=np.bool_),
        ),
        (
            np.full(20, 2, dtype=np.int64),
            np.linspace(-0.10, 0.20, 20),
            np.zeros(8, dtype=np.bool_),
        ),
    ],
)
def test_wall_edge_keeps_sparse_short_span_and_unknown_cells(
    cell_ids,
    heights,
    valid,
):
    selection = _select_wall_edges(cell_ids, heights, valid=valid)

    np.testing.assert_array_equal(
        selection.keep_mask,
        np.ones(heights.shape, dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        selection.ignored_mask,
        np.zeros(heights.shape, dtype=np.bool_),
    )
    assert selection.sharpened_cell_ids.size == 0


@pytest.mark.parametrize(
    "cfg",
    [
        ElevationMapLifecycleConfig(enabled=False),
        ElevationMapLifecycleConfig(
            enabled=True,
            enable_wall_edge_sharpening=False,
        ),
    ],
)
def test_wall_edge_is_bypassed_when_disabled(cfg):
    heights = np.array([-0.10] + [0.20] * 19)
    selection = _select_wall_edges(
        np.full(20, 2, dtype=np.int64),
        heights,
        cfg=cfg,
    )

    np.testing.assert_array_equal(
        selection.keep_mask,
        np.ones(20, dtype=np.bool_),
    )
    assert not bool(np.any(selection.ignored_mask))
    assert selection.sharpened_cell_ids.size == 0


def test_wall_edge_returns_unique_sorted_sharpened_cells():
    cell_ids = np.repeat(np.array([5, 1, 5], dtype=np.int64), 20)
    heights = np.tile(np.linspace(-0.10, 0.20, 20), 3)
    selection = _select_wall_edges(cell_ids, heights)

    np.testing.assert_array_equal(selection.sharpened_cell_ids, [1, 5])


@pytest.mark.parametrize(
    ("cell_ids", "heights"),
    [
        (np.zeros((10, 2), dtype=np.int64), np.zeros(20)),
        (np.zeros(20, dtype=np.int64), np.zeros((10, 2))),
        (np.zeros(20, dtype=np.int64), np.zeros(19)),
    ],
)
def test_wall_edge_requires_equal_one_dimensional_point_arrays(
    cell_ids,
    heights,
):
    with pytest.raises(ValueError, match="one-dimensional|equal shape"):
        _select_wall_edges(cell_ids, heights)


@pytest.mark.parametrize(
    ("cell_ids", "heights"),
    [
        (np.zeros(20, dtype=np.uint64), np.zeros(20)),
        (np.zeros(20, dtype=np.bool_), np.zeros(20)),
        (np.zeros(20, dtype=np.int64), np.zeros(20, dtype=np.bool_)),
        (np.zeros(20, dtype=np.int64), np.zeros(20, dtype=np.complex128)),
    ],
)
def test_wall_edge_rejects_invalid_point_dtypes(cell_ids, heights):
    with pytest.raises(TypeError):
        _select_wall_edges(cell_ids, heights)


@pytest.mark.parametrize(
    ("mean", "valid", "expected_error"),
    [
        (np.zeros((2, 4)), np.ones(8, dtype=np.bool_), ValueError),
        (np.zeros(7), np.ones(8, dtype=np.bool_), ValueError),
        (np.zeros(8, dtype=np.int32), np.ones(8, dtype=np.bool_), TypeError),
        (np.zeros(8), np.ones(8, dtype=np.int8), TypeError),
    ],
)
def test_wall_edge_requires_strict_flat_map_state(mean, valid, expected_error):
    with pytest.raises(expected_error):
        _select_wall_edges(
            np.zeros(20, dtype=np.int64),
            np.zeros(20),
            mean=mean,
            valid=valid,
        )


@pytest.mark.parametrize("cell_id", [-1, 8])
def test_wall_edge_rejects_cell_ids_outside_map(cell_id):
    with pytest.raises(ValueError, match="outside"):
        _select_wall_edges(
            np.full(20, cell_id, dtype=np.int64),
            np.zeros(20),
        )


@pytest.mark.parametrize(
    ("heights", "mean"),
    [
        (np.full(20, np.nan), np.zeros(8)),
        (np.zeros(20), np.full(8, np.inf)),
    ],
)
def test_wall_edge_rejects_nonfinite_values(heights, mean):
    with pytest.raises(ValueError, match="finite"):
        _select_wall_edges(
            np.zeros(20, dtype=np.int64),
            heights,
            mean=mean,
        )


def test_wall_edge_rejects_invalid_config_type():
    with pytest.raises(TypeError, match="ElevationMapLifecycleConfig"):
        _select_wall_edges(
            np.zeros(20, dtype=np.int64),
            np.zeros(20),
            cfg=object(),
        )


def test_wall_edge_validates_before_disabled_bypass():
    with pytest.raises(ValueError, match="one-dimensional"):
        _select_wall_edges(
            np.zeros((10, 2), dtype=np.int64),
            np.zeros(20),
            cfg=ElevationMapLifecycleConfig(enabled=False),
        )


def test_wall_edge_accepts_empty_point_arrays():
    selection = _select_wall_edges(
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.float64),
    )

    assert selection.keep_mask.shape == (0,)
    assert selection.ignored_mask.shape == (0,)
    assert selection.sharpened_cell_ids.shape == (0,)
    assert selection.keep_mask.dtype == np.bool_
    assert selection.ignored_mask.dtype == np.bool_
    assert selection.sharpened_cell_ids.dtype == np.int64


def test_wall_edge_validation_uses_one_host_sync_before_bypass():
    xp = _TrackingNumpyBackend()
    selection = wall_edge_keep_mask(
        xp,
        np.zeros(20, dtype=np.int64),
        np.zeros(20),
        np.zeros(8),
        np.ones(8, dtype=np.bool_),
        ElevationMapLifecycleConfig(enabled=False),
    )

    assert selection.keep_mask.shape == (20,)
    assert xp.host_sync_count == 1


def test_height_drift_and_wall_edge_numpy_cupy_parity():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:  # pragma: no cover - depends on host CUDA runtime
        pytest.skip("CUDA is unavailable: {}".format(error))
    if device_count < 1:  # pragma: no cover - depends on host CUDA runtime
        pytest.skip("CUDA is unavailable: no CUDA device")

    cell_ids = np.arange(40, dtype=np.int64)
    heights = np.full(40, 0.10)
    measurement_variances = np.full(40, 0.0025)
    mean = np.zeros(40)
    map_variance = np.full(40, 0.005)
    validity = np.ones(40)
    valid = np.ones(40, dtype=np.bool_)
    accepted_frame_count = np.full(40, 3, dtype=np.int32)
    slope = np.zeros(40)
    cfg = ElevationMapLifecycleConfig(enabled=True)
    numpy_estimate = estimate_height_drift(
        np,
        cell_ids,
        heights,
        measurement_variances,
        mean,
        map_variance,
        validity,
        valid,
        accepted_frame_count,
        slope,
        0.0,
        cfg,
    )
    cupy_estimate = estimate_height_drift(
        cp,
        cp.asarray(cell_ids),
        cp.asarray(heights),
        cp.asarray(measurement_variances),
        cp.asarray(mean),
        cp.asarray(map_variance),
        cp.asarray(validity),
        cp.asarray(valid),
        cp.asarray(accepted_frame_count),
        cp.asarray(slope),
        0.0,
        cfg,
    )

    assert cupy_estimate.sample_count == numpy_estimate.sample_count
    assert cupy_estimate.correction == pytest.approx(numpy_estimate.correction)
    assert cupy_estimate.new_bias == pytest.approx(numpy_estimate.new_bias)

    wall_cell_ids = np.full(20, 3, dtype=np.int64)
    wall_heights = np.array([-0.10, -0.05] + [0.20] * 18)
    numpy_selection = wall_edge_keep_mask(
        np,
        wall_cell_ids,
        wall_heights,
        np.zeros(8),
        np.ones(8, dtype=np.bool_),
        cfg,
    )
    cupy_selection = wall_edge_keep_mask(
        cp,
        cp.asarray(wall_cell_ids),
        cp.asarray(wall_heights),
        cp.zeros(8, dtype=cp.float64),
        cp.ones(8, dtype=cp.bool_),
        cfg,
    )

    assert isinstance(cupy_selection.keep_mask, cp.ndarray)
    assert isinstance(cupy_selection.ignored_mask, cp.ndarray)
    assert isinstance(cupy_selection.sharpened_cell_ids, cp.ndarray)
    np.testing.assert_array_equal(
        numpy_selection.keep_mask,
        cp.asnumpy(cupy_selection.keep_mask),
    )
    np.testing.assert_array_equal(
        numpy_selection.ignored_mask,
        cp.asnumpy(cupy_selection.ignored_mask),
    )
    np.testing.assert_array_equal(
        numpy_selection.sharpened_cell_ids,
        cp.asnumpy(cupy_selection.sharpened_cell_ids),
    )


def test_visibility_ray_samples_is_frozen():
    samples = VisibilityRaySamples(
        cell_ids=np.array([1], dtype=np.int64),
        heights=np.array([0.5]),
        directions=np.array([[1.0, 0.0, 0.0]]),
        ray_indices=np.array([0], dtype=np.int64),
        distances=np.array([0.25]),
    )

    with pytest.raises(FrozenInstanceError):
        samples.cell_ids = np.array([2], dtype=np.int64)


def test_visibility_evidence_is_frozen():
    evidence = VisibilityEvidence(
        upper_bound_cell_ids=np.array([1], dtype=np.int64),
        upper_bound_candidates=np.array([0.5]),
        penetrated_cell_ids=np.array([2], dtype=np.int64),
    )

    with pytest.raises(FrozenInstanceError):
        evidence.penetrated_cell_ids = np.array([3], dtype=np.int64)


def test_visibility_endpoint_limit_is_exact_deterministic_and_ordered():
    points = np.column_stack(
        (
            np.arange(5000, dtype=np.float64),
            np.zeros(5000),
            np.ones(5000),
        )
    )

    first = select_visibility_endpoints(np, points, 4096)
    second = select_visibility_endpoints(np, points, 4096)

    assert first.shape == (4096, 3)
    np.testing.assert_array_equal(first, second)
    assert first[0, 0] == 0.0
    assert first[-1, 0] == 4999.0
    assert bool(np.all(np.diff(first[:, 0]) > 0.0))


@pytest.mark.parametrize("size", [0, 3])
def test_visibility_endpoint_limit_keeps_small_inputs_in_original_order(size):
    points = np.arange(size * 3, dtype=np.int64).reshape(size, 3)

    selected = select_visibility_endpoints(np, points, 4)

    np.testing.assert_array_equal(selected, points)
    assert selected.shape == (size, 3)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros(3),
        np.zeros((2, 2)),
        np.zeros((2, 4)),
    ],
)
def test_visibility_endpoint_limit_requires_n_by_three_points(points):
    with pytest.raises(ValueError, match=r"\[N,3\]"):
        select_visibility_endpoints(np, points, 2)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((2, 3), dtype=np.bool_),
        np.zeros((2, 3), dtype=np.complex128),
        np.full((2, 3), "1"),
    ],
)
def test_visibility_endpoint_limit_requires_real_numeric_points(points):
    with pytest.raises(TypeError, match="real numeric dtype"):
        select_visibility_endpoints(np, points, 2)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_visibility_endpoint_limit_rejects_nonfinite_points(value):
    points = np.array([[0.0, value, 0.0]])

    with pytest.raises(ValueError, match="finite"):
        select_visibility_endpoints(np, points, 1)


@pytest.mark.parametrize(
    ("max_rays", "expected_error"),
    [
        (True, TypeError),
        (1.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_visibility_endpoint_limit_requires_positive_integer_limit(
    max_rays,
    expected_error,
):
    with pytest.raises(expected_error, match="positive integer"):
        select_visibility_endpoints(np, np.empty((0, 3)), max_rays)


def test_raycast_samples_in_ray_distance_order_with_margin_and_deduplication():
    callback_shapes = []

    def world_xy_to_flat_ids(x, y):
        callback_shapes.append((x.shape, y.shape))
        return (
            np.floor(x / 0.2).astype(np.int64),
            (x >= 0.0) & (x < 10.0) & np.isfinite(y),
        )

    samples = raycast_grid_samples(
        np,
        np.array([0.0, 0.0, 1.0]),
        np.array([[2.0, 0.0, 0.0]]),
        resolution=0.2,
        max_length=3.0,
        endpoint_margin=0.3,
        world_xy_to_flat_ids=world_xy_to_flat_ids,
    )

    step = 0.2 / math.sqrt(2.0)
    expected_sample_indices = np.array([0, 2, 4, 5, 7, 8, 10, 12, 13])
    expected_distances = expected_sample_indices * step
    expected_direction = np.array([2.0, 0.0, -1.0]) / math.sqrt(5.0)
    expected_shape = (1, math.ceil(3.0 / step))
    assert callback_shapes == [(expected_shape, expected_shape)]
    np.testing.assert_array_equal(samples.cell_ids, np.arange(9))
    np.testing.assert_allclose(samples.distances, expected_distances)
    np.testing.assert_allclose(
        samples.heights,
        1.0 + expected_distances * expected_direction[2],
    )
    np.testing.assert_allclose(
        samples.directions,
        np.repeat(expected_direction[None, :], 9, axis=0),
    )
    np.testing.assert_array_equal(samples.ray_indices, np.zeros(9, dtype=np.int64))
    assert bool(np.all(np.diff(samples.distances) > 0.0))
    assert samples.distances[-1] < math.sqrt(5.0) - 0.3
    assert samples.distances[-1] + step >= math.sqrt(5.0) - 0.3
    assert samples.cell_ids.dtype.kind == "i"
    assert samples.ray_indices.dtype.kind == "i"
    assert samples.directions.shape == (samples.cell_ids.size, 3)
    assert all(
        values.shape[0] == samples.cell_ids.size
        for values in (
            samples.heights,
            samples.directions,
            samples.ray_indices,
            samples.distances,
        )
    )


def test_raycast_truncates_long_ray_without_real_endpoint_margin():
    def world_xy_to_flat_ids(x, y):
        return np.floor(x / 0.1).astype(np.int64), np.ones(x.shape, dtype=np.bool_)

    samples = raycast_grid_samples(
        np,
        np.zeros(3),
        np.array([[10.0, 0.0, 0.0]]),
        resolution=0.2,
        max_length=1.0,
        endpoint_margin=0.3,
        world_xy_to_flat_ids=world_xy_to_flat_ids,
    )

    assert samples.distances[-1] > 1.0 - 0.3
    assert samples.distances[-1] < 1.0
    np.testing.assert_allclose(samples.directions, [[1.0, 0.0, 0.0]] * 8)


def test_raycast_removes_zero_length_rays_and_keeps_shared_cells_per_ray():
    def one_cell(x, y):
        return np.full(x.shape, 7, dtype=np.int64), np.ones(x.shape, dtype=np.bool_)

    samples = raycast_grid_samples(
        np,
        np.zeros(3),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        resolution=0.2,
        max_length=2.0,
        endpoint_margin=0.3,
        world_xy_to_flat_ids=one_cell,
    )

    np.testing.assert_array_equal(samples.cell_ids, [7, 7])
    np.testing.assert_array_equal(samples.ray_indices, [1, 2])
    np.testing.assert_allclose(samples.directions, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(samples.distances, [0.0, 0.0])


def test_raycast_removes_samples_outside_map():
    def all_outside(x, y):
        return np.zeros(x.shape, dtype=np.int64), np.zeros(x.shape, dtype=np.bool_)

    samples = raycast_grid_samples(
        np,
        np.zeros(3),
        np.array([[1.0, 0.0, 0.0]]),
        resolution=0.2,
        max_length=2.0,
        endpoint_margin=0.3,
        world_xy_to_flat_ids=all_outside,
    )

    assert samples.cell_ids.shape == (0,)
    assert samples.heights.shape == (0,)
    assert samples.directions.shape == (0, 3)
    assert samples.ray_indices.shape == (0,)
    assert samples.distances.shape == (0,)


def test_raycast_excluded_samples_break_consecutive_cell_runs():
    def outside_gap(x, y):
        cell_ids = np.zeros(x.shape, dtype=np.int64)
        cell_ids[:, :3] = [7, 9, 7]
        in_map = np.zeros(x.shape, dtype=np.bool_)
        in_map[:, :3] = [True, False, True]
        return cell_ids, in_map

    samples = raycast_grid_samples(
        np,
        np.zeros(3),
        np.array([[1.0, 0.0, 0.0]]),
        resolution=0.2,
        max_length=2.0,
        endpoint_margin=0.3,
        world_xy_to_flat_ids=outside_gap,
    )

    step = 0.2 / math.sqrt(2.0)
    np.testing.assert_array_equal(samples.cell_ids, [7, 7])
    np.testing.assert_array_equal(samples.ray_indices, [0, 0])
    np.testing.assert_allclose(samples.distances, [0.0, 2.0 * step])


def test_raycast_accepts_empty_endpoints_without_calling_grid_callback():
    def unexpected_callback(*args):
        pytest.fail("empty endpoints must not call the grid callback")

    samples = raycast_grid_samples(
        np,
        np.zeros(3),
        np.empty((0, 3), dtype=np.float64),
        resolution=0.2,
        max_length=2.0,
        endpoint_margin=0.3,
        world_xy_to_flat_ids=unexpected_callback,
    )

    assert samples.cell_ids.shape == (0,)
    assert samples.cell_ids.dtype.kind == "i"
    assert samples.heights.shape == (0,)
    assert samples.directions.shape == (0, 3)
    assert samples.ray_indices.dtype.kind == "i"


@pytest.mark.parametrize(
    ("sensor_origin", "endpoints"),
    [
        (np.zeros((1, 3)), np.zeros((1, 3))),
        (np.zeros(2), np.zeros((1, 3))),
        (np.zeros(4), np.zeros((1, 3))),
        (np.zeros(3), np.zeros(3)),
        (np.zeros(3), np.zeros((1, 2))),
        (np.zeros(3), np.zeros((1, 4))),
    ],
)
def test_raycast_requires_origin_and_endpoint_shapes(sensor_origin, endpoints):
    with pytest.raises(ValueError, match=r"\[3\]|\[M,3\]"):
        raycast_grid_samples(
            np,
            sensor_origin,
            endpoints,
            resolution=0.2,
            max_length=2.0,
            endpoint_margin=0.3,
            world_xy_to_flat_ids=lambda x, y: (x.astype(np.int64), x == x),
        )


@pytest.mark.parametrize(
    ("sensor_origin", "endpoints"),
    [
        (np.zeros(3, dtype=np.bool_), np.zeros((1, 3))),
        (np.zeros(3, dtype=np.complex128), np.zeros((1, 3))),
        (np.zeros(3), np.zeros((1, 3), dtype=np.bool_)),
        (np.zeros(3), np.zeros((1, 3), dtype=np.complex128)),
        (np.zeros(3), np.full((1, 3), "1")),
    ],
)
def test_raycast_requires_real_numeric_coordinates(sensor_origin, endpoints):
    with pytest.raises(TypeError, match="real numeric dtype"):
        raycast_grid_samples(
            np,
            sensor_origin,
            endpoints,
            resolution=0.2,
            max_length=2.0,
            endpoint_margin=0.3,
            world_xy_to_flat_ids=lambda x, y: (x.astype(np.int64), x == x),
        )


@pytest.mark.parametrize("target", ["origin", "endpoints"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_raycast_rejects_nonfinite_coordinates(target, value):
    origin = np.zeros(3)
    endpoints = np.ones((1, 3))
    if target == "origin":
        origin[1] = value
    else:
        endpoints[0, 1] = value

    with pytest.raises(ValueError, match="finite"):
        raycast_grid_samples(
            np,
            origin,
            endpoints,
            resolution=0.2,
            max_length=2.0,
            endpoint_margin=0.3,
            world_xy_to_flat_ids=lambda x, y: (x.astype(np.int64), x == x),
        )


@pytest.mark.parametrize(
    ("changes", "expected_error"),
    [
        ({"resolution": True}, TypeError),
        ({"resolution": 0.0}, ValueError),
        ({"resolution": -0.1}, ValueError),
        ({"resolution": np.nan}, ValueError),
        ({"max_length": "2"}, TypeError),
        ({"max_length": 0.0}, ValueError),
        ({"max_length": np.inf}, ValueError),
        ({"endpoint_margin": False}, TypeError),
        ({"endpoint_margin": -0.1}, ValueError),
        ({"endpoint_margin": np.nan}, ValueError),
        ({"endpoint_margin": 2.0}, ValueError),
    ],
)
def test_raycast_rejects_invalid_sampling_parameters(changes, expected_error):
    parameters = {
        "resolution": 0.2,
        "max_length": 2.0,
        "endpoint_margin": 0.3,
    }
    parameters.update(changes)

    with pytest.raises(expected_error):
        raycast_grid_samples(
            np,
            np.zeros(3),
            np.ones((1, 3)),
            world_xy_to_flat_ids=lambda x, y: (x.astype(np.int64), x == x),
            **parameters,
        )


def test_raycast_requires_callable_grid_callback():
    with pytest.raises(TypeError, match="callable"):
        raycast_grid_samples(
            np,
            np.zeros(3),
            np.ones((1, 3)),
            resolution=0.2,
            max_length=2.0,
            endpoint_margin=0.3,
            world_xy_to_flat_ids=None,
        )


@pytest.mark.parametrize(
    ("callback", "expected_error"),
    [
        (lambda x, y: np.zeros(x.shape, dtype=np.int64), TypeError),
        (
            lambda x, y: (
                np.zeros((x.shape[0],), dtype=np.int64),
                np.ones(x.shape, dtype=np.bool_),
            ),
            ValueError,
        ),
        (
            lambda x, y: (
                np.zeros(x.shape, dtype=np.int64),
                np.ones((x.shape[0],), dtype=np.bool_),
            ),
            ValueError,
        ),
        (
            lambda x, y: (
                np.zeros(x.shape, dtype=np.uint64),
                np.ones(x.shape, dtype=np.bool_),
            ),
            TypeError,
        ),
        (
            lambda x, y: (
                np.zeros(x.shape, dtype=np.int64),
                np.ones(x.shape, dtype=np.int8),
            ),
            TypeError,
        ),
    ],
)
def test_raycast_validates_grid_callback_outputs(callback, expected_error):
    with pytest.raises(expected_error):
        raycast_grid_samples(
            np,
            np.zeros(3),
            np.ones((1, 3)),
            resolution=0.2,
            max_length=2.0,
            endpoint_margin=0.3,
            world_xy_to_flat_ids=callback,
        )


def test_visibility_sampling_numpy_cupy_parity():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:  # pragma: no cover - depends on host CUDA runtime
        pytest.skip("CUDA is unavailable: {}".format(error))
    if device_count < 1:  # pragma: no cover - depends on host CUDA runtime
        pytest.skip("CUDA is unavailable: no CUDA device")

    points = np.column_stack((np.arange(12), np.zeros(12), np.ones(12)))
    numpy_endpoints = select_visibility_endpoints(np, points, 7)
    cupy_endpoints = select_visibility_endpoints(cp, cp.asarray(points), 7)
    np.testing.assert_array_equal(numpy_endpoints, cp.asnumpy(cupy_endpoints))

    origin = np.array([0.0, 0.0, 1.0])
    endpoints = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.5]])

    def numpy_grid(x, y):
        cell_ids = np.floor(x / 0.2).astype(np.int64) + 100 * np.floor(
            y / 0.2
        ).astype(np.int64)
        return cell_ids, (x >= 0.0) & (y >= 0.0)

    def cupy_grid(x, y):
        cell_ids = cp.floor(x / 0.2).astype(cp.int64) + 100 * cp.floor(
            y / 0.2
        ).astype(cp.int64)
        return cell_ids, (x >= 0.0) & (y >= 0.0)

    numpy_samples = raycast_grid_samples(
        np,
        origin,
        endpoints,
        resolution=0.2,
        max_length=3.0,
        endpoint_margin=0.3,
        world_xy_to_flat_ids=numpy_grid,
    )
    cupy_samples = raycast_grid_samples(
        cp,
        cp.asarray(origin),
        cp.asarray(endpoints),
        resolution=0.2,
        max_length=3.0,
        endpoint_margin=0.3,
        world_xy_to_flat_ids=cupy_grid,
    )

    for name in ("cell_ids", "heights", "directions", "ray_indices", "distances"):
        numpy_values = getattr(numpy_samples, name)
        cupy_values = getattr(cupy_samples, name)
        assert isinstance(cupy_values, cp.ndarray)
        np.testing.assert_allclose(numpy_values, cp.asnumpy(cupy_values))


def _visibility_state(*, shape=(3, 3), height=1.0):
    cell_count = shape[0] * shape[1]
    return {
        "mean": np.full(cell_count, height, dtype=np.float64),
        "variance": np.full(cell_count, 0.01, dtype=np.float64),
        "valid": np.ones(cell_count, dtype=np.bool_),
        "last_seen": np.zeros(cell_count, dtype=np.float64),
    }


def _visibility_samples(cell_ids, heights, directions):
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    heights = np.asarray(heights, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    return VisibilityRaySamples(
        cell_ids=cell_ids,
        heights=heights,
        directions=directions,
        ray_indices=np.arange(cell_ids.size, dtype=np.int64),
        distances=np.arange(cell_ids.size, dtype=np.float64),
    )


def _compute_visibility(samples, state, *, cfg=None, shape=(3, 3), timestamp=2.0):
    return visibility_evidence(
        np,
        samples,
        state["mean"],
        state["variance"],
        state["valid"],
        state["last_seen"],
        map_shape=shape,
        resolution=0.2,
        timestamp=timestamp,
        cfg=ElevationMapLifecycleConfig(enabled=True) if cfg is None else cfg,
    )


def test_visibility_evidence_detects_low_ray_through_old_flat_surface():
    state = _visibility_state()
    samples = _visibility_samples([4], [0.0], [[0.0, 0.0, -5.0]])

    evidence = _compute_visibility(samples, state)

    np.testing.assert_array_equal(evidence.penetrated_cell_ids, [4])
    np.testing.assert_array_equal(evidence.upper_bound_cell_ids, [4])
    np.testing.assert_allclose(evidence.upper_bound_candidates, [0.0])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state, samples: state["last_seen"].__setitem__(4, 1.5),
        lambda state, samples: samples.directions.__setitem__(0, [4.0, 0.0, 0.0]),
        lambda state, samples: state["valid"].__setitem__(np.arange(9) != 4, False),
        lambda state, samples: samples.heights.__setitem__(0, 0.89),
    ],
    ids=("minimum-age-equality", "tangent-ray", "too-few-neighbours", "height-equality"),
)
def test_visibility_evidence_rejects_non_penetrating_samples(mutate):
    state = _visibility_state()
    samples = _visibility_samples([4], [0.0], [[0.0, 0.0, -1.0]])
    mutate(state, samples)

    evidence = _compute_visibility(samples, state)

    assert evidence.penetrated_cell_ids.size == 0
    np.testing.assert_array_equal(evidence.upper_bound_cell_ids, [4])


def test_visibility_evidence_supports_center_and_one_sided_height_differences():
    state = _visibility_state(shape=(5, 5), height=0.0)
    center = 12
    # Rows grow toward decreasing world x, so this array slope has world normal
    # proportional to (1,0,1).  The opposite sign is orthogonal and must fail.
    state["mean"] = np.repeat(0.2 * np.arange(5, dtype=np.float64), 5)
    state["variance"].fill(0.0)
    direction = np.array([1.0, 0.0, 1.0])
    samples = _visibility_samples([center], [-1.0], [direction])

    centered = _compute_visibility(samples, state, shape=(5, 5))
    np.testing.assert_array_equal(centered.penetrated_cell_ids, [center])

    # Remove the negative-row neighbour; the positive one-sided derivative remains.
    state["valid"][center - 5] = False
    one_sided = _compute_visibility(samples, state, shape=(5, 5))
    np.testing.assert_array_equal(one_sided.penetrated_cell_ids, [center])


def test_visibility_evidence_uses_unique_cells_and_minimum_upper_bound():
    state = _visibility_state()
    state["valid"][8] = False
    samples = _visibility_samples(
        [4, 8, 4, 8],
        [0.2, 0.7, -0.1, 0.3],
        [[0.0, 0.0, -1.0]] * 4,
    )

    evidence = _compute_visibility(samples, state)

    np.testing.assert_array_equal(evidence.upper_bound_cell_ids, [4, 8])
    np.testing.assert_allclose(evidence.upper_bound_candidates, [-0.1, 0.3])
    np.testing.assert_array_equal(evidence.penetrated_cell_ids, [4])


def test_visibility_evidence_switches_are_independent_and_inputs_are_unchanged():
    state = _visibility_state()
    samples = _visibility_samples([4], [0.0], [[0.0, 0.0, -1.0]])
    snapshots = {name: values.copy() for name, values in state.items()}
    sample_snapshots = {
        name: getattr(samples, name).copy()
        for name in ("cell_ids", "heights", "directions", "ray_indices", "distances")
    }

    cleanup_only = _compute_visibility(
        samples,
        state,
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_upper_bound=False,
        ),
    )
    upper_only = _compute_visibility(
        samples,
        state,
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
        ),
    )
    disabled = _compute_visibility(
        samples,
        state,
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
            enable_upper_bound=False,
        ),
    )
    lifecycle_disabled = _compute_visibility(
        samples,
        state,
        cfg=ElevationMapLifecycleConfig(enabled=False),
    )

    assert cleanup_only.upper_bound_cell_ids.size == 0
    np.testing.assert_array_equal(cleanup_only.penetrated_cell_ids, [4])
    np.testing.assert_array_equal(upper_only.upper_bound_cell_ids, [4])
    assert upper_only.penetrated_cell_ids.size == 0
    assert disabled.upper_bound_cell_ids.size == 0
    assert disabled.penetrated_cell_ids.size == 0
    assert lifecycle_disabled.upper_bound_cell_ids.size == 0
    assert lifecycle_disabled.penetrated_cell_ids.size == 0
    for name, expected in snapshots.items():
        np.testing.assert_array_equal(state[name], expected)
    for name, expected in sample_snapshots.items():
        np.testing.assert_array_equal(getattr(samples, name), expected)


def test_visibility_evidence_accepts_empty_samples():
    state = _visibility_state()
    samples = _visibility_samples([], [], np.empty((0, 3)))

    evidence = _compute_visibility(samples, state)

    assert evidence.upper_bound_cell_ids.shape == (0,)
    assert evidence.upper_bound_candidates.shape == (0,)
    assert evidence.penetrated_cell_ids.shape == (0,)


@pytest.mark.parametrize(
    ("change", "expected_error"),
    [
        (lambda state, samples: samples.cell_ids.__setitem__(0, 9), ValueError),
        (lambda state, samples: state["variance"].__setitem__(4, -0.1), ValueError),
        (lambda state, samples: state["last_seen"].__setitem__(4, np.nan), ValueError),
        (lambda state, samples: state["last_seen"].__setitem__(4, np.inf), ValueError),
        (lambda state, samples: samples.ray_indices.__setitem__(0, -1), ValueError),
    ],
)
def test_visibility_evidence_rejects_invalid_numeric_state(change, expected_error):
    state = _visibility_state()
    samples = _visibility_samples([4], [0.0], [[0.0, 0.0, -1.0]])
    change(state, samples)

    with pytest.raises(expected_error):
        _compute_visibility(samples, state)


def test_visibility_evidence_allows_negative_infinite_last_seen_sentinel():
    state = _visibility_state()
    state["last_seen"][4] = -np.inf
    samples = _visibility_samples([4], [0.0], [[0.0, 0.0, -1.0]])

    evidence = _compute_visibility(samples, state)

    np.testing.assert_array_equal(evidence.penetrated_cell_ids, [4])


@pytest.mark.parametrize("integer_dtype", [np.int8, np.int16, np.int32, np.int64])
def test_visibility_evidence_normalizes_signed_cell_id_width(integer_dtype):
    shape = (2, 200)
    state = _visibility_state(shape=shape)
    samples = replace(
        _visibility_samples([1], [0.25], [[0.0, 0.0, -1.0]]),
        cell_ids=np.array([1], dtype=integer_dtype),
    )

    evidence = _compute_visibility(
        samples,
        state,
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
        ),
        shape=shape,
    )

    np.testing.assert_array_equal(evidence.upper_bound_cell_ids, [1])
    assert evidence.upper_bound_cell_ids.dtype == np.int64


def test_visibility_evidence_rejects_nonfinite_ray_direction():
    state = _visibility_state()
    samples = _visibility_samples([4], [0.0], [[np.nan, 0.0, -1.0]])

    with pytest.raises(ValueError, match="visibility samples"):
        _compute_visibility(samples, state)


@pytest.mark.parametrize(
    "cfg",
    [
        ElevationMapLifecycleConfig(enabled=False),
        ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
            enable_upper_bound=False,
        ),
    ],
)
def test_visibility_evidence_disabled_path_does_not_scan_device_arrays(cfg):
    backend = _TrackingNumpyBackend()
    evidence = visibility_evidence(
        backend,
        object(),
        object(),
        object(),
        object(),
        object(),
        map_shape=(3, 3),
        resolution=0.2,
        timestamp=2.0,
        cfg=cfg,
    )

    assert backend.host_sync_count == 0
    assert evidence.upper_bound_cell_ids.size == 0
    assert evidence.penetrated_cell_ids.size == 0


def test_visibility_evidence_upper_only_does_not_read_elevation_state():
    samples = _visibility_samples([4, 4], [0.3, 0.1], [[0, 0, -1]] * 2)

    evidence = visibility_evidence(
        np,
        samples,
        object(),
        object(),
        object(),
        object(),
        map_shape=(3, 3),
        resolution=0.2,
        timestamp=2.0,
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
        ),
    )

    np.testing.assert_array_equal(evidence.upper_bound_cell_ids, [4])
    np.testing.assert_allclose(evidence.upper_bound_candidates, [0.1])


def test_visibility_evidence_numpy_cupy_parity():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:  # pragma: no cover - depends on CUDA runtime
        pytest.skip("CUDA is unavailable: {}".format(error))
    if device_count < 1:  # pragma: no cover
        pytest.skip("CUDA is unavailable: no CUDA device")

    state = _visibility_state()
    samples = _visibility_samples(
        [4, 8, 4],
        [0.1, 0.4, -0.2],
        [[0.0, 0.0, -2.0], [1.0, 0.0, 0.0], [0.0, 0.0, -3.0]],
    )
    numpy_result = _compute_visibility(samples, state)
    cupy_result = visibility_evidence(
        cp,
        VisibilityRaySamples(
            **{
                name: cp.asarray(getattr(samples, name))
                for name in (
                    "cell_ids",
                    "heights",
                    "directions",
                    "ray_indices",
                    "distances",
                )
            }
        ),
        cp.asarray(state["mean"]),
        cp.asarray(state["variance"]),
        cp.asarray(state["valid"]),
        cp.asarray(state["last_seen"]),
        map_shape=(3, 3),
        resolution=0.2,
        timestamp=2.0,
        cfg=ElevationMapLifecycleConfig(enabled=True),
    )

    for name in (
        "upper_bound_cell_ids",
        "upper_bound_candidates",
        "penetrated_cell_ids",
    ):
        cupy_values = getattr(cupy_result, name)
        assert isinstance(cupy_values, cp.ndarray)
        np.testing.assert_allclose(getattr(numpy_result, name), cp.asnumpy(cupy_values))


def test_visibility_update_result_is_frozen():
    result = VisibilityUpdateResult(
        validity_updated_cell_ids=np.array([1]),
        cleared_cell_ids=np.array([2]),
        upper_bound_updated_cell_ids=np.array([3]),
        accepted_cell_ids=np.array([4]),
    )

    with pytest.raises(FrozenInstanceError):
        result.cleared_cell_ids = np.array([5])


def _visibility_mutable_state(size=4):
    return {
        "mean": np.ones(size, dtype=np.float64),
        "variance": np.full(size, 0.1, dtype=np.float64),
        "valid": np.ones(size, dtype=np.bool_),
        "validity": np.ones(size, dtype=np.float64),
        "upper": np.zeros(size, dtype=np.float64),
        "upper_valid": np.zeros(size, dtype=np.bool_),
        "upper_time": np.full(size, -np.inf, dtype=np.float64),
    }


def _visibility_evidence_for_update(penetrated=(), upper_ids=(), upper_values=()):
    return VisibilityEvidence(
        upper_bound_cell_ids=np.asarray(upper_ids, dtype=np.int64),
        upper_bound_candidates=np.asarray(upper_values, dtype=np.float64),
        penetrated_cell_ids=np.asarray(penetrated, dtype=np.int64),
    )


def _apply_visibility_update(
    state,
    evidence,
    *,
    accepted=(),
    cfg=None,
    timestamp=5.0,
    clear_layers=(),
):
    return apply_visibility_and_upper_bounds(
        np,
        evidence,
        np.asarray(accepted, dtype=np.int64),
        state["mean"],
        state["variance"],
        state["valid"],
        state["validity"],
        state["upper"],
        state["upper_valid"],
        state["upper_time"],
        timestamp=timestamp,
        max_variance=1.0,
        outlier_variance=0.2,
        cfg=ElevationMapLifecycleConfig(enabled=True) if cfg is None else cfg,
        clear_layers=clear_layers,
    )


def test_visibility_update_reduces_validity_then_clears_on_third_penetration():
    state = _visibility_mutable_state()
    support = np.array([5, 6, 7, 8], dtype=np.int32)
    evidence = _visibility_evidence_for_update(penetrated=[1])

    first = _apply_visibility_update(
        state, evidence, clear_layers=((support, 0),)
    )
    np.testing.assert_allclose(state["validity"][1], 0.8)
    np.testing.assert_allclose(state["variance"][1], 0.3)
    np.testing.assert_array_equal(first.validity_updated_cell_ids, [1])
    assert first.cleared_cell_ids.size == 0
    assert support[1] == 6

    second = _apply_visibility_update(
        state, evidence, clear_layers=((support, 0),)
    )
    np.testing.assert_allclose(state["validity"][1], 0.6)
    np.testing.assert_allclose(state["variance"][1], 0.5)
    assert second.cleared_cell_ids.size == 0

    third = _apply_visibility_update(
        state, evidence, clear_layers=((support, 0),)
    )
    assert not state["valid"][1]
    assert state["validity"][1] == 0.0
    assert state["mean"][1] == 0.0
    assert state["variance"][1] == 1.0
    assert support[1] == 0
    np.testing.assert_array_equal(third.cleared_cell_ids, [1])

    unchanged = {name: values.copy() for name, values in state.items()}
    fourth = _apply_visibility_update(
        state, evidence, clear_layers=((support, 0),)
    )
    assert fourth.validity_updated_cell_ids.size == 0
    for name, expected in unchanged.items():
        np.testing.assert_array_equal(state[name], expected)


def test_visibility_update_without_continuous_validity_clears_immediately():
    state = _visibility_mutable_state()
    result = _apply_visibility_update(
        state,
        _visibility_evidence_for_update(penetrated=[2]),
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_continuous_validity=False,
        ),
    )

    assert not state["valid"][2]
    assert state["validity"][2] == 0.0
    np.testing.assert_array_equal(result.cleared_cell_ids, [2])


def test_visibility_update_writes_upper_bound_after_same_frame_clear():
    state = _visibility_mutable_state()
    state["validity"][1] = 0.6
    evidence = _visibility_evidence_for_update(
        penetrated=[1], upper_ids=[1], upper_values=[0.25]
    )

    result = _apply_visibility_update(state, evidence, timestamp=5.0)

    assert not state["valid"][1]
    assert state["upper_valid"][1]
    assert state["upper"][1] == 0.25
    assert state["upper_time"][1] == 5.0
    np.testing.assert_array_equal(result.cleared_cell_ids, [1])
    np.testing.assert_array_equal(result.upper_bound_updated_cell_ids, [1])


def test_visibility_update_only_tightens_upper_bound_and_timestamp():
    state = _visibility_mutable_state()
    state["valid"][1] = False
    state["validity"][1] = 0.0
    state["upper"][1] = 0.3
    state["upper_valid"][1] = True
    state["upper_time"][1] = 1.0

    weaker = _apply_visibility_update(
        state,
        _visibility_evidence_for_update(upper_ids=[1], upper_values=[0.4]),
        timestamp=5.0,
    )
    assert weaker.upper_bound_updated_cell_ids.size == 0
    assert state["upper"][1] == 0.3
    assert state["upper_time"][1] == 1.0

    tighter = _apply_visibility_update(
        state,
        _visibility_evidence_for_update(upper_ids=[1], upper_values=[0.2]),
        timestamp=6.0,
    )
    np.testing.assert_array_equal(tighter.upper_bound_updated_cell_ids, [1])
    assert state["upper"][1] == 0.2
    assert state["upper_time"][1] == 6.0

    equal = _apply_visibility_update(
        state,
        _visibility_evidence_for_update(upper_ids=[1], upper_values=[0.2]),
        timestamp=7.0,
    )
    assert equal.upper_bound_updated_cell_ids.size == 0
    assert state["upper_time"][1] == 6.0


def test_visibility_update_accepted_observation_wins_and_clears_upper_flag():
    state = _visibility_mutable_state()
    state["mean"][2] = 2.0
    state["validity"][2] = 0.4
    state["upper"][2] = 0.5
    state["upper_valid"][2] = True
    state["upper_time"][2] = 1.0
    evidence = _visibility_evidence_for_update(
        penetrated=[2], upper_ids=[2], upper_values=[0.1]
    )

    result = _apply_visibility_update(state, evidence, accepted=[2], timestamp=5.0)

    assert state["valid"][2]
    assert state["validity"][2] == 1.0
    assert state["mean"][2] == 2.0
    assert state["upper"][2] == 2.0
    assert not state["upper_valid"][2]
    assert state["upper_time"][2] == -np.inf
    assert result.cleared_cell_ids.size == 0
    np.testing.assert_array_equal(result.accepted_cell_ids, [2])


def test_visibility_update_disabled_does_not_read_or_mutate_state():
    result = apply_visibility_and_upper_bounds(
        np,
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        timestamp=5.0,
        max_variance=1.0,
        outlier_variance=0.2,
        cfg=ElevationMapLifecycleConfig(enabled=False),
    )

    assert result.validity_updated_cell_ids.size == 0
    assert result.cleared_cell_ids.size == 0
    assert result.upper_bound_updated_cell_ids.size == 0
    assert result.accepted_cell_ids.size == 0


def test_visibility_update_subfeature_switches_leave_corresponding_state_unchanged():
    state = _visibility_mutable_state()
    state["valid"][2] = False
    state["validity"][2] = 0.0
    evidence = _visibility_evidence_for_update(
        penetrated=[1], upper_ids=[2], upper_values=[0.1]
    )
    before = {name: values.copy() for name, values in state.items()}

    result = _apply_visibility_update(
        state,
        evidence,
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
            enable_upper_bound=False,
        ),
    )

    for name, expected in before.items():
        np.testing.assert_array_equal(state[name], expected)
    assert result.validity_updated_cell_ids.size == 0
    assert result.upper_bound_updated_cell_ids.size == 0


def test_visibility_update_rejects_out_of_range_evidence_before_mutation():
    state = _visibility_mutable_state()
    before = {name: values.copy() for name, values in state.items()}

    with pytest.raises(ValueError, match="out-of-range"):
        _apply_visibility_update(
            state,
            _visibility_evidence_for_update(
                upper_ids=[4], upper_values=[0.1]
            ),
        )

    for name, expected in before.items():
        np.testing.assert_array_equal(state[name], expected)


def test_visibility_update_rejects_read_only_state_before_any_mutation():
    state = _visibility_mutable_state()
    state["variance"].flags.writeable = False
    before = {name: values.copy() for name, values in state.items()}

    with pytest.raises(ValueError, match="writable"):
        _apply_visibility_update(
            state,
            _visibility_evidence_for_update(penetrated=[1]),
        )

    for name, expected in before.items():
        np.testing.assert_array_equal(state[name], expected)


def test_visibility_update_rejects_non_array_mutable_state():
    state = _visibility_mutable_state()
    state["validity"] = state["validity"].tolist()

    with pytest.raises(TypeError, match="backend array"):
        _apply_visibility_update(
            state,
            _visibility_evidence_for_update(penetrated=[1]),
        )


@pytest.mark.parametrize("overflow_source", ["timestamp", "candidate", "variance"])
def test_visibility_update_rejects_values_not_representable_by_target_dtype(
    overflow_source,
):
    state = _visibility_mutable_state()
    for name in ("mean", "variance", "validity", "upper", "upper_time"):
        state[name] = state[name].astype(np.float32)
    state["valid"][1] = False
    state["validity"][1] = 0.0
    evidence = _visibility_evidence_for_update(upper_ids=[1], upper_values=[0.1])
    timestamp = 5.0
    max_variance = 1.0
    if overflow_source == "timestamp":
        timestamp = 1e300
    elif overflow_source == "candidate":
        evidence = _visibility_evidence_for_update(
            upper_ids=[1], upper_values=[1e300]
        )
    else:
        max_variance = 1e300
    before = {name: values.copy() for name, values in state.items()}

    with pytest.raises(ValueError, match="representable"):
        apply_visibility_and_upper_bounds(
            np,
            evidence,
            np.empty((0,), dtype=np.int64),
            state["mean"],
            state["variance"],
            state["valid"],
            state["validity"],
            state["upper"],
            state["upper_valid"],
            state["upper_time"],
            timestamp=timestamp,
            max_variance=max_variance,
            outlier_variance=0.2,
            cfg=ElevationMapLifecycleConfig(enabled=True),
        )

    for name, expected in before.items():
        np.testing.assert_array_equal(state[name], expected)


def test_visibility_update_rejects_accepted_mean_that_overflows_upper_dtype():
    state = _visibility_mutable_state()
    state["mean"][1] = 1e300
    state["upper"] = state["upper"].astype(np.float32)
    before = {name: values.copy() for name, values in state.items()}

    with pytest.raises(ValueError, match="representable"):
        _apply_visibility_update(
            state,
            _visibility_evidence_for_update(),
            accepted=[1],
        )

    for name, expected in before.items():
        np.testing.assert_array_equal(state[name], expected)


def test_visibility_update_cleanup_and_upper_switches_are_independent():
    cleanup_state = _visibility_mutable_state()
    cleanup_state["validity"][1] = 0.6
    cleanup_before_upper = cleanup_state["upper"].copy()
    _apply_visibility_update(
        cleanup_state,
        _visibility_evidence_for_update(
            penetrated=[1], upper_ids=[1], upper_values=[0.2]
        ),
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_upper_bound=False,
        ),
    )
    assert not cleanup_state["valid"][1]
    np.testing.assert_array_equal(cleanup_state["upper"], cleanup_before_upper)

    upper_state = _visibility_mutable_state()
    upper_state["valid"][2] = False
    upper_state["validity"][2] = 0.0
    before_validity = upper_state["validity"].copy()
    _apply_visibility_update(
        upper_state,
        _visibility_evidence_for_update(
            penetrated=[1], upper_ids=[2], upper_values=[0.2]
        ),
        cfg=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
        ),
    )
    np.testing.assert_array_equal(upper_state["validity"], before_validity)
    assert upper_state["upper_valid"][2]


@pytest.mark.parametrize("alias_kind", ["core", "clear"])
def test_visibility_update_rejects_overlapping_mutable_layers(alias_kind):
    state = _visibility_mutable_state()
    clear_layers = ()
    if alias_kind == "core":
        state["variance"] = state["mean"]
    else:
        clear_layers = ((state["valid"], True),)
    before = {name: values.copy() for name, values in state.items()}

    with pytest.raises(ValueError, match="overlap"):
        _apply_visibility_update(
            state,
            _visibility_evidence_for_update(penetrated=[1]),
            clear_layers=clear_layers,
        )

    for name, expected in before.items():
        np.testing.assert_array_equal(state[name], expected)


def test_visibility_update_reports_only_accepted_cells_that_actually_change():
    state = _visibility_mutable_state()
    state["upper"][:] = state["mean"]
    before = {name: values.copy() for name, values in state.items()}

    result = _apply_visibility_update(
        state,
        _visibility_evidence_for_update(),
        accepted=[1],
    )

    for name, expected in before.items():
        np.testing.assert_array_equal(state[name], expected)
    assert result.validity_updated_cell_ids.size == 0
    assert result.upper_bound_updated_cell_ids.size == 0
    np.testing.assert_array_equal(result.accepted_cell_ids, [1])


def test_visibility_update_compares_accepted_upper_after_target_dtype_cast():
    state = _visibility_mutable_state()
    state["mean"][1] = 1.0 - 1e-10
    state["upper"] = state["upper"].astype(np.float32)
    state["upper"][1] = 1.0

    result = _apply_visibility_update(
        state,
        _visibility_evidence_for_update(),
        accepted=[1],
    )

    assert state["upper"][1] == np.float32(1.0)
    assert result.upper_bound_updated_cell_ids.size == 0


def test_visibility_update_omits_rounded_away_validity_change():
    state = _visibility_mutable_state()
    state["validity"] = state["validity"].astype(np.float32)
    before = {name: values.copy() for name, values in state.items()}
    cfg = replace(
        ElevationMapLifecycleConfig(enabled=True),
        visibility_cleanup_step=1e-300,
    )

    result = apply_visibility_and_upper_bounds(
        np,
        _visibility_evidence_for_update(penetrated=[1]),
        np.empty((0,), dtype=np.int64),
        state["mean"],
        state["variance"],
        state["valid"],
        state["validity"],
        state["upper"],
        state["upper_valid"],
        state["upper_time"],
        timestamp=5.0,
        max_variance=1.0,
        outlier_variance=0.0,
        cfg=cfg,
    )

    for name, expected in before.items():
        np.testing.assert_array_equal(state[name], expected)
    assert result.validity_updated_cell_ids.size == 0


def test_visibility_update_reports_variance_only_change():
    state = _visibility_mutable_state()
    state["validity"] = state["validity"].astype(np.float32)
    cfg = replace(
        ElevationMapLifecycleConfig(enabled=True),
        visibility_cleanup_step=1e-300,
    )

    result = apply_visibility_and_upper_bounds(
        np,
        _visibility_evidence_for_update(penetrated=[1]),
        np.empty((0,), dtype=np.int64),
        state["mean"],
        state["variance"],
        state["valid"],
        state["validity"],
        state["upper"],
        state["upper_valid"],
        state["upper_time"],
        timestamp=5.0,
        max_variance=1.0,
        outlier_variance=0.2,
        cfg=cfg,
    )

    assert state["validity"][1] == np.float32(1.0)
    np.testing.assert_allclose(state["variance"][1], 0.3)
    np.testing.assert_array_equal(result.validity_updated_cell_ids, [1])


@pytest.mark.parametrize("underflow_source", ["timestamp", "candidate"])
def test_visibility_update_rejects_nonzero_float32_underflow(underflow_source):
    state = _visibility_mutable_state()
    for name in ("mean", "variance", "validity", "upper", "upper_time"):
        state[name] = state[name].astype(np.float32)
    state["valid"][1] = False
    state["validity"][1] = 0.0
    timestamp = 5.0
    candidate = 0.1
    if underflow_source == "timestamp":
        timestamp = 1e-300
    else:
        candidate = 1e-300

    with pytest.raises(ValueError, match="representable"):
        apply_visibility_and_upper_bounds(
            np,
            _visibility_evidence_for_update(
                upper_ids=[1], upper_values=[candidate]
            ),
            np.empty((0,), dtype=np.int64),
            state["mean"],
            state["variance"],
            state["valid"],
            state["validity"],
            state["upper"],
            state["upper_valid"],
            state["upper_time"],
            timestamp=timestamp,
            max_variance=1.0,
            outlier_variance=0.2,
            cfg=ElevationMapLifecycleConfig(enabled=True),
        )


def test_visibility_update_numpy_cupy_parity():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:  # pragma: no cover
        pytest.skip("CUDA is unavailable: {}".format(error))
    if device_count < 1:  # pragma: no cover
        pytest.skip("CUDA is unavailable: no CUDA device")

    numpy_state = _visibility_mutable_state()
    numpy_state["validity"][1] = 0.6
    evidence = _visibility_evidence_for_update(
        penetrated=[1], upper_ids=[1, 3], upper_values=[0.2, 0.4]
    )
    numpy_result = _apply_visibility_update(numpy_state, evidence, accepted=[3])

    cupy_state = {name: cp.asarray(values.copy()) for name, values in _visibility_mutable_state().items()}
    cupy_state["validity"][1] = 0.6
    cupy_result = apply_visibility_and_upper_bounds(
        cp,
        VisibilityEvidence(
            upper_bound_cell_ids=cp.asarray(evidence.upper_bound_cell_ids),
            upper_bound_candidates=cp.asarray(evidence.upper_bound_candidates),
            penetrated_cell_ids=cp.asarray(evidence.penetrated_cell_ids),
        ),
        cp.asarray([3], dtype=cp.int64),
        cupy_state["mean"],
        cupy_state["variance"],
        cupy_state["valid"],
        cupy_state["validity"],
        cupy_state["upper"],
        cupy_state["upper_valid"],
        cupy_state["upper_time"],
        timestamp=5.0,
        max_variance=1.0,
        outlier_variance=0.2,
        cfg=ElevationMapLifecycleConfig(enabled=True),
    )

    for name, expected in numpy_state.items():
        np.testing.assert_allclose(expected, cp.asnumpy(cupy_state[name]))
    for name in (
        "validity_updated_cell_ids",
        "cleared_cell_ids",
        "upper_bound_updated_cell_ids",
        "accepted_cell_ids",
    ):
        np.testing.assert_array_equal(
            getattr(numpy_result, name), cp.asnumpy(getattr(cupy_result, name))
        )
