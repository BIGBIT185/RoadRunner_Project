import ast
from dataclasses import FrozenInstanceError, fields, replace
import inspect

import numpy as np
import pytest

from roadrunner_mapping.geometry.gaussian_elevation import (
    GaussianElevationConfig,
    aggregate_frame_observations,
    point_height_variance,
    select_initial_clusters,
    update_gaussian_elevation,
)


def test_config_defaults_and_frozen_contract():
    cfg = GaussianElevationConfig()
    expected = {
        "sensor_std_m": 0.02,
        "range_variance_factor": 1e-5,
        "measurement_variance_min": 0.0004,
        "measurement_variance_max": 0.25,
        "pose_vertical_variance": 0.0009,
        "frame_variance_min": 0.0004,
        "posterior_variance_min": 0.0004,
        "single_point_initial_variance": 0.01,
        "multi_point_initial_variance": 0.005,
        "max_variance": 0.25,
        "process_variance_per_sec": 0.0004,
        "outlier_variance_per_frame": 0.01,
        "mahalanobis_threshold": 2.5,
        "max_effective_points_per_frame": 8,
        "initial_cluster_gap_m": 0.08,
        "initial_min_cluster_points": 2,
        "conflict_tau_sec": 2.0,
        "confidence_variance_good": 0.0025,
        "confidence_variance_bad": 0.04,
        "uncertainty_std_good": 0.04,
        "uncertainty_std_bad": 0.12,
        "local_coverage_low": 0.15,
        "local_coverage_high": 0.60,
    }
    assert {field.name: getattr(cfg, field.name) for field in fields(cfg)} == expected
    with pytest.raises(FrozenInstanceError):
        cfg.sensor_std_m = 1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"sensor_std_m": 0.0},
        {"sensor_std_m": np.nan},
        {"range_variance_factor": -1.0},
        {"measurement_variance_min": 0.0},
        {"measurement_variance_min": 0.2, "measurement_variance_max": 0.1},
        {"pose_vertical_variance": 0.0},
        {"posterior_variance_min": 0.01, "frame_variance_min": 0.005},
        {"frame_variance_min": 0.3},
        {"single_point_initial_variance": 0.0001},
        {"multi_point_initial_variance": 0.3},
        {
            "single_point_initial_variance": 0.005,
            "multi_point_initial_variance": 0.01,
        },
        {"process_variance_per_sec": -1.0},
        {"outlier_variance_per_frame": -1.0},
        {"mahalanobis_threshold": 0.0},
        {"initial_cluster_gap_m": 0.0},
        {"conflict_tau_sec": 0.0},
        {"max_effective_points_per_frame": 0},
        {"max_effective_points_per_frame": 1.5},
        {"initial_min_cluster_points": True},
        {"confidence_variance_good": 0.04, "confidence_variance_bad": 0.04},
        {"uncertainty_std_good": 0.12, "uncertainty_std_bad": 0.12},
        {"local_coverage_low": -0.1},
        {"local_coverage_high": 1.1},
        {"local_coverage_low": 0.6, "local_coverage_high": 0.6},
    ],
)
def test_config_rejects_invalid_values(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(GaussianElevationConfig(), **changes)


def test_point_height_variance_grows_with_range_without_pose_term():
    cfg = GaussianElevationConfig()
    ranges = np.array([0.0, 1.0, 20.0, 1000.0])
    result = point_height_variance(np, ranges, cfg)
    expected = np.clip(
        cfg.sensor_std_m**2 + cfg.range_variance_factor * ranges**2,
        cfg.measurement_variance_min,
        cfg.measurement_variance_max,
    )
    np.testing.assert_allclose(result, expected)
    assert result[2] > result[1]
    assert result[-1] == cfg.measurement_variance_max
    assert result[0] != pytest.approx(
        cfg.measurement_variance_min + cfg.pose_vertical_variance
    )


def test_point_height_variance_rejects_nonfinite_and_negative_ranges():
    cfg = GaussianElevationConfig()
    with pytest.raises(ValueError):
        point_height_variance(np, np.array([1.0, np.nan]), cfg)
    with pytest.raises(ValueError):
        point_height_variance(np, np.array([-1.0]), cfg)


def test_heteroscedastic_frame_aggregation_matches_complete_formula():
    cfg = replace(GaussianElevationConfig(), max_effective_points_per_frame=2)
    cell_ids = np.array([3, 3, 3], dtype=np.int64)
    heights = np.array([0.0, 1.0, 2.0])
    variances = np.array([1.0, 0.25, 1.0])

    result = aggregate_frame_observations(
        np,
        cell_ids,
        heights,
        variances,
        np.ones(3, dtype=bool),
        cfg,
    )

    precision = np.array([1.0, 4.0, 1.0])
    expected_mean = np.sum(precision * heights) / np.sum(precision)
    expected_scatter = (
        np.sum(precision * (heights - expected_mean) ** 2) / np.sum(precision)
    )
    effective_precision = np.sum(precision) * (2.0 / 3.0)
    expected_variance = max(
        1.0 / effective_precision
        + expected_scatter / 2.0
        + cfg.pose_vertical_variance,
        cfg.frame_variance_min,
    )
    np.testing.assert_array_equal(result.cell_ids, [3])
    np.testing.assert_array_equal(result.count, [3])
    np.testing.assert_allclose(result.mean, [expected_mean])
    np.testing.assert_allclose(result.scatter, [expected_scatter])
    np.testing.assert_allclose(result.precision, [np.sum(precision)])
    np.testing.assert_allclose(result.variance, [expected_variance])


def test_shared_pose_variance_does_not_disappear_with_point_count():
    cfg = replace(
        GaussianElevationConfig(),
        measurement_variance_min=1e-6,
        frame_variance_min=1e-6,
        posterior_variance_min=1e-6,
        pose_vertical_variance=0.04,
        max_effective_points_per_frame=100,
    )
    count = 100
    result = aggregate_frame_observations(
        np,
        np.zeros(count, dtype=np.int64),
        np.ones(count),
        np.ones(count) * 0.01,
        np.ones(count, dtype=bool),
        cfg,
    )
    assert result.variance[0] == pytest.approx(0.01 / count + 0.04)
    assert result.variance[0] >= cfg.pose_vertical_variance


def test_initial_clusters_choose_largest_contiguous_height_mode():
    cfg = GaussianElevationConfig()
    cell_ids = np.array([4, 2, 4, 2, 2, 4, 2])
    heights = np.array([1.01, 0.0, 1.0, 0.51, 0.01, 1.02, 0.50])
    selected = select_initial_clusters(np, cell_ids, heights, cfg)
    # Cell 2 has two two-point modes; the lower-mean mode wins. Cell 4 keeps all.
    np.testing.assert_array_equal(selected, [True, True, True, False, True, True, False])


def test_initial_cluster_gap_equal_to_threshold_does_not_split():
    cfg = GaussianElevationConfig(initial_cluster_gap_m=0.125)
    selected = select_initial_clusters(
        np,
        np.array([0, 0, 0]),
        np.array([0.0, 0.125, 0.250001]),
        cfg,
    )
    np.testing.assert_array_equal(selected, [True, True, False])


def test_initial_cluster_lexsort_uses_cupy_12_3_key_matrix_contract():
    class CuPy123LexsortContract:
        ndarray = np.ndarray

        def __getattr__(self, name):
            return getattr(np, name)

        def lexsort(self, keys):
            assert isinstance(keys, np.ndarray)
            assert keys.shape == (2, 5)
            np.testing.assert_allclose(keys[0], [1.0, 0.5, 0.0, 0.51, 0.01])
            np.testing.assert_array_equal(keys[1], [1, 0, 1, 0, 1])
            return np.lexsort(keys)

    selected = select_initial_clusters(
        CuPy123LexsortContract(),
        np.array([1, 0, 1, 0, 1], dtype=np.int32),
        np.array([1.0, 0.5, 0.0, 0.51, 0.01], dtype=np.float32),
        GaussianElevationConfig(),
    )
    np.testing.assert_array_equal(selected, [False, True, True, True, True])


def test_initial_cluster_atomic_count_uses_cupy_12_3_supported_dtype():
    class CuPy123MaximumContract:
        @staticmethod
        def at(target, indices, values):
            supported = (np.int32, np.float32, np.float64, np.uint32, np.uint64)
            if target.dtype.type not in supported:
                raise TypeError("CuPy 12.3 maximum.at unsupported dtype")
            np.maximum.at(target, indices, values)

    class CuPy123AtomicContract:
        ndarray = np.ndarray
        maximum = CuPy123MaximumContract()

        def __getattr__(self, name):
            return getattr(np, name)

    selected = select_initial_clusters(
        CuPy123AtomicContract(),
        np.array([0, 0, 0, 1], dtype=np.int64),
        np.array([0.0, 0.01, 1.0, 2.0], dtype=np.float64),
        GaussianElevationConfig(),
    )
    np.testing.assert_array_equal(selected, [True, True, False, True])


def test_initial_cluster_tie_break_avoids_lossy_float_atomic_roundtrip():
    class LossyCuPy123Minimum:
        @staticmethod
        def at(target, indices, values):
            rounded = values.astype(np.float32).astype(target.dtype)
            np.minimum.at(target, indices, rounded)

    class LossyCuPy123AtomicContract:
        ndarray = np.ndarray
        minimum = LossyCuPy123Minimum()

        def __getattr__(self, name):
            return getattr(np, name)

    selected = select_initial_clusters(
        LossyCuPy123AtomicContract(),
        np.array([0, 0, 0, 1, 1], dtype=np.int64),
        np.array([0.0, 0.08, 0.5, 1.0, 1.01], dtype=np.float64),
        GaussianElevationConfig(),
    )
    np.testing.assert_array_equal(selected, [True, True, False, True, True])


def test_all_rejected_update_does_not_call_cupy_bincount_with_empty_input():
    class CuPy123EmptyBincountContract:
        ndarray = np.ndarray

        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def bincount(values, *args, **kwargs):
            if values.size == 0:
                raise ValueError("CuPy 12.3 cannot bincount an empty array")
            return np.bincount(values, *args, **kwargs)

    state = _state(1)
    state["valid"][0] = True
    state["mean"][0] = 0.0
    state["variance"][0] = 0.01
    state["variance_updated_at"][0] = 0.0
    result = update_gaussian_elevation(
        CuPy123EmptyBincountContract(),
        np.array([0], dtype=np.int64),
        np.array([10.0], dtype=np.float64),
        np.array([0.01], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        1.0,
        cfg=GaussianElevationConfig(),
        **state,
    )
    assert result.accepted_cell_ids.size == 0
    assert result.reject_fraction[0] == 1.0


def test_frame_aggregation_ignores_unaccepted_points_and_groups_cells():
    cfg = GaussianElevationConfig()
    result = aggregate_frame_observations(
        np,
        np.array([8, 2, 8, 2]),
        np.array([100.0, 2.0, 4.0, 6.0]),
        np.array([0.01, 1.0, 0.25, 1.0]),
        np.array([False, True, True, True]),
        cfg,
    )
    np.testing.assert_array_equal(result.cell_ids, [2, 8])
    np.testing.assert_array_equal(result.count, [2, 1])
    np.testing.assert_allclose(result.mean, [4.0, 4.0])


def test_device_grouping_has_no_point_or_cell_scaled_python_loops():
    cluster_tree = ast.parse(inspect.getsource(select_initial_clusters))
    assert not any(
        isinstance(node, (ast.For, ast.While)) for node in ast.walk(cluster_tree)
    )
    update_tree = ast.parse(inspect.getsource(update_gaussian_elevation))
    dynamic_range_loops = [
        node
        for node in ast.walk(update_tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    ]
    assert not dynamic_range_loops
    module_source = inspect.getsource(
        __import__(
            "roadrunner_mapping.geometry.gaussian_elevation",
            fromlist=["dummy"],
        )
    )
    assert ".item(" not in module_source


def test_group_extrema_use_only_cupy_12_3_supported_ufunc_operations():
    module_source = inspect.getsource(
        __import__(
            "roadrunner_mapping.geometry.gaussian_elevation",
            fromlist=["dummy"],
        )
    )
    assert "maximum.reduceat" not in module_source
    assert "minimum.reduceat" not in module_source


def test_float16_observations_are_promoted_before_precision_products():
    result = aggregate_frame_observations(
        np,
        np.array([0, 0], dtype=np.int32),
        np.array([60000.0, 60000.0], dtype=np.float16),
        np.array([0.001, 0.002], dtype=np.float16),
        np.array([True, True]),
        GaussianElevationConfig(),
    )
    assert result.mean.dtype == np.float32
    assert result.variance.dtype == np.float32
    assert np.all(np.isfinite(result.mean))
    assert result.mean[0] == pytest.approx(60000.0, rel=1e-5)


def test_point_variance_promotes_float16_range_before_squaring():
    result = point_height_variance(
        np,
        np.array([1000.0], dtype=np.float16),
        GaussianElevationConfig(),
    )
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [0.25])


def _state(size=4):
    return {
        "mean": np.zeros(size, dtype=np.float64),
        "variance": np.full(size, 0.25, dtype=np.float64),
        "valid": np.zeros(size, dtype=bool),
        "support_count": np.zeros(size, dtype=np.int64),
        "accepted_frame_count": np.zeros(size, dtype=np.int64),
        "last_seen": np.full(size, -np.inf, dtype=np.float64),
        "variance_updated_at": np.full(size, -np.inf, dtype=np.float64),
        "conflict": np.zeros(size, dtype=np.float64),
        "conflict_updated_at": np.full(size, -np.inf, dtype=np.float64),
        "ground_sum": np.zeros(size, dtype=np.float64),
        "min_z": np.full(size, np.inf, dtype=np.float64),
        "max_z": np.full(size, -np.inf, dtype=np.float64),
    }


def _update(state, cell_ids, heights, variances, ground_weights, timestamp, cfg):
    return update_gaussian_elevation(
        np,
        np.asarray(cell_ids, dtype=np.int64),
        np.asarray(heights, dtype=np.float64),
        np.asarray(variances, dtype=np.float64),
        np.asarray(ground_weights, dtype=np.float64),
        timestamp,
        cfg=cfg,
        **state,
    )


def _raw_update(
    state,
    *,
    cell_ids=np.array([0], dtype=np.int32),
    heights=np.array([1.0], dtype=np.float32),
    variances=np.array([0.01], dtype=np.float32),
    ground_weights=np.array([1.0], dtype=np.float32),
    validate_state_values=True,
):
    return update_gaussian_elevation(
        np,
        cell_ids,
        heights,
        variances,
        ground_weights,
        1.0,
        validate_state_values=validate_state_values,
        cfg=GaussianElevationConfig(),
        **state,
    )


def _assert_state_equal(actual, expected):
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_readonly_state_is_rejected_atomically_before_any_write():
    state = _state(2)
    snapshot = {name: values.copy() for name, values in state.items()}
    state["conflict"].flags.writeable = False
    with pytest.raises(ValueError):
        _raw_update(state)
    _assert_state_equal(state, snapshot)


@pytest.mark.parametrize(
    ("name", "bad_value"),
    [
        ("mean", np.zeros(2, dtype=np.float16)),
        ("mean", np.zeros(2, dtype=np.longdouble)),
        ("valid", np.zeros(2, dtype=np.uint8)),
        ("support_count", np.zeros(2, dtype=np.uint32)),
        ("accepted_frame_count", np.zeros(2, dtype=np.int16)),
    ],
)
def test_state_arrays_reject_wrong_dtypes_without_mutation(name, bad_value):
    state = _state(2)
    state[name] = bad_value
    snapshot = {key: values.copy() for key, values in state.items()}
    with pytest.raises(TypeError):
        _raw_update(state)
    _assert_state_equal(state, snapshot)


def test_state_arrays_must_already_belong_to_requested_backend():
    state = _state(2)
    state["mean"] = [0.0, 0.0]
    with pytest.raises(TypeError, match="backend"):
        _raw_update(state)


@pytest.mark.parametrize(
    ("state_name", "bad_value"),
    [
        ("mean", np.nan),
        ("variance", 0.0),
        ("conflict", 1.5),
        ("ground_sum", -1.0),
        ("support_count", -1),
    ],
)
def test_illegal_state_values_are_rejected_before_mutation(state_name, bad_value):
    state = _state(2)
    state[state_name][1] = bad_value
    snapshot = {name: values.copy() for name, values in state.items()}
    with pytest.raises(ValueError):
        _raw_update(state)
    _assert_state_equal(state, snapshot)


def test_state_value_oracle_can_be_disabled_for_owned_online_state():
    state = _state(2)
    state["mean"][1] = np.nan

    result = _raw_update(state, validate_state_values=False)

    np.testing.assert_array_equal(result.touched_cell_ids, [0])
    assert state["mean"][0] == pytest.approx(1.0)
    assert np.isnan(state["mean"][1])


def test_point_range_checks_do_not_short_circuit_into_multiple_device_syncs():
    module = __import__(
        "roadrunner_mapping.geometry.gaussian_elevation",
        fromlist=["dummy"],
    )
    module_source = inspect.getsource(module)
    flag_source = inspect.getsource(module._device_error_flags)

    assert "bool(xp.any(cell_ids < 0)) or" not in module_source
    assert "bool(xp.any(~xp.isfinite(ranges)))" not in module_source
    assert flag_source.count(".tolist()") == 1


def test_masks_and_cell_ids_require_exact_safe_dtypes():
    with pytest.raises(TypeError, match="boolean"):
        aggregate_frame_observations(
            np,
            np.array([0], dtype=np.int32),
            np.array([1.0], dtype=np.float32),
            np.array([0.01], dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
            GaussianElevationConfig(),
        )
    state = _state(1)
    with pytest.raises(TypeError, match="signed integer"):
        _raw_update(state, cell_ids=np.array([0], dtype=np.uint32))


@pytest.mark.parametrize("counter_name", ["support_count", "accepted_frame_count"])
def test_counter_overflow_is_rejected_atomically(counter_name):
    state = _state(1)
    state[counter_name] = np.array(
        [np.iinfo(np.int32).max], dtype=np.int32
    )
    snapshot = {name: values.copy() for name, values in state.items()}
    with pytest.raises(OverflowError):
        _raw_update(state)
    _assert_state_equal(state, snapshot)


def test_float32_state_path_preserves_state_dtypes():
    state = _state(2)
    for name in (
        "mean",
        "variance",
        "last_seen",
        "variance_updated_at",
        "conflict",
        "conflict_updated_at",
        "ground_sum",
        "min_z",
        "max_z",
    ):
        state[name] = state[name].astype(np.float32)
    state["support_count"] = state["support_count"].astype(np.int32)
    state["accepted_frame_count"] = state["accepted_frame_count"].astype(np.int32)
    _raw_update(state)
    assert all(
        state[name].dtype == np.float32
        for name in (
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
    )
    assert state["support_count"].dtype == np.int32
    assert state["accepted_frame_count"].dtype == np.int32


def test_unknown_cells_have_unique_single_and_multi_point_initial_variance():
    cfg = GaussianElevationConfig()
    state = _state()
    result = _update(
        state,
        [0, 1, 1],
        [2.0, 4.0, 4.02],
        [0.01, 0.01, 0.01],
        [0.25, 0.5, 0.75],
        1.0,
        cfg,
    )
    single_rf = 0.01 + cfg.pose_vertical_variance
    multi_scatter = 0.0001
    multi_rf = 0.005 + multi_scatter / 2.0 + cfg.pose_vertical_variance
    assert state["mean"][0] == pytest.approx(2.0)
    assert state["variance"][0] == pytest.approx(
        max(single_rf, cfg.single_point_initial_variance)
    )
    assert state["mean"][1] == pytest.approx(4.01)
    assert state["variance"][1] == pytest.approx(
        max(multi_rf, cfg.multi_point_initial_variance)
    )
    np.testing.assert_array_equal(result.touched_cell_ids, [0, 1])
    np.testing.assert_array_equal(result.accepted_cell_ids, [0, 1])
    np.testing.assert_array_equal(result.accepted_point_mask, [True, True, True])
    np.testing.assert_array_equal(state["support_count"][:2], [1, 2])
    np.testing.assert_array_equal(state["accepted_frame_count"][:2], [1, 1])
    np.testing.assert_allclose(state["ground_sum"][:2], [0.25, 1.25])
    np.testing.assert_allclose(state["min_z"][:2], [2.0, 4.0])
    np.testing.assert_allclose(state["max_z"][:2], [2.0, 4.02])


def test_unknown_cluster_rejection_inflates_initialized_variance():
    cfg = GaussianElevationConfig()
    state = _state(1)
    result = _update(
        state,
        [0, 0, 0],
        [0.0, 0.01, 1.0],
        [0.01, 0.01, 0.01],
        [1.0, 1.0, 0.0],
        1.0,
        cfg,
    )
    frame_variance = 0.005 + 0.000025 / 2.0 + cfg.pose_vertical_variance
    expected = max(frame_variance, cfg.multi_point_initial_variance)
    expected += cfg.outlier_variance_per_frame / 3.0
    assert state["variance"][0] == pytest.approx(expected)
    assert state["conflict"][0] == pytest.approx(1.0 / 3.0)
    np.testing.assert_array_equal(result.accepted_point_mask, [True, True, False])


def test_known_cell_uses_process_gate_kalman_then_outlier_inflation():
    cfg = replace(
        GaussianElevationConfig(),
        process_variance_per_sec=0.01,
        pose_vertical_variance=0.001,
    )
    state = _state(1)
    state["valid"][0] = True
    state["mean"][0] = 0.0
    state["variance"][0] = 0.04
    state["variance_updated_at"][0] = 1.0
    result = _update(
        state,
        [0, 0],
        [0.1, 10.0],
        [0.01, 0.01],
        [0.8, 0.2],
        3.0,
        cfg,
    )
    predicted = 0.04 + 2.0 * 0.01
    frame_variance = 0.01 + cfg.pose_vertical_variance
    gain = predicted / (predicted + frame_variance)
    expected_posterior = (1.0 - gain) * predicted
    assert state["mean"][0] == pytest.approx(gain * 0.1)
    assert state["variance"][0] == pytest.approx(
        expected_posterior + cfg.outlier_variance_per_frame * 0.5
    )
    assert state["support_count"][0] == 1
    assert state["accepted_frame_count"][0] == 1
    assert state["last_seen"][0] == 3.0
    assert state["variance_updated_at"][0] == 3.0
    assert state["conflict_updated_at"][0] == 3.0
    np.testing.assert_array_equal(result.accepted_point_mask, [True, False])


def test_all_rejected_only_predicts_and_inflates_without_frame_count():
    cfg = replace(
        GaussianElevationConfig(), process_variance_per_sec=0.01
    )
    state = _state(1)
    state["valid"][0] = True
    state["mean"][0] = 1.0
    state["variance"][0] = 0.02
    state["variance_updated_at"][0] = 5.0
    state["last_seen"][0] = 4.0
    result = _update(state, [0, 0], [9.0, 10.0], [0.001, 0.001], [1, 1], 7.0, cfg)
    assert state["mean"][0] == 1.0
    assert state["variance"][0] == pytest.approx(0.02 + 0.02 + 0.01)
    assert state["accepted_frame_count"][0] == 0
    assert state["last_seen"][0] == 4.0
    assert state["conflict"][0] == 1.0
    assert result.accepted_cell_ids.size == 0


def test_gate_denominator_includes_shared_pose_variance():
    cfg = replace(
        GaussianElevationConfig(),
        mahalanobis_threshold=1.0,
        pose_vertical_variance=0.04,
        process_variance_per_sec=0.0,
    )
    state = _state(1)
    state["valid"][0] = True
    state["variance"][0] = 0.01
    state["variance_updated_at"][0] = 1.0
    # 0.2 is outside sqrt(P + R), but inside sqrt(P + R + shared pose).
    result = _update(state, [0], [0.2], [0.01], [1.0], 2.0, cfg)
    assert result.accepted_point_mask[0]


def test_basic_batch_update_matches_analytic_sequential_precision():
    cfg = replace(
        GaussianElevationConfig(),
        pose_vertical_variance=1e-12,
        frame_variance_min=1e-12,
        posterior_variance_min=1e-12,
        max_effective_points_per_frame=10,
        process_variance_per_sec=0.0,
    )
    state = _state(1)
    state["valid"][0] = True
    state["mean"][0] = 2.0
    state["variance"][0] = 0.2
    state["variance_updated_at"][0] = 1.0
    _update(state, [0, 0], [3.0, 3.0], [0.2, 0.1], [1.0, 1.0], 2.0, cfg)
    expected_variance = 1.0 / (1.0 / 0.2 + 1.0 / 0.2 + 1.0 / 0.1)
    expected_mean = expected_variance * (2.0 / 0.2 + 3.0 / 0.2 + 3.0 / 0.1)
    assert state["variance"][0] == pytest.approx(expected_variance, abs=2e-12)
    assert state["mean"][0] == pytest.approx(expected_mean, abs=4e-12)


def test_invalid_points_are_excluded_before_touching_state():
    state = _state(2)
    result = _update(
        state,
        [0, 0, 1],
        [np.nan, 1.0, 2.0],
        [0.01, np.inf, 0.01],
        [1.0, 1.0, np.nan],
        1.0,
        GaussianElevationConfig(),
    )
    assert result.touched_cell_ids.size == 0
    assert not np.any(state["valid"])


def test_update_requires_strictly_increasing_time_for_known_touched_cells():
    cfg = GaussianElevationConfig()
    state = _state(1)
    _update(state, [0], [1.0], [0.01], [1.0], 5.0, cfg)
    snapshot = {name: values.copy() for name, values in state.items()}
    for timestamp in (5.0, 4.0):
        with pytest.raises(ValueError, match="strictly increasing"):
            _update(state, [0], [1.0], [0.01], [1.0], timestamp, cfg)
        for name in state:
            np.testing.assert_array_equal(state[name], snapshot[name])
    # A duplicate frame is a caller no-op: not invoking the core changes nothing.
    for name in state:
        np.testing.assert_array_equal(state[name], snapshot[name])


def test_numpy_cupy_state_and_update_parity():
    cp = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:  # pragma: no cover - depends on host CUDA runtime
        pytest.skip("CUDA is unavailable: {}".format(error))
    if device_count < 1:  # pragma: no cover - depends on host CUDA runtime
        pytest.skip("CUDA is unavailable: no CUDA device")

    cfg = GaussianElevationConfig(initial_cluster_gap_m=0.08)
    numpy_state = _state(3)
    cupy_state = {name: cp.asarray(values) for name, values in numpy_state.items()}
    frames = (
        (
            [0, 0, 0, 1, 1],
            [0.0, 0.08, 0.5, 1.0, 1.01],
            [0.01, 0.02, 0.03, 0.01, 0.02],
            [1.0, 0.8, 0.0, 0.5, 0.7],
            1.0,
        ),
        (
            [0, 0, 1, 1, 2],
            [0.04, 2.0, 5.0, 6.0, -1.0],
            [0.01, 0.01, 0.01, 0.01, 0.04],
            [1.0, 0.0, 0.2, 0.2, 0.3],
            2.0,
        ),
    )
    for cell_ids, heights, point_variance, ground, timestamp in frames:
        numpy_result = _update(
            numpy_state,
            cell_ids,
            heights,
            point_variance,
            ground,
            timestamp,
            cfg,
        )
        cupy_result = update_gaussian_elevation(
            cp,
            cp.asarray(cell_ids, dtype=cp.int64),
            cp.asarray(heights, dtype=cp.float64),
            cp.asarray(point_variance, dtype=cp.float64),
            cp.asarray(ground, dtype=cp.float64),
            timestamp,
            cfg=cfg,
            **cupy_state,
        )
        for field_name in (
            "touched_cell_ids",
            "accepted_cell_ids",
            "accepted_point_mask",
            "rejected_point_mask",
            "reject_fraction",
        ):
            np.testing.assert_allclose(
                getattr(numpy_result, field_name),
                cp.asnumpy(getattr(cupy_result, field_name)),
                rtol=1e-5,
                atol=1e-6,
            )

    for name, numpy_values in numpy_state.items():
        np.testing.assert_allclose(
            numpy_values,
            cp.asnumpy(cupy_state[name]),
            rtol=1e-5,
            atol=1e-6,
        )
