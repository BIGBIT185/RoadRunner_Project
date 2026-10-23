import ast
from pathlib import Path
import threading
import time
from types import MethodType
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from roadrunner_mapping.online import online_backend
from roadrunner_mapping.online.online_backend import (
    BoundedCameraInfoCache,
    EomtHealthMonitor,
    camera_info_matches_image,
    initialize_inline_segmenter,
    resolve_semantic_backend,
)
from roadrunner_mapping.geometry.gaussian_elevation import (
    GaussianElevationConfig,
    point_height_variance,
)
from roadrunner_mapping.online.online_fusion import RecentFrameWindow
from roadrunner_mapping.online.incremental_semantic_grid import (
    IncrementalSemanticGrid,
)
from roadrunner_mapping.online.gpu_incremental_semantic_grid import (
    GpuIncrementalSemanticGrid,
)
from roadrunner_mapping.geometry.gpu_rolling_elevation import (
    GpuRollingElevationMap,
)
from roadrunner_mapping.geometry.schema import MapSpec


PACKAGE = Path(__file__).resolve().parents[1]
NODE = PACKAGE / "scripts/mapping_node.py"

GEOMETRY_TOP_LEVEL_TIMING = {
    "geometry_prepare_sec": 0.01,
    "geometry_drift_sec": 0.02,
    "geometry_visibility_sec": 0.03,
    "geometry_map_update_sec": 0.04,
    "geometry_world_fields_sec": 0.05,
    "geometry_risk_sec": 0.06,
    "geometry_sample_sec": 0.07,
    "geometry_transfer_sec": 0.08,
}
GEOMETRY_MAP_TIMING = {
    "geometry_map_center_sec": 0.001,
    "geometry_map_maintenance_sec": 0.002,
    "geometry_map_backup_sec": 0.003,
    "geometry_map_clear_sec": 0.004,
    "geometry_map_visibility_apply_sec": 0.005,
    "geometry_map_fusion_sec": 0.01,
    "geometry_map_commit_sec": 0.015,
}
GEOMETRY_TIMING = {**GEOMETRY_TOP_LEVEL_TIMING, **GEOMETRY_MAP_TIMING}


class _NumpyDevice:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _NumpyPool:
    def set_limit(self, size):
        self.limit = size

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0


class _NumpyDeviceScalar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def __int__(self):
        return int(self.value)


class _NumpyCupy:
    ndarray = np.ndarray
    cuda = SimpleNamespace(Device=lambda index: _NumpyDevice())

    def __init__(self):
        self._pool = _NumpyPool()

    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, values):
        return np.asarray(values)

    def count_nonzero(self, values):
        return _NumpyDeviceScalar(np.count_nonzero(values))

    def get_default_memory_pool(self):
        return self._pool


def _numpy_map(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module
    from roadrunner_mapping.geometry.schema import MapSpec

    backend = _NumpyCupy()
    monkeypatch.setattr(
        module,
        "_load_gpu_modules",
        lambda: module._GpuModules(cp=backend, ndimage=ndimage),
    )
    return module.GpuRollingElevationMap(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 0.2),
        world_map_length=4.0,
        gaussian_config=GaussianElevationConfig(),
    )


def _node_function(name, extra_namespace=None):
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "np": np,
        "time": time,
        "point_height_variance": point_height_variance,
        "GaussianElevationConfig": GaussianElevationConfig,
        "IncrementalSemanticGrid": IncrementalSemanticGrid,
        "GpuIncrementalSemanticGrid": GpuIncrementalSemanticGrid,
    }
    if extra_namespace:
        namespace.update(extra_namespace)
    exec(compile(ast.fix_missing_locations(module), str(NODE), "exec"), namespace)
    return namespace[name]


def _node_method(name, namespace):
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    method = next(
        member
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoadRunnerMapping"
        for member in node.body
        if isinstance(member, ast.FunctionDef) and member.name == name
    )
    wrapper = ast.ClassDef(
        name="ProcessNode",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(body=[wrapper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(NODE), "exec"), namespace)
    return namespace["ProcessNode"].__dict__[name]


class _Stamp:
    def __init__(self, seconds):
        self.seconds = float(seconds)

    def to_sec(self):
        return self.seconds

    def to_nsec(self):
        return int(round(self.seconds * 1e9))


def _cloud_message(timestamp, seq=0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_Stamp(timestamp), seq=seq),
        height=1,
        width=1,
    )


class _GeometryMapProbe:
    def __init__(self):
        self.gaussian_config = GaussianElevationConfig()
        self.fail = False
        self.frames = []
        self.clear_count = 0
        self.last_geometry_timing = dict(GEOMETRY_TIMING)

    def clear(self):
        self.clear_count += 1
        self.frames.clear()

    def advance(self, frame_ids, frame_loader, *, target_world_from_body):
        loaded = [frame_loader(frame_id) for frame_id in frame_ids]
        prepared = [
            GpuRollingElevationMap._prepare_offline_frame(None, frame)
            for frame in loaded
        ]
        if self.fail:
            raise RuntimeError("injected map failure")
        self.frames = loaded
        return {"prepared": prepared}


def _process_node(*, strict_window=False, decoder=None):
    if decoder is None:
        decoder = lambda message, **kwargs: np.array(
            [[1.0, 0.0, 0.2, 10.0]], dtype=np.float32
        )
    ros = SimpleNamespace(
        Time=SimpleNamespace(now=lambda: _Stamp(100.0)),
        logwarn=lambda *args: None,
        logwarn_throttle=lambda *args: None,
    )
    namespace = {
        "np": np,
        "time": time,
        "SimpleNamespace": SimpleNamespace,
        "rospy": ros,
        "timestamp_is_future": lambda *args, **kwargs: False,
        "decode_pointcloud2_gpu": lambda *args, **kwargs: None,
        "decode_goose_cloud": decoder,
        "decode_goose_cloud_with_rows": lambda message, **kwargs: (
            decoder(message, **kwargs),
            np.zeros(1, dtype=np.int32),
        ),
        "transform_points": lambda points, transform: (
            points @ transform[:3, :3].T + transform[:3, 3]
        ),
        "apply_row_transforms": lambda points, rows, transforms: points,
        "_lidar_height_variance": _node_function("_lidar_height_variance"),
        "_geometry_frame_time_action": _node_function(
            "_geometry_frame_time_action"
        ),
        "_record_successful_traversability_publish": _node_function(
            "_record_successful_traversability_publish"
        ),
        "_GEOMETRY_TOP_LEVEL_TIMING_NAMES": tuple(GEOMETRY_TOP_LEVEL_TIMING),
        "_GEOMETRY_MAP_TIMING_NAMES": tuple(GEOMETRY_MAP_TIMING),
        "_GEOMETRY_DIAGNOSTIC_TIMING_NAMES": tuple(GEOMETRY_TIMING),
        "_snapshot_online_state": _node_function("_snapshot_online_state"),
        "_restore_online_state": _node_function("_restore_online_state"),
        "RecentFrameWindow": RecentFrameWindow,
        "apply_grass_rule_to_accumulated_geometry": lambda geometry, semantic: geometry,
        "geometry_risk_variants": lambda geometry, **kwargs: (geometry, geometry),
        "semantic_risk_variants": lambda semantic, **kwargs: (semantic, semantic),
        "geometry_cvar_risk": lambda geometry, **kwargs: geometry,
        "semantic_cvar_risk": lambda semantic, **kwargs: semantic,
        "fuse_accumulated_geometry_semantics": lambda geometry, semantic: {},
    }
    process = _node_method("_process_geometry", namespace)
    process_active_epoch = _node_method(
        "_process_geometry_active_epoch", namespace
    )
    node = SimpleNamespace()
    node._process_geometry = MethodType(process, node)
    node._process_geometry_active_epoch = MethodType(
        process_active_epoch, node
    )
    node.active_epoch = 0
    node.last_stage = "idle"
    node.last_traversability_publish_monotonic = None
    node.traversability_publish_interval_ema_sec = None
    node.traversability_publish_hz = 0.0
    node.last_timestamp = 1.0
    node.scan_period_estimate_sec = 0.1
    node.scan_period_config_sec = 0.1
    node.scan_period_min_sec = 0.08
    node.scan_period_max_sec = 0.12
    node.scan_period_ema_alpha = 0.1
    node.geometry_ingested_ids = {"prior"}
    node.geometry_window = RecentFrameWindow(1.0)
    prior = SimpleNamespace(frame_id="prior", timestamp=1.0)
    node.geometry_window.append(prior)
    node.geometry_by_id = {"prior": prior}
    node.local_from_world = None
    node.pose_lock = threading.Lock()
    node.geometry_map = _GeometryMapProbe()
    node.semantic_map = None
    node.semantic_map_lock = threading.Lock()
    node.gaussian_config = node.geometry_map.gaussian_config
    node.gpu_pointcloud = False
    node.strict_window = strict_window
    node.scan_deskew_mode = "off"
    node.minimum_range = 0.0
    node.maximum_range = 10.0
    node.minimum_forward_x = None
    node.future_stamp_tolerance_sec = 1.0
    node.lidar_frame = "lidar"
    node.base_frame = "base"
    node.world_frame = "world"
    node.last_semantic_stats = {"semantic_valid_cells": 0}
    node.publish_legacy_probability = True
    node.use_geometry_cvar = True
    node.use_semantic_cvar = True
    node.processed = 0
    node.geometry_epoch_discards = 0
    node.gpu_decode_fallbacks = 0
    node.scan_deskew_applied = 0
    node.scan_deskew_fallbacks = 0
    node.scan_transform_cache = {}
    node.scan_transform_cache_lock = threading.Lock()
    node._update_scan_period = MethodType(
        _node_method("_update_scan_period", namespace), node
    )
    node._reset_stream = MethodType(_node_method("_reset_stream", namespace), node)
    node._lookup_fail = False

    def local_pose(self, stamp):
        if self.local_from_world is None:
            self.local_from_world = np.eye(4)
        return np.eye(4)

    def lookup(self, target, source, stamp):
        if self._lookup_fail:
            raise RuntimeError("injected TF failure")
        return np.eye(4)

    node._local_pose = MethodType(local_pose, node)
    node._lookup = MethodType(lookup, node)
    node._scan_row_world_from_lidar = MethodType(
        lambda self, *args, **kwargs: None, node
    )
    node._geometry_teacher = MethodType(lambda self, maps: {}, node)
    node._empty_semantic_snapshot = MethodType(
        lambda self: {"semantic_valid_cells": 0}, node
    )
    node._publish_maps = MethodType(lambda self, *args: None, node)
    node._publish_diagnostics = MethodType(lambda self, *args: None, node)
    return node


def test_successful_publish_rate_reports_four_hz_from_controlled_clock():
    node = SimpleNamespace(
        last_traversability_publish_monotonic=None,
        traversability_publish_interval_ema_sec=None,
        traversability_publish_hz=0.0,
    )
    namespace = {"time": time, "np": np}
    update = _node_function(
        "_record_successful_traversability_publish", namespace
    )

    for now in (10.0, 10.25, 10.5, 10.75):
        update(node, now=now)

    assert node.traversability_publish_hz == pytest.approx(4.0)


def test_failed_map_publish_does_not_update_publish_rate():
    node = _process_node()

    def fail_publish(self, *args):
        raise RuntimeError("injected publish failure")

    node._publish_maps = MethodType(fail_publish, node)

    with pytest.raises(RuntimeError, match="injected publish failure"):
        node._process_geometry(0, _cloud_message(1.1, seq=2))

    assert node.last_traversability_publish_monotonic is None
    assert node.traversability_publish_interval_ema_sec is None
    assert node.traversability_publish_hz == 0.0


def test_diagnostics_expose_serializable_lifecycle_stats_and_publish_rate():
    class KeyValue:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class DiagnosticStatus:
        OK = 0
        WARN = 1
        ERROR = 2

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class GeometryWindow:
        span_sec = 0.5

        def __len__(self):
            return 1

    diagnostic_status = _node_method(
        "_diagnostic_status",
        {
            "DiagnosticStatus": DiagnosticStatus,
            "KeyValue": KeyValue,
            "_GEOMETRY_TOP_LEVEL_TIMING_NAMES": tuple(
                GEOMETRY_TOP_LEVEL_TIMING
            ),
            "_GEOMETRY_MAP_TIMING_NAMES": tuple(GEOMETRY_MAP_TIMING),
            "_GEOMETRY_DIAGNOSTIC_TIMING_NAMES": tuple(GEOMETRY_TIMING),
        },
    )
    lifecycle_stats = {
        "visibility_input_rays": 0,
        "visibility_cleaned_cells": 0,
        "upper_bound_cells": 0,
        "geometry_valid_cells": 0,
        "height_drift_bias_z": 0.0,
        "height_drift_sample_cells": 0,
    }
    node = SimpleNamespace(
        processed=1,
        semantic_enabled=False,
        last_semantic_error="",
        eomt_health=None,
        last_error="",
        last_stage="idle",
        last_semantic_stage="idle",
        window_sec=1.0,
        geometry_window=GeometryWindow(),
        semantic_decay_tau_sec=3.0,
        semantic_stale_sec=1.0,
        received=1,
        dropped=0,
        failures=0,
        geometry_epoch_discards=0,
        semantic_processed=0,
        semantic_dropped=0,
        semantic_failures=0,
        semantic_epoch_discards=0,
        gpu_decode_fallbacks=0,
        gpu_pointcloud=True,
        gpu_semantic_map=True,
        sample_mode="bilinear",
        sample_min_valid_weight=0.5,
        scan_deskew_mode="off",
        scan_stamp_reference="end",
        scan_row_direction=1,
        scan_period_estimate_sec=0.1,
        scan_deskew_applied=0,
        scan_deskew_fallbacks=0,
        strict_window=False,
        semantic_backend="disabled",
        publish_legacy_probability=True,
        camera_info_late_matched=0,
        camera_info_pending_dropped=0,
        camera_info_cache=SimpleNamespace(evicted_count=0),
        geometry_map=SimpleNamespace(
            memory_pool_used_bytes=0,
            memory_pool_total_bytes=0,
            last_lifecycle_stats=lifecycle_stats,
        ),
        traversability_publish_hz=4.0,
        last_semantic_stats={},
        semantic_matcher=None,
        device="cuda:0",
    )

    status = diagnostic_status(
        node,
        **GEOMETRY_TIMING,
        geometry_profile_gap_sec=0.64,
    )
    values = {item.key: item.value for item in status.values}

    assert values["traversability_publish_hz"] == "4.000"
    assert {name: values[name] for name in GEOMETRY_TIMING} == {
        name: f"{value:.4f}" for name, value in GEOMETRY_TIMING.items()
    }
    assert values["geometry_profile_gap_sec"] == "0.6400"
    assert {name: values[name] for name in lifecycle_stats} == {
        name: str(value) for name, value in lifecycle_stats.items()
    }
    assert all(isinstance(key, str) for key in values)
    assert all(isinstance(value, str) for value in values.values())


def test_geometry_subtimings_do_not_reduce_top_level_unaccounted_time():
    node = _process_node()
    process_globals = node._process_geometry_active_epoch.__func__.__globals__
    ticks = iter(float(value) for value in range(18))
    process_globals["time"] = SimpleNamespace(perf_counter=lambda: next(ticks))

    node._process_geometry(0, _cloud_message(1.1, seq=3))

    assert {name: node.last_timing[name] for name in GEOMETRY_TIMING} == (
        GEOMETRY_TIMING
    )
    assert node.last_timing["geometry_map_update_sec"] == pytest.approx(
        sum(GEOMETRY_MAP_TIMING.values())
    )
    assert node.last_timing["geometry_sec"] == pytest.approx(1.0)
    assert node.last_timing["geometry_profile_gap_sec"] == pytest.approx(0.64)
    assert node.last_timing["unaccounted_sec"] == pytest.approx(7.0)


def _process_state(node):
    map_frames = getattr(node.geometry_map, "frames", None)
    return {
        "active_epoch": node.active_epoch,
        "last_timestamp": node.last_timestamp,
        "scan_period": node.scan_period_estimate_sec,
        "local_from_world": (
            None if node.local_from_world is None else node.local_from_world.copy()
        ),
        "window": tuple(frame.frame_id for frame in node.geometry_window),
        "cache": tuple(sorted(node.geometry_by_id)),
        "ingested": frozenset(node.geometry_ingested_ids),
        "map_frames": (
            None
            if map_frames is None
            else tuple(frame.frame_id for frame in map_frames)
        ),
    }


def _assert_process_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if name == "local_from_world":
            if expected[name] is None:
                assert actual[name] is None
            else:
                np.testing.assert_array_equal(actual[name], expected[name])
        else:
            assert actual[name] == expected[name], name


_CPU_SEMANTIC_ARRAYS = (
    "_key_x",
    "_key_y",
    "_risk_sum",
    "_risk_sq_sum",
    "_weight_sum",
    "_grass_sum",
    "_count_sum",
    "_confidence",
    "_latest_risk",
    "_latest_confidence",
    "_last_update",
    "_last_conflict",
    "_last_dynamic",
)


def _cpu_semantic_state(semantic_map):
    return {
        "arrays": {
            name: getattr(semantic_map, name).copy()
            for name in _CPU_SEMANTIC_ARRAYS
        },
        "histogram": tuple(
            histogram.copy() for histogram in semantic_map._risk_histogram
        ),
        "metadata": (
            semantic_map._last_timestamp,
            None
            if semantic_map._last_position is None
            else semantic_map._last_position.copy(),
            semantic_map._last_input_points,
            semantic_map._last_aggregated_cells,
        ),
    }


def _assert_cpu_semantic_state_equal(actual, expected):
    for name, values in expected["arrays"].items():
        np.testing.assert_array_equal(actual["arrays"][name], values)
    for actual_histogram, expected_histogram in zip(
        actual["histogram"], expected["histogram"]
    ):
        np.testing.assert_array_equal(actual_histogram, expected_histogram)
    actual_metadata = actual["metadata"]
    expected_metadata = expected["metadata"]
    assert actual_metadata[0] == expected_metadata[0]
    if expected_metadata[1] is None:
        assert actual_metadata[1] is None
    else:
        np.testing.assert_array_equal(actual_metadata[1], expected_metadata[1])
    assert actual_metadata[2:] == expected_metadata[2:]


def test_default_backend_is_eomt_and_invalid_backend_is_rejected():
    assert resolve_semantic_backend(None) == "eomt"
    assert resolve_semantic_backend("eomt") == "eomt"
    assert resolve_semantic_backend("deeplab") == "deeplab"
    assert resolve_semantic_backend("disabled") == "disabled"
    assert resolve_semantic_backend("none") == "disabled"
    with pytest.raises(ValueError, match="semantic_backend"):
        resolve_semantic_backend("silent_fallback")


def test_eomt_never_constructs_deeplab_but_deeplab_does():
    calls = []

    def factory():
        calls.append("constructed")
        return object()

    assert initialize_inline_segmenter("eomt", factory) is None
    assert calls == []
    assert initialize_inline_segmenter("disabled", factory) is None
    assert calls == []
    assert initialize_inline_segmenter("deeplab", factory) is not None
    assert calls == ["constructed"]


@pytest.mark.parametrize(
    ("camera_stamp_ns", "expected"),
    (
        (0, True),
        (1_000_000_000, True),
        (1_079_999_999, True),
        (1_080_000_001, False),
        (920_000_000, True),
        (919_999_999, False),
    ),
)
def test_camera_info_stamp_gate(camera_stamp_ns, expected):
    assert (
        camera_info_matches_image(
            camera_stamp_ns,
            1_000_000_000,
            tolerance_sec=0.08,
        )
        is expected
    )


def test_cached_pair_can_retain_callback_specific_camera_info():
    old = SimpleNamespace(header=SimpleNamespace(stamp="old"))
    pair = ("image", "cloud", old)
    newest = SimpleNamespace(header=SimpleNamespace(stamp="new"))
    assert pair[2] is old
    assert pair[2] is not newest


def test_camera_info_cache_matches_exact_image_before_newer_calibration():
    cache = BoundedCameraInfoCache(max_entries=4, tolerance_sec=0.08)
    exact = object()
    newer = object()
    cache.add(1_000_000_000, exact)
    cache.add(1_110_000_000, newer)
    assert cache.match(1_000_000_000) is exact


def test_camera_info_cache_accepts_timeless_and_nearest_within_gate():
    cache = BoundedCameraInfoCache(max_entries=4, tolerance_sec=0.08)
    timeless = object()
    cache.add(0, timeless)
    assert cache.match(20_000_000_000) is timeless

    cache = BoundedCameraInfoCache(max_entries=4, tolerance_sec=0.08)
    near = object()
    cache.add(1_040_000_000, near)
    assert cache.match(1_000_000_000) is near
    assert cache.match(2_000_000_000) is None


def test_health_uses_wall_time_and_error_is_persistent():
    health = EomtHealthMonitor(
        started_monotonic=10.0,
        readiness_timeout_sec=120.0,
        staleness_timeout_sec=5.0,
    )
    assert not health.can_publish
    assert health.check(now=129.9) is None
    assert health.check(now=130.1) == "eomt_readiness_timeout"
    assert health.check(now=999.0) == "eomt_readiness_timeout"
    assert not health.can_publish


def test_ready_output_becomes_stale_after_five_seconds():
    health = EomtHealthMonitor(started_monotonic=10.0)
    health.mark_ready(now=11.0)
    health.mark_output(now=12.0)
    assert health.can_publish
    assert health.check(now=17.0) is None
    assert health.check(now=17.001) == "eomt_output_stale"
    assert not health.can_publish


def test_stale_output_recovers_when_new_output_arrives():
    health = EomtHealthMonitor(started_monotonic=10.0)
    health.mark_ready(now=11.0)
    health.mark_output(now=12.0)
    assert health.check(now=17.001) == "eomt_output_stale"

    health.mark_output(now=18.0)

    assert health.fatal_reason == ""
    assert health.check(now=18.0) is None
    assert health.can_publish


def test_invalid_prediction_sets_fatal_state_and_reason_counter():
    health = EomtHealthMonitor(started_monotonic=0.0)
    health.mark_ready(now=0.1)
    health.fail("bad_encoding")
    health.fail("bad_encoding")
    assert health.fatal_reason == "bad_encoding"
    assert health.reason_counters["bad_encoding"] == 2
    assert not health.can_publish


def test_semantic_matcher_clear_removes_pending_and_processed_epochs():
    from roadrunner_mapping.semantics.semantic_transport import BoundedSemanticMatcher

    matcher = BoundedSemanticMatcher(max_age_sec=5.0, max_entries=4)
    matcher.add_sensor(1, "old-sensor", now=0.0)
    matcher.add_prediction(2, "old-prediction", now=0.0)
    matcher.add_sensor(3, "matched-sensor", now=0.0)
    assert matcher.add_prediction(3, "matched-prediction", now=0.1) == (
        "matched-sensor",
        "matched-prediction",
    )

    matcher.clear()

    assert matcher.pending_sensor_count == 0
    assert matcher.pending_prediction_count == 0
    assert matcher.add_sensor(3, "new-sensor", now=0.2) is None


@pytest.mark.parametrize(
    ("timestamp", "current_time", "expected"),
    (
        (130.0, 1.0, True),
        (10.08, 10.0, False),
        (11.0, 10.0, False),
        (11.000001, 10.0, True),
        (130.0, 0.0, False),
    ),
)
def test_future_timestamp_gate_discards_only_old_epoch_work(
    timestamp,
    current_time,
    expected,
):
    assert (
        online_backend.timestamp_is_future(
            timestamp,
            current_time,
            tolerance_sec=1.0,
        )
        is expected
    )


def test_cpu_fallback_variance_uses_lidar_range_before_world_transform():
    helper = _node_function("_lidar_height_variance")
    cfg = GaussianElevationConfig()
    points_lidar = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    translated_world = points_lidar.copy()
    translated_world[:, 0] += 1000.0

    actual = helper(points_lidar, cfg)
    expected = point_height_variance(np, np.array([5.0]), cfg)
    wrong_world_variance = point_height_variance(
        np, np.linalg.norm(translated_world, axis=1), cfg
    )

    np.testing.assert_allclose(actual, expected)
    assert actual[0] < wrong_world_variance[0]


def test_four_measurement_entry_paths_produce_identical_posterior(monkeypatch):
    cfg = GaussianElevationConfig()
    raw = np.array([[0.5, 0.0, 0.2]], dtype=np.float32)
    intensity = np.zeros(1, dtype=np.float32)
    transform = np.eye(4)
    explicit_variance = point_height_variance(
        np, np.linalg.norm(raw, axis=1), cfg
    ).astype(np.float32)

    states = []

    rigid = _numpy_map(monkeypatch)
    rigid.advance_pointcloud(
        "rigid",
        1.0,
        raw,
        intensity,
        world_from_lidar=transform,
        minimum_range=0.0,
        maximum_range=10.0,
        target_world_from_body=transform,
    )
    states.append(rigid)

    deskew = _numpy_map(monkeypatch)
    deskew.advance_pointcloud(
        "deskew",
        1.0,
        raw,
        intensity,
        world_from_lidar=transform,
        minimum_range=0.0,
        maximum_range=10.0,
        target_world_from_body=transform,
        row_world_from_lidar=np.asarray([transform]),
        point_rows=np.array([0], dtype=np.int32),
    )
    states.append(deskew)

    observation = SimpleNamespace(
        xyz=raw,
        ground_probability=np.ones(1, dtype=np.float32),
        dynamic=np.zeros(1, dtype=bool),
        height_variance=explicit_variance,
    )
    cpu = _numpy_map(monkeypatch)
    cpu_frame = SimpleNamespace(
        timestamp=1.0,
        world_from_body=transform,
        observation_world=observation,
    )
    cpu.advance(("cpu",), lambda _: cpu_frame, target_world_from_body=transform)
    states.append(cpu)

    offline = _numpy_map(monkeypatch)
    offline_frame = SimpleNamespace(
        timestamp=1.0,
        world_from_body=transform,
        observation_world=SimpleNamespace(
            xyz=raw,
            ground_probability=np.ones(1, dtype=np.float32),
            dynamic=np.zeros(1, dtype=bool),
        ),
    )
    offline.advance(
        ("offline",), lambda _: offline_frame, target_world_from_body=transform
    )
    states.append(offline)

    reference = states[0]
    for candidate in states[1:]:
        np.testing.assert_array_equal(candidate.elevation_valid, reference.elevation_valid)
        np.testing.assert_allclose(candidate.elevation_mean, reference.elevation_mean)
        np.testing.assert_allclose(
            candidate.elevation_variance, reference.elevation_variance
        )
        np.testing.assert_array_equal(
            candidate.geometry_support_count, reference.geometry_support_count
        )
        np.testing.assert_array_equal(
            candidate.accepted_frame_count, reference.accepted_frame_count
        )
        np.testing.assert_allclose(
            candidate.geometry_conflict_score, reference.geometry_conflict_score
        )


def test_strict_replay_reuses_cpu_variance_and_consumes_risk_once_per_frame(
    monkeypatch,
):
    transform = np.eye(4)
    x, y = np.meshgrid(np.arange(-0.6, 0.61, 0.2), np.arange(-0.6, 0.61, 0.2))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(
        np.float32
    )
    cached_variance = np.full(points.shape[0], 0.002, dtype=np.float32)

    def frame(timestamp):
        return SimpleNamespace(
            timestamp=timestamp,
            world_from_body=transform,
            observation_world=SimpleNamespace(
                xyz=points,
                ground_probability=np.ones(points.shape[0], dtype=np.float32),
                dynamic=np.zeros(points.shape[0], dtype=bool),
                height_variance=cached_variance,
            ),
        )

    frames = {"one": frame(1.0), "two": frame(2.0)}
    sequential = _numpy_map(monkeypatch)
    sequential.advance(("one",), frames.__getitem__, target_world_from_body=transform)
    sequential.advance(("two",), frames.__getitem__, target_world_from_body=transform)

    replay = _numpy_map(monkeypatch)
    replay.advance(
        ("one", "two"), frames.__getitem__, target_world_from_body=transform
    )

    np.testing.assert_allclose(replay.elevation_mean, sequential.elevation_mean)
    np.testing.assert_allclose(
        replay.elevation_variance, sequential.elevation_variance
    )
    np.testing.assert_array_equal(
        replay.accepted_frame_count, sequential.accepted_frame_count
    )
    np.testing.assert_allclose(replay.risk_weight, sequential.risk_weight)
    for replay_histogram, sequential_histogram in zip(
        replay.risk_histogram, sequential.risk_histogram
    ):
        np.testing.assert_allclose(replay_histogram, sequential_histogram)


def test_node_duplicate_id_check_precedes_epoch_and_timestamp_reset():
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    process = next(
        member
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoadRunnerMapping"
        for member in node.body
        if isinstance(member, ast.FunctionDef) and member.name == "_process_geometry"
    )
    frame_assignment = next(
        node
        for node in ast.walk(process)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "frame_id" for target in node.targets)
    )
    duplicate_guard = next(
        node
        for node in ast.walk(process)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "frame_id"
        and any(isinstance(operator, ast.In) for operator in node.test.ops)
    )
    reset_lines = sorted(
        node.lineno
        for node in ast.walk(process)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_reset_stream"
    )
    old_epoch_guard = next(
        node
        for node in ast.walk(process)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "epoch"
        and any(isinstance(operator, ast.Lt) for operator in node.test.ops)
    )

    assert old_epoch_guard.lineno < frame_assignment.lineno
    assert frame_assignment.lineno < duplicate_guard.lineno < reset_lines[0]


def test_node_frame_time_gate_ignores_duplicate_before_equal_or_rewind_checks():
    gate = _node_function("_geometry_frame_time_action")

    assert gate("known", 5.0, {"known"}, 10.0) == "duplicate"
    assert gate("new", 9.0, {"known"}, 10.0) == "rewind"
    assert gate("new", 11.0, {"known"}, 10.0) == "accept"
    with pytest.raises(ValueError, match="timestamp"):
        gate("new", 10.0, {"known"}, 10.0)


def test_node_caches_cpu_variance_and_rejects_equal_time_new_id():
    source = NODE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    process = next(
        member
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoadRunnerMapping"
        for member in node.body
        if isinstance(member, ast.FunctionDef)
        and member.name == "_process_geometry_active_epoch"
    )
    calls = [
        node
        for node in ast.walk(process)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_lidar_height_variance"
    ]
    observation_builders = [
        node
        for node in ast.walk(process)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SimpleNamespace"
        and any(keyword.arg == "height_variance" for keyword in node.keywords)
    ]

    assert len(calls) == 1
    assert len(observation_builders) >= 1
    assert "_geometry_frame_time_action" in source


@pytest.mark.parametrize(
    ("strict_window", "timestamp"),
    ((False, 1.1), (True, 3.0)),
)
def test_process_geometry_cache_is_directly_accepted_by_offline_ingestion(
    strict_window,
    timestamp,
):
    node = _process_node(strict_window=strict_window)

    node._process_geometry(0, _cloud_message(timestamp, seq=1))

    cached = node.geometry_by_id[next(reversed(node.geometry_by_id))]
    np.testing.assert_array_equal(cached.world_from_body, np.eye(4))
    np.testing.assert_array_equal(cached.sensor_origin_world, np.zeros(3))
    np.testing.assert_array_equal(
        cached.observation_world.relative_points_lidar,
        np.array([[1.0, 0.0, 0.2]], dtype=np.float32),
    )
    prepared = GpuRollingElevationMap._prepare_offline_frame(None, cached)
    np.testing.assert_allclose(
        prepared[4],
        point_height_variance(
            np,
            np.array([np.sqrt(1.0**2 + 0.2**2)]),
            node.gaussian_config,
        ),
    )


@pytest.mark.parametrize("failure_stage", ("decode", "tf", "map"))
def test_process_geometry_failure_is_atomic_and_same_frame_can_retry(
    failure_stage,
):
    decoder_state = {"fail": failure_stage == "decode"}

    def decoder(message, **kwargs):
        if decoder_state["fail"]:
            raise RuntimeError("injected decode failure")
        return np.array([[1.0, 0.0, 0.2, 10.0]], dtype=np.float32)

    node = _process_node(decoder=decoder)
    node._lookup_fail = failure_stage == "tf"
    node.geometry_map.fail = failure_stage == "map"
    before = _process_state(node)
    message = _cloud_message(1.09, seq=7)

    with pytest.raises(RuntimeError, match="injected"):
        node._process_geometry(0, message)

    _assert_process_state_equal(_process_state(node), before)

    decoder_state["fail"] = False
    node._lookup_fail = False
    node.geometry_map.fail = False
    node._process_geometry(0, message)
    assert node.last_timestamp == pytest.approx(1.09)
    assert "1090000000:7" in node.geometry_ingested_ids


def test_old_epoch_is_discarded_before_timestamp_or_duplicate_gates():
    node = _process_node()
    node.active_epoch = 2
    before = _process_state(node)

    node._process_geometry(1, _cloud_message(1.0, seq=99))

    _assert_process_state_equal(_process_state(node), before)


def test_new_epoch_duplicate_id_is_global_and_does_not_reset():
    node = _process_node()
    old_id = "1000000000:0"
    prior = next(iter(node.geometry_window))
    prior.frame_id = old_id
    node.geometry_by_id = {old_id: prior}
    node.geometry_ingested_ids = {old_id}
    before = _process_state(node)

    node._process_geometry(1, _cloud_message(1.0, seq=0))

    _assert_process_state_equal(_process_state(node), before)


def test_new_epoch_different_id_resets_and_accepts_same_timestamp():
    node = _process_node()
    old_id = "1000000000:0"
    prior = next(iter(node.geometry_window))
    prior.frame_id = old_id
    node.geometry_by_id = {old_id: prior}
    node.geometry_ingested_ids = {old_id}

    node._process_geometry(1, _cloud_message(1.0, seq=99))

    expected_id = "1000000000:99"
    assert node.active_epoch == 1
    assert node.geometry_ingested_ids == {old_id, expected_id}
    assert tuple(frame.frame_id for frame in node.geometry_window) == (expected_id,)

    before_duplicate = _process_state(node)
    node._process_geometry(1, _cloud_message(1.0, seq=0))
    _assert_process_state_equal(_process_state(node), before_duplicate)


def test_new_epoch_failed_fusion_restores_old_stream_and_can_retry():
    node = _process_node()
    before = _process_state(node)
    node.geometry_map.fail = True
    message = _cloud_message(1.0, seq=99)

    with pytest.raises(RuntimeError, match="injected map failure"):
        node._process_geometry(1, message)

    _assert_process_state_equal(_process_state(node), before)
    node.geometry_map.fail = False
    node._process_geometry(1, message)
    assert node.active_epoch == 1
    assert node.geometry_ingested_ids == {"prior", "1000000000:99"}


def _offline_geometry_frame(timestamp, x):
    points = np.array([[x, 0.0, 0.0]], dtype=np.float32)
    return SimpleNamespace(
        frame_id="frame-{}".format(timestamp),
        timestamp=float(timestamp),
        world_from_body=np.eye(4),
        observation_world=SimpleNamespace(
            xyz=points,
            ground_probability=np.ones(1, dtype=np.float32),
            dynamic=np.zeros(1, dtype=bool),
            height_variance=np.full(1, 0.002, dtype=np.float32),
        ),
    )


def test_strict_rebuild_second_frame_failure_restores_full_shared_grid(
    monkeypatch,
):
    geometry_map = _numpy_map(monkeypatch)
    expired = _offline_geometry_frame(1.0, -0.4)
    kept = _offline_geometry_frame(2.0, 0.4)
    cached = {expired.frame_id: expired, kept.frame_id: kept}
    geometry_map.advance(
        (expired.frame_id, kept.frame_id),
        cached.__getitem__,
        target_world_from_body=np.eye(4),
    )
    shared_semantic = np.full(geometry_map.rolling_grid.shape, 77.0, dtype=np.float32)
    geometry_map.rolling_grid.register_layer(shared_semantic, 0.0)
    geometry_map.rolling_grid.move_center(np.array([0.4, -0.2]))
    for index, (layer, _) in enumerate(geometry_map.rolling_grid._layers):
        if layer.dtype.kind == "b":
            layer[...] = bool(index % 2)
        elif layer.dtype.kind in "iu":
            layer[...] = index + 1
        else:
            layer[...] = index + 0.25

    node = _process_node(strict_window=True)
    node.geometry_map = geometry_map
    node.gaussian_config = geometry_map.gaussian_config
    node.geometry_window = RecentFrameWindow(2.0)
    node.geometry_window.append(expired)
    node.geometry_window.append(kept)
    node.geometry_by_id = dict(cached)
    node.geometry_ingested_ids = set(cached)
    node.last_timestamp = 2.0
    geometry_map.height_drift_bias_z = 0.15
    geometry_map.height_drift_last_update = 2.0
    geometry_map.last_lifecycle_stats = {
        "visibility_input_rays": 3,
        "height_drift_bias_z": 0.15,
    }
    before_node = _process_state(node)
    before_layers = tuple(
        layer.copy() for layer, _ in geometry_map.rolling_grid._layers
    )
    before_center = geometry_map.rolling_grid.center
    before_initialized = geometry_map.rolling_grid.initialized
    before_map_metadata = (
        geometry_map._time_origin,
        geometry_map._latest_timestamp,
        geometry_map.ingested_frame_ids,
        geometry_map.ingested_timestamps,
        geometry_map.height_drift_bias_z,
        geometry_map.height_drift_last_update,
        dict(geometry_map.last_lifecycle_stats),
    )
    original_advance = geometry_map.advance

    def fail_second(frame_ids, frame_loader, *, target_world_from_body):
        calls = {"count": 0}

        geometry_map.height_drift_bias_z = 0.9
        geometry_map.height_drift_last_update = 4.0
        geometry_map.last_lifecycle_stats = {
            "visibility_input_rays": 99,
            "height_drift_bias_z": 0.9,
        }

        def failing_loader(frame_id):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("injected second replay failure")
            return frame_loader(frame_id)

        return original_advance(
            frame_ids,
            failing_loader,
            target_world_from_body=target_world_from_body,
        )

    geometry_map.advance = fail_second
    message = _cloud_message(4.0, seq=5)
    with pytest.raises(RuntimeError, match="second replay"):
        node._process_geometry(0, message)

    _assert_process_state_equal(_process_state(node), before_node)
    for current, expected in zip(
        (layer for layer, _ in geometry_map.rolling_grid._layers),
        before_layers,
    ):
        np.testing.assert_array_equal(current, expected)
    np.testing.assert_array_equal(geometry_map.rolling_grid.center, before_center)
    assert geometry_map.rolling_grid.initialized is before_initialized
    assert (
        geometry_map._time_origin,
        geometry_map._latest_timestamp,
        geometry_map.ingested_frame_ids,
        geometry_map.ingested_timestamps,
        geometry_map.height_drift_bias_z,
        geometry_map.height_drift_last_update,
        geometry_map.last_lifecycle_stats,
    ) == before_map_metadata
    np.testing.assert_array_equal(shared_semantic, before_layers[-1])

    geometry_map.advance = original_advance
    node._process_geometry(0, message)
    assert "4000000000:5" in node.geometry_ingested_ids


def test_new_epoch_failure_restores_complete_cpu_semantic_grid():
    node = _process_node()
    semantic_map = IncrementalSemanticGrid(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 0.2),
        semantic_risk={1: 0.8},
        risk_histogram_bins=4,
    )
    semantic_map.update(
        timestamp=2.0,
        world_from_body=np.eye(4),
        xyz_world=np.array([[0.2, 0.1, 0.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.uint8),
        confidence=np.array([0.9], dtype=np.float32),
        valid=np.array([True]),
        dynamic=np.array([True]),
    )
    node.semantic_map = semantic_map
    before = _cpu_semantic_state(semantic_map)
    node.geometry_map.fail = True

    with pytest.raises(RuntimeError, match="injected map failure"):
        node._process_geometry(1, _cloud_message(3.0, seq=8))

    _assert_cpu_semantic_state_equal(_cpu_semantic_state(semantic_map), before)


def test_new_epoch_semantic_clear_failure_restores_cpu_grid():
    node = _process_node()
    semantic_map = IncrementalSemanticGrid(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 0.2),
        semantic_risk={1: 0.8},
        risk_histogram_bins=4,
    )
    semantic_map.update(
        timestamp=2.0,
        world_from_body=np.eye(4),
        xyz_world=np.array([[0.2, 0.1, 0.0]], dtype=np.float32),
        labels=np.array([1], dtype=np.uint8),
        confidence=np.array([0.9], dtype=np.float32),
        valid=np.array([True]),
        dynamic=np.array([False]),
    )
    node.semantic_map = semantic_map
    before_semantic = _cpu_semantic_state(semantic_map)
    before_node = _process_state(node)
    original_clear = semantic_map.clear

    def failing_clear():
        original_clear()
        raise RuntimeError("injected semantic clear failure")

    semantic_map.clear = failing_clear
    with pytest.raises(RuntimeError, match="semantic clear"):
        node._process_geometry(1, _cloud_message(3.0, seq=8))

    _assert_cpu_semantic_state_equal(
        _cpu_semantic_state(semantic_map), before_semantic
    )
    _assert_process_state_equal(_process_state(node), before_node)


def test_strict_transaction_blocks_gpu_semantic_update_until_rollback(
    monkeypatch,
):
    geometry_map = _numpy_map(monkeypatch)
    expired = _offline_geometry_frame(1.0, -0.4)
    kept = _offline_geometry_frame(2.0, 0.4)
    cached = {expired.frame_id: expired, kept.frame_id: kept}
    geometry_map.advance(
        tuple(cached), cached.__getitem__, target_world_from_body=np.eye(4)
    )
    semantic_map = GpuIncrementalSemanticGrid(
        map_spec=geometry_map.map_spec,
        semantic_risk={1: 0.8},
        rolling_grid=geometry_map.rolling_grid,
        stale_sec=90.0,
        risk_histogram_bins=4,
    )
    node = _process_node(strict_window=True)
    node.geometry_map = geometry_map
    node.gaussian_config = geometry_map.gaussian_config
    node.semantic_map = semantic_map
    node.geometry_window = RecentFrameWindow(2.0)
    node.geometry_window.append(expired)
    node.geometry_window.append(kept)
    node.geometry_by_id = dict(cached)
    node.geometry_ingested_ids = set(cached)
    node.last_timestamp = 2.0

    replay_entered = threading.Event()
    allow_failure = threading.Event()
    update_attempted = threading.Event()
    update_acquired = threading.Event()
    errors = []

    def blocked_failure(*args, **kwargs):
        replay_entered.set()
        if not allow_failure.wait(timeout=2.0):
            raise AssertionError("test did not release strict replay")
        raise RuntimeError("injected strict replay failure")

    geometry_map.advance = blocked_failure

    def run_geometry():
        try:
            node._process_geometry(0, _cloud_message(4.0, seq=5))
        except RuntimeError as error:
            errors.append(error)

    def update_semantics():
        update_attempted.set()
        with node.semantic_map_lock:
            update_acquired.set()
            semantic_map.update(
                timestamp=10.0,
                world_from_body=np.eye(4),
                xyz_world=np.array([[0.2, 0.1, 0.0]], dtype=np.float32),
                labels=np.array([1], dtype=np.uint8),
                confidence=np.array([1.0], dtype=np.float32),
                valid=np.array([True]),
                dynamic=np.array([False]),
            )

    geometry_thread = threading.Thread(target=run_geometry)
    semantic_thread = threading.Thread(target=update_semantics)
    geometry_thread.start()
    assert replay_entered.wait(timeout=2.0)
    semantic_thread.start()
    assert update_attempted.wait(timeout=2.0)
    was_blocked = not update_acquired.wait(timeout=0.1)

    allow_failure.set()
    geometry_thread.join(timeout=2.0)
    semantic_thread.join(timeout=2.0)
    assert not geometry_thread.is_alive()
    assert not semantic_thread.is_alive()
    assert was_blocked
    assert len(errors) == 1
    assert update_acquired.is_set()
    assert semantic_map._last_timestamp == pytest.approx(10.0)
    assert int(semantic_map._last_input_points) == 1
    assert float(np.sum(semantic_map._weight_sum)) > 0.0
