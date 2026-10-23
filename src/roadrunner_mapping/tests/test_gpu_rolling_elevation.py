import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from roadrunner_mapping.geometry.gaussian_elevation import (
    GaussianElevationConfig,
)
from roadrunner_mapping.geometry.elevation_map_lifecycle import (
    ElevationMapLifecycleConfig,
)


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


class _NumpyCupy:
    ndarray = np.ndarray
    cuda = SimpleNamespace(Device=lambda index: _NumpyDevice())

    def __init__(self):
        self._pool = _NumpyPool()

    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, values):
        return np.asarray(values)

    def get_default_memory_pool(self):
        return self._pool


@pytest.fixture
def numpy_map(monkeypatch):
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
    )


@pytest.fixture
def lifecycle_numpy_map(monkeypatch):
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
        lifecycle_config=ElevationMapLifecycleConfig(enabled=True),
    )


def _new_numpy_map(monkeypatch, *, lifecycle_config=None):
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
        lifecycle_config=lifecycle_config,
    )


def _dense_points(height=0.0):
    x, y = np.meshgrid(np.arange(-1.5, 1.5, 0.1), np.arange(-1.5, 1.5, 0.1))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height))).astype(
        np.float32
    )


def _advance_points(grid, frame_id, timestamp, points):
    return grid.advance_pointcloud(
        frame_id,
        timestamp,
        points,
        np.zeros(len(points), dtype=np.float32),
        world_from_lidar=np.eye(4),
        minimum_range=0.0,
        maximum_range=10.0,
        target_world_from_body=np.eye(4),
    )


_GEOMETRY_TOP_LEVEL_TIMING_FIELDS = {
    "geometry_prepare_sec",
    "geometry_drift_sec",
    "geometry_visibility_sec",
    "geometry_map_update_sec",
    "geometry_world_fields_sec",
    "geometry_risk_sec",
    "geometry_sample_sec",
    "geometry_transfer_sec",
}
_GEOMETRY_MAP_TIMING_FIELDS = {
    "geometry_map_center_sec",
    "geometry_map_maintenance_sec",
    "geometry_map_backup_sec",
    "geometry_map_clear_sec",
    "geometry_map_visibility_apply_sec",
    "geometry_map_fusion_sec",
    "geometry_map_commit_sec",
}
_GEOMETRY_TIMING_FIELDS = (
    _GEOMETRY_TOP_LEVEL_TIMING_FIELDS | _GEOMETRY_MAP_TIMING_FIELDS
)


class _RetryableTimerDevice:
    def __init__(self):
        self.active = False

    def __enter__(self):
        assert not self.active
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False
        return False


class _RetryableTimerEvent:
    def __init__(self, cuda, index):
        self.cuda = cuda
        self.index = index

    def record(self, stream):
        assert self.cuda.device.active
        assert stream is self.cuda.stream
        self.cuda.recorded.append(self.index)

    def synchronize(self):
        assert self.cuda.device.active
        self.cuda.synchronized.append(self.index)
        if self.cuda.fail_synchronize:
            raise RuntimeError("synchronize failed")


class _RetryableTimerCuda:
    def __init__(self, device, elapsed_ms):
        self.device = device
        self.elapsed_ms = tuple(elapsed_ms)
        self.stream = object()
        self.events = []
        self.recorded = []
        self.synchronized = []
        self.elapsed_queries = []
        self.fail_elapsed_index = None
        self.fail_synchronize = False

    def Event(self):
        assert self.device.active
        event = _RetryableTimerEvent(self, len(self.events))
        self.events.append(event)
        return event

    def get_current_stream(self):
        assert self.device.active
        return self.stream

    def get_elapsed_time(self, started, ended):
        assert self.device.active
        query_index = started.index // 2
        self.elapsed_queries.append((started.index, ended.index))
        if query_index == self.fail_elapsed_index:
            raise RuntimeError("elapsed query failed")
        return self.elapsed_ms[query_index]


def _new_retryable_cuda_timer(module, elapsed_ms):
    device = _RetryableTimerDevice()
    cuda = _RetryableTimerCuda(device, elapsed_ms)
    timer = module._GeometryStageTimer(
        cp=SimpleNamespace(cuda=cuda),
        device=device,
    )
    return timer, cuda


def test_geometry_stage_timer_perf_counter_accumulates_fixed_metrics(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    readings = iter((10.0, 10.25, 11.0, 11.5, 12.0, 12.75))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(readings))
    timer = module._GeometryStageTimer(
        cp=SimpleNamespace(cuda=SimpleNamespace()),
        device=_NumpyDevice(),
    )

    with timer.segment("geometry_prepare_sec"):
        pass
    with timer.segment("geometry_drift_sec"):
        pass
    with timer.segment("geometry_prepare_sec"):
        pass
    metrics = timer.finish()

    assert set(metrics) == _GEOMETRY_TIMING_FIELDS
    assert metrics["geometry_prepare_sec"] == pytest.approx(1.0)
    assert metrics["geometry_drift_sec"] == pytest.approx(0.5)
    assert all(
        value == 0.0
        for name, value in metrics.items()
        if name not in {"geometry_prepare_sec", "geometry_drift_sec"}
    )


def test_geometry_stage_timer_perf_counter_returns_finite_nonnegative_values(
    monkeypatch,
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    readings = iter((2.0, 1.0, 3.0, float("inf")))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(readings))
    timer = module._GeometryStageTimer(
        cp=SimpleNamespace(cuda=SimpleNamespace()),
        device=_NumpyDevice(),
    )

    with timer.segment("geometry_prepare_sec"):
        pass
    with timer.segment("geometry_drift_sec"):
        pass
    metrics = timer.finish()

    assert all(np.isfinite(value) and value >= 0.0 for value in metrics.values())
    assert metrics["geometry_prepare_sec"] == 0.0
    assert metrics["geometry_drift_sec"] == 0.0


def test_geometry_stage_timer_derives_map_update_and_rejects_direct_segment(
    monkeypatch,
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    readings = iter((1.0, 1.25, 2.0, 2.5))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(readings))
    timer = module._GeometryStageTimer(
        cp=SimpleNamespace(cuda=SimpleNamespace()),
        device=_NumpyDevice(),
    )

    with pytest.raises(ValueError, match="aggregate geometry timing field"):
        timer.start("geometry_map_update_sec")
    with timer.segment("geometry_map_center_sec"):
        pass
    with timer.segment("geometry_map_maintenance_sec"):
        pass

    metrics = timer.finish()

    assert metrics["geometry_map_center_sec"] == pytest.approx(0.25)
    assert metrics["geometry_map_maintenance_sec"] == pytest.approx(0.5)
    assert metrics["geometry_map_update_sec"] == pytest.approx(0.75)


def test_geometry_stage_timer_map_update_aggregate_saturates(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    readings = iter((0.0, 1e308, 0.0, 1e308))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(readings))
    timer = module._GeometryStageTimer(
        cp=SimpleNamespace(cuda=SimpleNamespace()),
        device=_NumpyDevice(),
    )

    with timer.segment("geometry_map_center_sec"):
        pass
    with timer.segment("geometry_map_fusion_sec"):
        pass

    duration = timer.finish()["geometry_map_update_sec"]
    assert np.isfinite(duration)
    assert duration == sys.float_info.max
    assert duration >= 1e308


def test_geometry_stage_timer_keeps_repeated_overflow_finite(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    readings = iter((0.0, 1e308, 0.0, 1e308))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(readings))
    timer = module._GeometryStageTimer(
        cp=SimpleNamespace(cuda=SimpleNamespace()),
        device=_NumpyDevice(),
    )

    with timer.segment("geometry_prepare_sec"):
        pass
    with timer.segment("geometry_prepare_sec"):
        pass

    duration = timer.finish()["geometry_prepare_sec"]
    assert np.isfinite(duration)
    assert duration == sys.float_info.max
    assert duration >= 1e308


def test_geometry_stage_timer_rejects_nested_segments(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    readings = iter((1.0, 2.0))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(readings))
    timer = module._GeometryStageTimer(
        cp=SimpleNamespace(cuda=SimpleNamespace()),
        device=_NumpyDevice(),
    )

    with pytest.raises(RuntimeError, match="nested geometry timing segments"):
        with timer.segment("geometry_prepare_sec"):
            with timer.segment("geometry_drift_sec"):
                pass


def test_geometry_stage_timer_cuda_events_share_device_stream_and_sync_once():
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    operations = []
    stream = object()

    class FakeDevice:
        active = False

        def __enter__(self):
            assert not self.active
            self.active = True
            operations.append(("device_enter",))
            return self

        def __exit__(self, *args):
            operations.append(("device_exit",))
            self.active = False
            return False

    device = FakeDevice()

    class FakeEvent:
        def __init__(self):
            assert device.active
            self.index = len([op for op in operations if op[0] == "event"])
            operations.append(("event", self.index))

        def record(self, selected_stream):
            assert device.active
            operations.append(("record", self.index, selected_stream))

        def synchronize(self):
            assert device.active
            operations.append(("synchronize", self.index))

    elapsed_ms = {(0, 1): 125.0, (2, 3): 250.0, (4, 5): 375.0}

    def get_current_stream():
        assert device.active
        operations.append(("get_current_stream", stream))
        return stream

    def get_elapsed_time(start, end):
        assert device.active
        assert any(op[0] == "synchronize" for op in operations)
        operations.append(("elapsed", start.index, end.index))
        return elapsed_ms[(start.index, end.index)]

    cuda = SimpleNamespace(
        Event=FakeEvent,
        get_current_stream=get_current_stream,
        get_elapsed_time=get_elapsed_time,
    )
    timer = module._GeometryStageTimer(
        cp=SimpleNamespace(cuda=cuda),
        device=device,
    )

    with timer.segment("geometry_prepare_sec"):
        pass
    with timer.segment("geometry_drift_sec"):
        pass
    with timer.segment("geometry_prepare_sec"):
        pass
    metrics = timer.finish()

    assert metrics["geometry_prepare_sec"] == pytest.approx(0.5)
    assert metrics["geometry_drift_sec"] == pytest.approx(0.25)
    assert [op[1:] for op in operations if op[0] == "record"] == [
        (index, stream) for index in range(6)
    ]
    assert [op[1] for op in operations if op[0] == "synchronize"] == [5]
    synchronize_position = next(
        index for index, op in enumerate(operations) if op[0] == "synchronize"
    )
    assert all(
        index > synchronize_position
        for index, op in enumerate(operations)
        if op[0] == "elapsed"
    )
    assert len([op for op in operations if op[0] == "get_current_stream"]) == 1


def test_geometry_stage_timer_cuda_elapsed_failure_is_atomically_retryable():
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    timer, cuda = _new_retryable_cuda_timer(module, (100.0, 200.0))
    with timer.segment("geometry_prepare_sec"):
        pass
    with timer.segment("geometry_drift_sec"):
        pass
    cuda.fail_elapsed_index = 1

    with pytest.raises(RuntimeError, match="elapsed query failed"):
        timer.finish()

    assert all(duration == 0.0 for duration in timer._durations.values())
    assert len(timer._event_segments) == 2
    cuda.fail_elapsed_index = None
    metrics = timer.finish()
    assert metrics["geometry_prepare_sec"] == pytest.approx(0.1)
    assert metrics["geometry_drift_sec"] == pytest.approx(0.2)
    assert timer.finish() == metrics
    assert cuda.elapsed_queries == [(0, 1), (2, 3), (0, 1), (2, 3)]
    assert cuda.synchronized == [3, 3]


def test_geometry_stage_timer_cuda_sync_failure_is_atomically_retryable():
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    timer, cuda = _new_retryable_cuda_timer(module, (100.0, 200.0))
    with timer.segment("geometry_prepare_sec"):
        pass
    with timer.segment("geometry_drift_sec"):
        pass
    cuda.fail_synchronize = True

    with pytest.raises(RuntimeError, match="synchronize failed"):
        timer.finish()

    assert all(duration == 0.0 for duration in timer._durations.values())
    assert len(timer._event_segments) == 2
    assert cuda.elapsed_queries == []
    cuda.fail_synchronize = False
    metrics = timer.finish()
    assert metrics["geometry_prepare_sec"] == pytest.approx(0.1)
    assert metrics["geometry_drift_sec"] == pytest.approx(0.2)
    assert timer.finish() == metrics
    assert cuda.elapsed_queries == [(0, 1), (2, 3)]
    assert cuda.synchronized == [3, 3]


def test_geometry_stage_timer_cuda_aggregate_failure_is_atomically_retryable():
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    timer, cuda = _new_retryable_cuda_timer(module, (100.0, 200.0))
    with timer.segment("geometry_map_center_sec"):
        pass
    with timer.segment("geometry_map_maintenance_sec"):
        pass
    aggregate = timer._map_update_duration

    def fail_aggregate(durations):
        raise RuntimeError("aggregate failed")

    timer._map_update_duration = fail_aggregate
    with pytest.raises(RuntimeError, match="aggregate failed"):
        timer.finish()

    assert all(duration == 0.0 for duration in timer._durations.values())
    assert len(timer._event_segments) == 2
    assert cuda.elapsed_queries == [(0, 1), (2, 3)]

    timer._map_update_duration = aggregate
    metrics = timer.finish()
    assert metrics["geometry_map_center_sec"] == pytest.approx(0.1)
    assert metrics["geometry_map_maintenance_sec"] == pytest.approx(0.2)
    assert metrics["geometry_map_update_sec"] == pytest.approx(0.3)
    assert cuda.elapsed_queries == [(0, 1), (2, 3), (0, 1), (2, 3)]
    assert cuda.synchronized == [3, 3]


class _RecordingGeometryTimer:
    instances = []

    def __init__(self, *, cp, device):
        self.cp = cp
        self.device = device
        self.active = None
        self.segments = []
        type(self).instances.append(self)

    class _Segment:
        def __init__(self, timer, name):
            self.timer = timer
            self.name = name

        def __enter__(self):
            self.timer.start(self.name)

        def __exit__(self, *args):
            self.timer.stop()
            return False

    def start(self, name):
        assert self.active is None
        self.active = name
        self.segments.append(name)

    def stop(self):
        assert self.active is not None
        self.active = None

    def segment(self, name):
        assert name in _GEOMETRY_TIMING_FIELDS
        return self._Segment(self, name)

    def finish(self):
        assert self.active is None
        timing = {
            name: float(self.segments.count(name))
            for name in _GEOMETRY_TIMING_FIELDS
        }
        timing["geometry_map_update_sec"] = sum(
            timing[name] for name in _GEOMETRY_MAP_TIMING_FIELDS
        )
        return timing


def _record_map_timing_segments(monkeypatch, grid, frame_id, points):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    _RecordingGeometryTimer.instances.clear()
    monkeypatch.setattr(module, "_GeometryStageTimer", _RecordingGeometryTimer)
    _advance_points(grid, frame_id, 1.0, points)
    timer = _RecordingGeometryTimer.instances[-1]
    return [
        name for name in timer.segments if name in _GEOMETRY_MAP_TIMING_FIELDS
    ]


def test_baseline_map_update_records_nonempty_leaf_sequence(
    numpy_map, monkeypatch
):
    segments = _record_map_timing_segments(
        monkeypatch, numpy_map, "baseline", _dense_points()
    )

    assert segments == [
        "geometry_map_center_sec",
        "geometry_map_backup_sec",
        "geometry_map_maintenance_sec",
        "geometry_map_clear_sec",
        "geometry_map_fusion_sec",
        "geometry_map_commit_sec",
    ]


def test_baseline_map_update_empty_frame_records_only_executed_leaves(
    numpy_map, monkeypatch
):
    points = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)

    segments = _record_map_timing_segments(
        monkeypatch, numpy_map, "outside", points
    )

    assert segments == [
        "geometry_map_center_sec",
        "geometry_map_backup_sec",
        "geometry_map_commit_sec",
    ]


def test_lifecycle_map_update_records_repeated_maintenance_and_clear_sequence(
    lifecycle_numpy_map, monkeypatch
):
    segments = _record_map_timing_segments(
        monkeypatch, lifecycle_numpy_map, "lifecycle", _dense_points()
    )

    assert segments == [
        "geometry_map_center_sec",
        "geometry_map_maintenance_sec",
        "geometry_map_backup_sec",
        "geometry_map_clear_sec",
        "geometry_map_visibility_apply_sec",
        "geometry_map_maintenance_sec",
        "geometry_map_clear_sec",
        "geometry_map_fusion_sec",
        "geometry_map_commit_sec",
    ]


def test_advance_pointcloud_commits_complete_baseline_geometry_timing(numpy_map):
    assert set(numpy_map.last_geometry_timing) == _GEOMETRY_TIMING_FIELDS
    assert all(value == 0.0 for value in numpy_map.last_geometry_timing.values())

    _advance_points(numpy_map, "timed", 1.0, _dense_points())

    timing = numpy_map.last_geometry_timing
    assert set(timing) == _GEOMETRY_TIMING_FIELDS
    assert all(
        type(value) is float and np.isfinite(value) and value >= 0.0
        for value in timing.values()
    )
    assert timing["geometry_drift_sec"] == 0.0
    assert timing["geometry_visibility_sec"] == 0.0
    for name in (
        "geometry_prepare_sec",
        "geometry_map_update_sec",
        "geometry_world_fields_sec",
        "geometry_risk_sec",
        "geometry_sample_sec",
        "geometry_transfer_sec",
    ):
        assert timing[name] > 0.0
    assert timing["geometry_map_update_sec"] == pytest.approx(
        sum(timing[name] for name in _GEOMETRY_MAP_TIMING_FIELDS)
    )


def test_lifecycle_advance_pointcloud_times_drift_and_visibility(
    lifecycle_numpy_map,
):
    _advance_points(lifecycle_numpy_map, "timed", 1.0, _dense_points())

    timing = lifecycle_numpy_map.last_geometry_timing
    assert set(timing) == _GEOMETRY_TIMING_FIELDS
    assert timing["geometry_drift_sec"] > 0.0
    assert timing["geometry_visibility_sec"] > 0.0
    assert timing["geometry_map_update_sec"] == pytest.approx(
        sum(timing[name] for name in _GEOMETRY_MAP_TIMING_FIELDS)
    )


def test_offline_advance_accumulates_nonoverlapping_segments_across_frames(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    _RecordingGeometryTimer.instances.clear()
    monkeypatch.setattr(module, "_GeometryStageTimer", _RecordingGeometryTimer)
    frames = {
        "first": _offline_test_frame(1.0),
        "second": _offline_test_frame(2.0, 0.01),
    }

    numpy_map.advance(
        tuple(frames),
        frames.__getitem__,
        target_world_from_body=np.eye(4),
    )

    assert len(_RecordingGeometryTimer.instances) == 1
    timer = _RecordingGeometryTimer.instances[0]
    assert timer.cp is numpy_map.cp
    assert timer.device is numpy_map.device
    assert numpy_map.last_geometry_timing == {
        "geometry_prepare_sec": 6.0,
        "geometry_drift_sec": 0.0,
        "geometry_visibility_sec": 0.0,
        "geometry_map_update_sec": 12.0,
        "geometry_world_fields_sec": 2.0,
        "geometry_risk_sec": 3.0,
        "geometry_sample_sec": 1.0,
        "geometry_transfer_sec": 1.0,
        "geometry_map_center_sec": 2.0,
        "geometry_map_maintenance_sec": 2.0,
        "geometry_map_backup_sec": 2.0,
        "geometry_map_clear_sec": 2.0,
        "geometry_map_visibility_apply_sec": 0.0,
        "geometry_map_fusion_sec": 2.0,
        "geometry_map_commit_sec": 2.0,
    }


def test_duplicate_render_commits_a_complete_new_timing_snapshot(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    _advance_points(numpy_map, "first", 1.0, _dense_points())
    _RecordingGeometryTimer.instances.clear()
    monkeypatch.setattr(module, "_GeometryStageTimer", _RecordingGeometryTimer)

    _advance_points(numpy_map, "first", 1.0, _dense_points())

    assert len(_RecordingGeometryTimer.instances) == 1
    assert numpy_map.last_geometry_timing == {
        "geometry_prepare_sec": 0.0,
        "geometry_drift_sec": 0.0,
        "geometry_visibility_sec": 0.0,
        "geometry_map_update_sec": 0.0,
        "geometry_world_fields_sec": 1.0,
        "geometry_risk_sec": 1.0,
        "geometry_sample_sec": 1.0,
        "geometry_transfer_sec": 1.0,
        "geometry_map_center_sec": 0.0,
        "geometry_map_maintenance_sec": 0.0,
        "geometry_map_backup_sec": 0.0,
        "geometry_map_clear_sec": 0.0,
        "geometry_map_visibility_apply_sec": 0.0,
        "geometry_map_fusion_sec": 0.0,
        "geometry_map_commit_sec": 0.0,
    }


def test_clear_resets_last_geometry_timing(numpy_map):
    _advance_points(numpy_map, "first", 1.0, _dense_points())
    committed = numpy_map.last_geometry_timing

    numpy_map.clear()

    assert numpy_map.last_geometry_timing is not committed
    assert set(numpy_map.last_geometry_timing) == _GEOMETRY_TIMING_FIELDS
    assert all(value == 0.0 for value in numpy_map.last_geometry_timing.values())


def test_late_stage_failure_keeps_previous_geometry_timing_identity(
    numpy_map, monkeypatch
):
    _advance_points(numpy_map, "first", 1.0, _dense_points())
    previous = numpy_map.last_geometry_timing
    previous_values = dict(previous)

    def fail_sample(*args, **kwargs):
        raise RuntimeError("synthetic late geometry failure")

    monkeypatch.setattr(numpy_map, "_sample_current_body", fail_sample)
    with pytest.raises(RuntimeError, match="late geometry failure"):
        _advance_points(numpy_map, "second", 2.0, _dense_points(0.01))

    assert numpy_map.last_geometry_timing is previous
    assert numpy_map.last_geometry_timing == previous_values


def test_partial_result_transfer_keeps_previous_geometry_timing_identity(
    numpy_map, monkeypatch
):
    _advance_points(numpy_map, "first", 1.0, _dense_points())
    previous = numpy_map.last_geometry_timing
    previous_values = dict(previous)
    original_asnumpy = numpy_map.cp.asnumpy
    transfer_count = 0

    def fail_mid_transfer(values):
        nonlocal transfer_count
        transfer_count += 1
        if transfer_count == 3:
            raise RuntimeError("synthetic partial transfer failure")
        return original_asnumpy(values)

    monkeypatch.setattr(numpy_map.cp, "asnumpy", fail_mid_transfer)
    with pytest.raises(RuntimeError, match="partial transfer failure"):
        _advance_points(numpy_map, "first", 1.0, _dense_points())

    assert transfer_count == 3
    assert numpy_map.last_geometry_timing is previous
    assert numpy_map.last_geometry_timing == previous_values


def test_timer_finish_failure_keeps_previous_geometry_timing_identity(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    _advance_points(numpy_map, "first", 1.0, _dense_points())
    previous = numpy_map.last_geometry_timing
    previous_values = dict(previous)

    class _FailingFinishTimer(_RecordingGeometryTimer):
        def finish(self):
            raise RuntimeError("synthetic timer finish failure")

    monkeypatch.setattr(module, "_GeometryStageTimer", _FailingFinishTimer)
    with pytest.raises(RuntimeError, match="timer finish failure"):
        _advance_points(numpy_map, "first", 1.0, _dense_points())

    assert numpy_map.last_geometry_timing is previous
    assert numpy_map.last_geometry_timing == previous_values


def test_geometry_risk_distribution_ignores_invalid_sparse_samples():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        _update_risk_distribution,
    )

    risk_sum = np.zeros((1, 1), dtype=np.float32)
    risk_sq_sum = np.zeros((1, 1), dtype=np.float32)
    risk_weight = np.zeros((1, 1), dtype=np.float32)
    risk_last_update = np.full((1, 1), -np.inf, dtype=np.float64)
    risk_histogram = [
        np.zeros((1, 1), dtype=np.float32) for _ in range(4)
    ]

    _update_risk_distribution(
        cp=np,
        risk_sum=risk_sum,
        risk_sq_sum=risk_sq_sum,
        risk_weight=risk_weight,
        risk_histogram=risk_histogram,
        risk_last_update=risk_last_update,
        risk=np.asarray([[0.9]], dtype=np.float32),
        confidence=np.asarray([[0.2]], dtype=np.float32),
        valid=np.asarray([[False]]),
        timestamp=1.0,
        decay_tau_sec=10.0,
    )

    np.testing.assert_allclose(risk_weight, 0.0)
    np.testing.assert_allclose(risk_sum, 0.0)
    np.testing.assert_allclose(risk_histogram[3], 0.0)

    _update_risk_distribution(
        cp=np,
        risk_sum=risk_sum,
        risk_sq_sum=risk_sq_sum,
        risk_weight=risk_weight,
        risk_histogram=risk_histogram,
        risk_last_update=risk_last_update,
        risk=np.asarray([[0.9]], dtype=np.float32),
        confidence=np.asarray([[0.2]], dtype=np.float32),
        valid=np.asarray([[True]]),
        timestamp=1.0,
        decay_tau_sec=10.0,
    )

    np.testing.assert_allclose(risk_weight, 0.2, atol=1e-6)
    np.testing.assert_allclose(risk_sum, 0.18, atol=1e-6)
    np.testing.assert_allclose(risk_sq_sum, 0.162, atol=1e-6)
    np.testing.assert_allclose(risk_histogram[3], 0.2, atol=1e-6)


def test_empty_risk_mask_uses_device_empty_operations_without_any_reduction():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        _update_risk_distribution,
    )

    class _NoAnyNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def any(self, values):
            raise AssertionError('risk update must not synchronize through cp.any')

    cp = _NoAnyNumpy()
    risk_sum = np.array([[0.3]], dtype=np.float32)
    risk_sq_sum = np.array([[0.2]], dtype=np.float32)
    risk_weight = np.array([[0.5]], dtype=np.float32)
    risk_last_update = np.array([[2.0]], dtype=np.float64)
    histogram = [np.array([[0.1]], dtype=np.float32) for _ in range(4)]
    before = tuple(
        values.copy()
        for values in (risk_sum, risk_sq_sum, risk_weight, risk_last_update, *histogram)
    )
    _update_risk_distribution(
        cp=cp,
        risk_sum=risk_sum,
        risk_sq_sum=risk_sq_sum,
        risk_weight=risk_weight,
        risk_histogram=histogram,
        risk_last_update=risk_last_update,
        risk=np.array([[0.9]], dtype=np.float32),
        confidence=np.array([[0.8]], dtype=np.float32),
        valid=np.array([[False]]),
        timestamp=3.0,
        decay_tau_sec=10.0,
    )
    for actual, expected in zip(
        (risk_sum, risk_sq_sum, risk_weight, risk_last_update, *histogram), before
    ):
        np.testing.assert_array_equal(actual, expected)


def test_geometry_risk_distribution_reports_mean_std_and_upper_tail_cvar():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        _compute_risk_distribution_fields,
    )

    fields = _compute_risk_distribution_fields(
        cp=np,
        risk_sum=np.asarray([[0.9]], dtype=np.float32),
        risk_sq_sum=np.asarray([[0.65]], dtype=np.float32),
        risk_weight=np.asarray([[2.0]], dtype=np.float32),
        risk_histogram=[
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([[0.0]], dtype=np.float32),
            np.asarray([[1.0]], dtype=np.float32),
        ],
        cvar_alpha=0.5,
    )

    np.testing.assert_allclose(fields["geometry_risk_mean"], 0.45)
    np.testing.assert_allclose(fields["geometry_risk_std"], 0.35)
    # Histogram CVaR uses the upper edge of the occupied tail bin, preserving
    # the intended conservative interpretation.
    np.testing.assert_allclose(fields["geometry_risk_cvar"], 1.0)
    np.testing.assert_array_equal(
        fields["geometry_risk_distribution_valid"], [[True]]
    )


def test_world_geometry_fields_are_computed_without_a_body_pose():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        _compute_world_geometry_fields,
    )

    shape = (15, 15)
    count = np.ones(shape, dtype=np.int32)
    elevation = np.zeros(shape, dtype=np.float32)
    variance = np.full(shape, 0.0004, dtype=np.float32)
    ground_sum = np.ones(shape, dtype=np.float32)
    last_seen = np.zeros(shape, dtype=np.float32)

    fields = _compute_world_geometry_fields(
        cp=np,
        ndimage=ndimage,
        elevation_mean=elevation,
        elevation_variance=variance,
        elevation_valid=np.ones(shape, dtype=bool),
        geometry_support_count=count,
        accepted_frame_count=np.full(shape, 3, dtype=np.int32),
        ground_sum=ground_sum,
        min_z=elevation.copy(),
        max_z=elevation.copy(),
        last_seen=last_seen,
        variance_updated_at=last_seen.copy(),
        conflict_score=np.zeros(shape, dtype=np.float32),
        conflict_updated_at=last_seen.copy(),
        latest_timestamp=0.0,
        time_origin=0.0,
        max_geometry_age_sec=120.0,
        resolution=0.2,
        gaussian_config=GaussianElevationConfig(),
    )

    assert fields["geometry_risk"].shape == shape
    assert fields["geometry_valid_mask"].all()
    np.testing.assert_allclose(fields["geometry_risk"], 0.0, atol=1e-6)
    np.testing.assert_allclose(fields["elevation_variance"], variance)
    np.testing.assert_allclose(fields["elevation_std"], 0.02)
    np.testing.assert_allclose(fields["uncertainty_risk"], 0.0)


def test_lightweight_world_slope_matches_full_geometry_slope():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        _compute_world_geometry_fields,
        _compute_world_slope_fields,
    )

    shape = (15, 15)
    rows, columns = np.indices(shape, dtype=np.float32)
    elevation = 0.03 * rows - 0.02 * columns
    elevation[6:9, 6:9] += 0.08
    elevation_valid = np.ones(shape, dtype=bool)
    elevation_valid[7, 7] = False
    last_seen = np.full(shape, 9.0, dtype=np.float32)
    last_seen[2, 2] = 0.0
    common = {
        "cp": np,
        "ndimage": ndimage,
        "elevation_mean": elevation,
        "elevation_valid": elevation_valid,
        "last_seen": last_seen,
        "latest_timestamp": 10.0,
        "time_origin": 0.0,
        "max_geometry_age_sec": 5.0,
        "resolution": 0.2,
    }

    slope_fields = _compute_world_slope_fields(**common)
    full_fields = _compute_world_geometry_fields(
        **common,
        elevation_variance=np.full(shape, 0.0004, dtype=np.float32),
        geometry_support_count=elevation_valid.astype(np.int32),
        accepted_frame_count=np.full(shape, 3, dtype=np.int32),
        ground_sum=elevation_valid.astype(np.float32),
        min_z=elevation.copy(),
        max_z=elevation.copy(),
        variance_updated_at=last_seen.copy(),
        conflict_score=np.zeros(shape, dtype=np.float32),
        conflict_updated_at=last_seen.copy(),
        gaussian_config=GaussianElevationConfig(),
    )

    np.testing.assert_allclose(
        slope_fields["slope"], full_fields["slope"], rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    ("upper_height", "expect_negative_risk"),
    ((-0.5, True), (0.5, False)),
)
def test_pure_upper_bound_only_changes_adjacent_real_negative_step_risk(
    upper_height,
    expect_negative_risk,
):
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        _compute_world_geometry_fields,
    )

    shape = (15, 15)
    observed = np.ones(shape, dtype=bool)
    upper_cell = (7, 8)
    center = (7, 7)
    observed[upper_cell] = False
    upper_valid = np.zeros(shape, dtype=bool)
    upper_valid[upper_cell] = True
    upper = np.zeros(shape, dtype=np.float32)
    upper[upper_cell] = upper_height
    fields = _compute_world_geometry_fields(
        cp=np,
        ndimage=ndimage,
        elevation_mean=np.zeros(shape, dtype=np.float32),
        elevation_variance=np.full(shape, 0.0004, dtype=np.float32),
        elevation_valid=observed,
        elevation_validity=observed.astype(np.float32),
        upper_bound=upper,
        upper_bound_valid=upper_valid,
        height_drift_bias_z=0.12,
        geometry_support_count=observed.astype(np.int32),
        accepted_frame_count=np.full(shape, 3, dtype=np.int32),
        ground_sum=observed.astype(np.float32),
        min_z=np.zeros(shape, dtype=np.float32),
        max_z=np.zeros(shape, dtype=np.float32),
        last_seen=np.zeros(shape, dtype=np.float32),
        variance_updated_at=np.zeros(shape, dtype=np.float32),
        conflict_score=np.zeros(shape, dtype=np.float32),
        conflict_updated_at=np.zeros(shape, dtype=np.float32),
        latest_timestamp=0.0,
        time_origin=0.0,
        max_geometry_age_sec=120.0,
        resolution=0.2,
        gaussian_config=GaussianElevationConfig(),
    )

    if expect_negative_risk:
        assert fields["negative_step_risk"][center] > 0.9
    else:
        assert fields["negative_step_risk"][center] == pytest.approx(0.0)
    assert fields["slope_risk"][center] == pytest.approx(0.0)
    assert fields["positive_step_risk"][center] == pytest.approx(0.0)
    assert fields["roughness_risk"][center] == pytest.approx(0.0)
    assert not fields["geometry_valid_mask"][upper_cell]
    assert fields["geometry_confidence"][upper_cell] == pytest.approx(0.0)
    assert fields["elevation_upper_bound_valid"][upper_cell]
    assert fields["elevation_upper_bound"][upper_cell] == pytest.approx(
        upper_height
    )
    np.testing.assert_allclose(fields["height_drift_bias_z"], 0.12)


def test_render_prediction_is_read_only_and_increases_uncertainty():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        _compute_world_geometry_fields,
    )

    shape = (15, 15)
    stored_variance = np.full(shape, 0.005, dtype=np.float32)
    before = stored_variance.copy()
    fields = _compute_world_geometry_fields(
        cp=np,
        ndimage=ndimage,
        elevation_mean=np.zeros(shape, dtype=np.float32),
        elevation_variance=stored_variance,
        elevation_valid=np.ones(shape, dtype=bool),
        geometry_support_count=np.ones(shape, dtype=np.int32),
        accepted_frame_count=np.ones(shape, dtype=np.int32),
        ground_sum=np.ones(shape, dtype=np.float32),
        min_z=np.zeros(shape, dtype=np.float32),
        max_z=np.zeros(shape, dtype=np.float32),
        last_seen=np.zeros(shape, dtype=np.float32),
        variance_updated_at=np.zeros(shape, dtype=np.float32),
        conflict_score=np.zeros(shape, dtype=np.float32),
        conflict_updated_at=np.zeros(shape, dtype=np.float32),
        latest_timestamp=2.0,
        time_origin=0.0,
        max_geometry_age_sec=120.0,
        resolution=0.2,
        gaussian_config=GaussianElevationConfig(process_variance_per_sec=0.001),
    )
    np.testing.assert_array_equal(stored_variance, before)
    np.testing.assert_allclose(fields["elevation_variance"], 0.007)


def test_body_sampling_preserves_a_pose_independent_world_risk_layer():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        GpuRollingElevationMap,
    )
    from roadrunner_mapping.geometry.schema import MapSpec

    class _Grid:
        center = np.zeros(2, dtype=np.float64)
        world_map_length = 4.0
        resolution = 1.0
        cell_n = 4

    map_instance = object.__new__(GpuRollingElevationMap)
    map_instance.cp = np
    map_instance.map_spec = MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0)
    map_instance.output_resolution = 1.0
    map_instance.resolution = 1.0
    map_instance.world_map_length = 4.0
    map_instance.cell_n = 4
    map_instance.sample_mode = "bilinear"
    map_instance.sample_min_valid_weight = 0.5
    map_instance.rolling_grid = _Grid()

    shape = (4, 4)
    fields = {
        "observed": np.ones(shape, dtype=bool),
        "geometry_risk": np.full(shape, 0.3, dtype=np.float32),
        "geometry_confidence": np.ones(shape, dtype=np.float32),
        "geometry_valid_mask": np.ones(shape, dtype=bool),
        "geometry_support_count": np.full(shape, 8, dtype=np.int32),
        "elevation": np.zeros(shape, dtype=np.float32),
        "elevation_variance": np.full(shape, 0.01, dtype=np.float32),
        "elevation_std": np.full(shape, 0.1, dtype=np.float32),
        "uncertainty_risk": np.full(shape, 0.8, dtype=np.float32),
        "geometry_age": np.zeros(shape, dtype=np.float32),
        "geometry_conflict_score": np.zeros(shape, dtype=np.float32),
        "ground_fit_valid": np.ones(shape, dtype=bool),
        "geometry_risk_std": np.full(shape, 0.1, dtype=np.float32),
        "geometry_risk_cvar": np.full(shape, 0.6, dtype=np.float32),
        "geometry_risk_distribution_weight": np.ones(
            shape, dtype=np.float32
        ),
        "geometry_risk_distribution_valid": np.ones(shape, dtype=bool),
    }

    identity = map_instance._sample_current_body(fields, np.eye(4))
    quarter_turn = np.eye(4)
    quarter_turn[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
    rotated = map_instance._sample_current_body(fields, quarter_turn)

    np.testing.assert_allclose(identity["geometry_risk"], 0.3)
    np.testing.assert_allclose(rotated["geometry_risk"], 0.3)
    np.testing.assert_allclose(identity["geometry_risk_std"], 0.1)
    np.testing.assert_allclose(rotated["geometry_risk_cvar"], 0.6)
    np.testing.assert_allclose(identity["elevation_variance"], 0.01)
    np.testing.assert_allclose(rotated["elevation_std"], 0.1)
    np.testing.assert_allclose(identity["uncertainty_risk"], 0.8)
    np.testing.assert_array_equal(
        identity["geometry_risk_distribution_valid"],
        rotated["geometry_risk_distribution_valid"],
    )
    assert identity["geometry_risk_distribution_valid"].dtype == bool
    np.testing.assert_array_equal(
        identity["geometry_valid_mask"], rotated["geometry_valid_mask"]
    )


def test_body_sampling_applies_height_drift_to_robot_height_too():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        GpuRollingElevationMap,
    )
    from roadrunner_mapping.geometry.schema import MapSpec

    class _Grid:
        center = np.zeros(2, dtype=np.float64)

    map_instance = object.__new__(GpuRollingElevationMap)
    map_instance.cp = np
    map_instance.map_spec = MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0)
    map_instance.output_resolution = 1.0
    map_instance.resolution = 1.0
    map_instance.world_map_length = 4.0
    map_instance.cell_n = 4
    map_instance.sample_mode = "bilinear"
    map_instance.sample_min_valid_weight = 0.5
    map_instance.rolling_grid = _Grid()

    shape = (4, 4)
    fields = {
        "observed": np.ones(shape, dtype=bool),
        "geometry_risk": np.zeros(shape, dtype=np.float32),
        "geometry_confidence": np.ones(shape, dtype=np.float32),
        "geometry_valid_mask": np.ones(shape, dtype=bool),
        "geometry_support_count": np.ones(shape, dtype=np.int32),
        "elevation": np.zeros(shape, dtype=np.float32),
        "elevation_variance": np.full(shape, 0.01, dtype=np.float32),
        "elevation_std": np.full(shape, 0.1, dtype=np.float32),
        "uncertainty_risk": np.zeros(shape, dtype=np.float32),
        "geometry_age": np.zeros(shape, dtype=np.float32),
        "geometry_conflict_score": np.zeros(shape, dtype=np.float32),
        "ground_fit_valid": np.ones(shape, dtype=bool),
        "height_drift_bias_z": np.full(shape, 0.1, dtype=np.float32),
    }
    body_pose_with_common_drift = np.eye(4)
    body_pose_with_common_drift[2, 3] = 1.1

    sampled = map_instance._sample_current_body(
        fields,
        body_pose_with_common_drift,
    )

    valid = sampled["geometry_valid_mask"]
    np.testing.assert_allclose(sampled["elevation"][valid], -1.0, atol=1e-6)


def test_bilinear_sampling_derives_std_from_sampled_variance():
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        GpuRollingElevationMap,
    )
    from roadrunner_mapping.geometry.schema import MapSpec

    class _Grid:
        center = np.zeros(2, dtype=np.float64)

    map_instance = object.__new__(GpuRollingElevationMap)
    map_instance.cp = np
    map_instance.map_spec = MapSpec(-1.0, 1.0, -1.0, 1.0, 1.0)
    map_instance.output_resolution = 1.0
    map_instance.resolution = 1.0
    map_instance.world_map_length = 4.0
    map_instance.cell_n = 4
    map_instance.sample_mode = 'bilinear'
    map_instance.sample_min_valid_weight = 0.5
    map_instance.rolling_grid = _Grid()
    shape = (4, 4)
    variance = np.linspace(0.0004, 0.16, 16, dtype=np.float32).reshape(shape)
    fields = {
        'observed': np.ones(shape, dtype=bool),
        'geometry_risk': np.zeros(shape, dtype=np.float32),
        'geometry_confidence': np.ones(shape, dtype=np.float32),
        'geometry_valid_mask': np.ones(shape, dtype=bool),
        'geometry_support_count': np.ones(shape, dtype=np.int32),
        'elevation': np.zeros(shape, dtype=np.float32),
        'elevation_variance': variance,
        'elevation_std': np.sqrt(variance),
        'uncertainty_risk': np.zeros(shape, dtype=np.float32),
        'geometry_age': np.zeros(shape, dtype=np.float32),
        'geometry_conflict_score': np.zeros(shape, dtype=np.float32),
        'ground_fit_valid': np.ones(shape, dtype=bool),
    }
    translated = np.eye(4)
    translated[:2, 3] = 0.25
    sampled = map_instance._sample_current_body(fields, translated)
    valid = sampled['geometry_valid_mask']
    np.testing.assert_allclose(
        sampled['elevation_std'][valid] ** 2,
        sampled['elevation_variance'][valid],
        rtol=1e-6,
        atol=1e-7,
    )


def test_map_owns_complete_gaussian_state_without_legacy_moments(numpy_map):
    assert not hasattr(numpy_map, "count")
    assert not hasattr(numpy_map, "sum_z")
    assert not hasattr(numpy_map, "sum_z2")
    for name in (
        "elevation_mean",
        "elevation_variance",
        "elevation_valid",
        "geometry_support_count",
        "accepted_frame_count",
        "last_seen",
        "variance_updated_at",
        "geometry_conflict_score",
        "conflict_updated_at",
        "ground_sum",
        "min_z",
        "max_z",
        "risk_pending_observation",
    ):
        assert hasattr(numpy_map, name)


def _registered_layer_contract(grid):
    return (
        (grid.elevation_mean, 0.0),
        (grid.elevation_variance, grid.gaussian_config.max_variance),
        (grid.elevation_valid, False),
        (grid.geometry_support_count, 0),
        (grid.accepted_frame_count, 0),
        (grid.last_seen, -np.inf),
        (grid.variance_updated_at, -np.inf),
        (grid.geometry_conflict_score, 0.0),
        (grid.conflict_updated_at, -np.inf),
        (grid.ground_sum, 0.0),
        (grid.min_z, np.inf),
        (grid.max_z, -np.inf),
        (grid.risk_pending_observation, False),
        (grid.risk_sum, 0.0),
        (grid.risk_sq_sum, 0.0),
        (grid.risk_weight, 0.0),
        (grid.risk_last_update, -np.inf),
        (grid.elevation_validity, 0.0),
        (grid.upper_bound, 0.0),
        (grid.upper_bound_valid, False),
        (grid.upper_bound_last_update, -np.inf),
    )


def test_lifecycle_layers_and_drift_scalars_have_stable_contract(numpy_map):
    assert numpy_map.elevation_validity.dtype == np.float32
    assert numpy_map.upper_bound.dtype == np.float32
    assert numpy_map.upper_bound_valid.dtype == np.bool_
    assert numpy_map.upper_bound_last_update.dtype == np.float64
    np.testing.assert_array_equal(numpy_map.elevation_validity, 0.0)
    np.testing.assert_array_equal(numpy_map.upper_bound, 0.0)
    np.testing.assert_array_equal(numpy_map.upper_bound_valid, False)
    np.testing.assert_array_equal(numpy_map.upper_bound_last_update, -np.inf)
    assert numpy_map.height_drift_bias_z == 0.0
    assert numpy_map.height_drift_last_update == -np.inf

    numpy_map.height_drift_bias_z = 0.25
    numpy_map.height_drift_last_update = 4.0
    numpy_map.clear()

    assert numpy_map.height_drift_bias_z == 0.0
    assert numpy_map.height_drift_last_update == -np.inf


def test_lifecycle_config_is_validated_at_map_construction(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module
    from roadrunner_mapping.geometry.schema import MapSpec

    backend = _NumpyCupy()
    monkeypatch.setattr(
        module,
        "_load_gpu_modules",
        lambda: module._GpuModules(cp=backend, ndimage=ndimage),
    )
    with pytest.raises(TypeError, match="lifecycle_config"):
        module.GpuRollingElevationMap(
            map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 0.2),
            world_map_length=4.0,
            lifecycle_config=object(),
        )
    with pytest.raises(ValueError, match="drift_variance_max"):
        module.GpuRollingElevationMap(
            map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 0.2),
            world_map_length=4.0,
            gaussian_config=GaussianElevationConfig(
                max_variance=0.005,
                single_point_initial_variance=0.004,
                multi_point_initial_variance=0.003,
            ),
            lifecycle_config=ElevationMapLifecycleConfig(
                enabled=True,
                drift_variance_max=0.01,
            ),
        )

    disabled = module.GpuRollingElevationMap(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 0.2),
        world_map_length=4.0,
        gaussian_config=GaussianElevationConfig(
            max_variance=0.005,
            single_point_initial_variance=0.004,
            multi_point_initial_variance=0.003,
        ),
        lifecycle_config=ElevationMapLifecycleConfig(enabled=False),
    )
    assert not disabled.lifecycle_config.enabled


def test_disabled_lifecycle_is_bitwise_compatible_with_default_path(monkeypatch):
    baseline = _new_numpy_map(monkeypatch)
    disabled = _new_numpy_map(
        monkeypatch,
        lifecycle_config=ElevationMapLifecycleConfig(enabled=False),
    )
    outputs = []
    for grid in (baseline, disabled):
        first = _advance_points(grid, "one", 1.0, _dense_points(0.1))
        second = _advance_points(grid, "two", 2.0, _dense_points(0.12))
        outputs.append((first, second, _map_state_snapshot(grid)))

    for baseline_output, disabled_output in zip(outputs[0][:2], outputs[1][:2]):
        assert baseline_output.keys() == disabled_output.keys()
        for name in baseline_output:
            np.testing.assert_array_equal(
                baseline_output[name], disabled_output[name]
            )
    for actual, expected in zip(
        outputs[1][2]["layers"], outputs[0][2]["layers"]
    ):
        np.testing.assert_array_equal(actual, expected)


def test_disabled_lifecycle_ignores_new_offline_sensor_fields(monkeypatch):
    reference = _new_numpy_map(monkeypatch)
    candidate = _new_numpy_map(
        monkeypatch,
        lifecycle_config=ElevationMapLifecycleConfig(enabled=False),
    )
    point = np.array([[0.5, 0.0, 0.1]], dtype=np.float32)

    def frame(*, extra_fields):
        observation = SimpleNamespace(
            xyz=point,
            ground_probability=np.ones(1, dtype=np.float32),
            dynamic=np.zeros(1, dtype=bool),
            **extra_fields,
        )
        return SimpleNamespace(
            timestamp=1.0,
            world_from_body=np.eye(4),
            sensor_origin_world=np.array([np.nan, 0.0, 0.0]),
            observation_world=observation,
        )

    reference.advance(
        ("reference",),
        lambda _: frame(extra_fields={}),
        target_world_from_body=np.eye(4),
    )
    candidate.advance(
        ("candidate",),
        lambda _: frame(
            extra_fields={
                "relative_points_lidar": np.zeros((3, 2), dtype=np.float32)
            }
        ),
        target_world_from_body=np.eye(4),
    )
    for actual, expected in zip(
        _map_state_snapshot(candidate)["layers"],
        _map_state_snapshot(reference)["layers"],
    ):
        np.testing.assert_array_equal(actual, expected)


def test_enabled_lifecycle_filters_overhead_before_gaussian_support(
    lifecycle_numpy_map,
):
    points = np.array(
        [
            [0.01, 0.01, 0.0],
            [0.02, 0.02, 1.5],
        ],
        dtype=np.float32,
    )
    _advance_points(lifecycle_numpy_map, "overhead", 1.0, points)
    rows, columns, valid = lifecycle_numpy_map.rolling_grid.world_indices(
        points[:1]
    )
    assert bool(valid[0])
    cell = (int(rows[0]), int(columns[0]))
    assert lifecycle_numpy_map.geometry_support_count[cell] == 1
    assert lifecycle_numpy_map.elevation_mean[cell] == pytest.approx(0.0)
    assert lifecycle_numpy_map.elevation_validity[cell] == pytest.approx(1.0)


def test_enabled_offline_lifecycle_requires_true_sensor_origin(
    lifecycle_numpy_map,
):
    observation = SimpleNamespace(
        xyz=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
        ground_probability=np.ones(1, dtype=np.float32),
        dynamic=np.zeros(1, dtype=bool),
        height_variance=np.full(1, 0.002, dtype=np.float32),
    )
    frame = SimpleNamespace(
        timestamp=1.0,
        world_from_body=np.eye(4),
        observation_world=observation,
    )
    with pytest.raises(ValueError, match="sensor_origin_world"):
        lifecycle_numpy_map.advance(
            ("missing-origin",),
            lambda _: frame,
            target_world_from_body=np.eye(4),
        )


def test_enabled_offline_lifecycle_prefers_lidar_origin_over_body_origin(
    lifecycle_numpy_map,
):
    point = np.array([[0.1, 0.1, 1.0]], dtype=np.float32)
    frame = SimpleNamespace(
        timestamp=1.0,
        world_from_body=np.eye(4),
        sensor_origin_world=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        observation_world=SimpleNamespace(
            xyz=point,
            ground_probability=np.ones(1, dtype=np.float32),
            dynamic=np.zeros(1, dtype=bool),
            height_variance=np.full(1, 0.002, dtype=np.float32),
        ),
    )
    lifecycle_numpy_map.advance(
        ("lidar-origin",),
        lambda _: frame,
        target_world_from_body=np.eye(4),
    )
    rows, columns, valid = lifecycle_numpy_map.rolling_grid.world_indices(point)
    assert bool(valid[0])
    assert lifecycle_numpy_map.geometry_support_count[
        int(rows[0]), int(columns[0])
    ] == 1


def test_stable_ground_estimates_and_applies_common_height_drift(
    lifecycle_numpy_map,
):
    points = _dense_points(0.0)
    for index, timestamp in enumerate((1.0, 2.0, 3.0), start=1):
        _advance_points(
            lifecycle_numpy_map,
            "stable-{}".format(index),
            timestamp,
            points,
        )

    shifted = _dense_points(0.1)
    _advance_points(lifecycle_numpy_map, "shifted", 4.0, shifted)

    assert lifecycle_numpy_map.height_drift_bias_z == pytest.approx(0.01)
    assert lifecycle_numpy_map.height_drift_last_update == pytest.approx(3.0)
    valid = lifecycle_numpy_map.elevation_valid
    assert float(np.mean(lifecycle_numpy_map.elevation_mean[valid])) < 0.1


def test_wall_edge_rule_ignores_dense_low_points_before_gaussian_update(
    lifecycle_numpy_map,
):
    world_from_lidar = np.eye(4)
    world_from_lidar[0, 3] = -5.0
    high = np.column_stack(
        (
            np.full(24, 5.01),
            np.linspace(0.01, 0.05, 24),
            np.ones(24),
        )
    ).astype(np.float32)
    lifecycle_numpy_map.advance_pointcloud(
        "wall-seed",
        1.0,
        high,
        np.zeros(len(high), dtype=np.float32),
        world_from_lidar=world_from_lidar,
        minimum_range=0.0,
        maximum_range=10.0,
        target_world_from_body=np.eye(4),
    )
    mixed = high.copy()
    mixed[:12, 2] = 0.0
    lifecycle_numpy_map.advance_pointcloud(
        "wall-mixed",
        2.0,
        mixed,
        np.zeros(len(mixed), dtype=np.float32),
        world_from_lidar=world_from_lidar,
        minimum_range=0.0,
        maximum_range=10.0,
        target_world_from_body=np.eye(4),
    )
    world_point = np.array([[0.01, 0.03, 1.0]], dtype=np.float32)
    rows, columns, valid = lifecycle_numpy_map.rolling_grid.world_indices(
        world_point
    )
    assert bool(valid[0])
    cell = (int(rows[0]), int(columns[0]))
    assert lifecycle_numpy_map.elevation_mean[cell] == pytest.approx(1.0)
    assert lifecycle_numpy_map.geometry_conflict_score[cell] == pytest.approx(0.0)


def test_three_low_visibility_rays_clear_old_high_surface_and_keep_upper_bound(
    lifecycle_numpy_map,
):
    sensor_pose = np.eye(4)
    sensor_pose[:3, 3] = [-1.0, 0.0, 2.0]
    world_points = _dense_points(1.0)
    lidar_points = world_points - sensor_pose[:3, 3]
    for index, timestamp in enumerate((1.0, 2.0, 3.0), start=1):
        lifecycle_numpy_map.advance_pointcloud(
            "high-{}".format(index),
            timestamp,
            lidar_points,
            np.zeros(len(lidar_points), dtype=np.float32),
            world_from_lidar=sensor_pose,
            minimum_range=0.0,
            maximum_range=10.0,
            target_world_from_body=np.eye(4),
        )
    previously_valid = lifecycle_numpy_map.elevation_valid.copy()

    # The endpoint lies outside the rolling map, but its ray crosses the map
    # and must still provide visibility evidence.
    endpoint_world = np.array([[2.1, 0.0, 0.0]], dtype=np.float32)
    endpoint_lidar = endpoint_world - sensor_pose[:3, 3]
    for index, timestamp in enumerate((4.0, 5.0, 6.0), start=1):
        lifecycle_numpy_map.advance_pointcloud(
            "low-ray-{}".format(index),
            timestamp,
            endpoint_lidar,
            np.zeros(1, dtype=np.float32),
            world_from_lidar=sensor_pose,
            minimum_range=0.0,
            maximum_range=10.0,
            target_world_from_body=np.eye(4),
        )

    cleared = previously_valid & ~lifecycle_numpy_map.elevation_valid
    assert int(np.count_nonzero(cleared)) > 0
    assert np.all(lifecycle_numpy_map.elevation_validity[cleared] == 0.0)
    assert np.any(lifecycle_numpy_map.upper_bound_valid[cleared])
    assert np.all(
        lifecycle_numpy_map.upper_bound[
            cleared & lifecycle_numpy_map.upper_bound_valid
        ]
        < 1.0
    )
    stats = lifecycle_numpy_map.last_lifecycle_stats
    assert stats["visibility_input_rays"] == 1
    assert stats["visibility_cleaned_cells"] > 0
    assert stats["upper_bound_cells"] > 0
    assert stats["geometry_valid_cells"] == int(
        np.count_nonzero(lifecycle_numpy_map.elevation_valid)
    )
    assert isinstance(stats["height_drift_bias_z"], float)
    assert isinstance(stats["height_drift_sample_cells"], int)


def test_disabled_visibility_features_skip_rays_and_keep_real_elevation(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    grid = _new_numpy_map(
        monkeypatch,
        lifecycle_config=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
            enable_upper_bound=False,
        ),
    )

    def unexpected_raycast(*args, **kwargs):
        raise AssertionError("disabled visibility features must not sample rays")

    monkeypatch.setattr(module, "raycast_grid_samples", unexpected_raycast)
    _advance_points(grid, "no-rays", 1.0, _dense_points())

    assert np.count_nonzero(grid.elevation_valid) > 0
    np.testing.assert_array_equal(grid.elevation_mean[grid.elevation_valid], 0.0)
    np.testing.assert_array_equal(grid.elevation_validity[grid.elevation_valid], 1.0)
    assert not np.any(grid.upper_bound_valid)
    assert grid.last_lifecycle_stats["visibility_input_rays"] == 0
    assert grid.last_lifecycle_stats["visibility_cleaned_cells"] == 0


def test_disabled_visibility_features_only_refresh_accepted_observations(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    grid = _new_numpy_map(
        monkeypatch,
        lifecycle_config=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=False,
            enable_upper_bound=False,
        ),
    )
    apply_update = module.apply_visibility_and_upper_bounds
    refreshed = []

    def accepted_observations_only(*args, **kwargs):
        accepted_ids = args[2]
        assert accepted_ids.size > 0, "disabled features must skip empty visibility updates"
        refreshed.append(accepted_ids.copy())
        return apply_update(*args, **kwargs)

    monkeypatch.setattr(
        module, "apply_visibility_and_upper_bounds", accepted_observations_only
    )
    _advance_points(grid, "real-observation", 1.0, _dense_points())

    assert len(refreshed) == 1
    assert np.all(grid.elevation_validity.ravel()[refreshed[0]] == 1.0)


@pytest.mark.parametrize("cleanup,upper", [(False, False), (False, True), (True, False), (True, True)])
def test_lifecycle_visibility_options_are_independent_in_full_map(monkeypatch, cleanup, upper):
    grid = _new_numpy_map(
        monkeypatch,
        lifecycle_config=ElevationMapLifecycleConfig(
            enabled=True,
            enable_visibility_cleanup=cleanup,
            enable_upper_bound=upper,
        ),
    )
    sensor_pose = np.eye(4)
    sensor_pose[:3, 3] = [-1.0, 0.0, 2.0]
    high_points = _dense_points(1.0) - sensor_pose[:3, 3]
    low_endpoint = np.array([[2.1, 0.0, 0.0]], dtype=np.float32) - sensor_pose[:3, 3]

    def advance(name, timestamp, points):
        return grid.advance_pointcloud(
            name, timestamp, points, np.zeros(len(points), dtype=np.float32),
            world_from_lidar=sensor_pose,
            minimum_range=0.0, maximum_range=10.0,
            target_world_from_body=np.eye(4),
        )

    for index in range(3):
        advance("high-{}".format(index), float(index + 1), high_points)
    previously_valid = grid.elevation_valid.copy()
    for index in range(3):
        advance("low-{}".format(index), float(index + 4), low_endpoint)

    cleared = previously_valid & ~grid.elevation_valid
    assert bool(np.any(cleared)) == cleanup
    assert bool(np.any(grid.upper_bound_valid)) == upper
    if not cleanup:
        np.testing.assert_array_equal(grid.elevation_valid[previously_valid], True)
    if cleanup or upper:
        assert grid.last_lifecycle_stats["visibility_input_rays"] == 1
    else:
        assert grid.last_lifecycle_stats["visibility_input_rays"] == 0


def test_deskew_keeps_raw_lidar_range_and_one_rigid_sensor_origin(
    lifecycle_numpy_map,
):
    rigid_pose = np.eye(4)
    rigid_pose[:3, 3] = [1.0, 2.0, 3.0]
    row_pose = np.eye(4)
    row_pose[:3, 3] = [10.0, 0.0, 0.0]
    raw = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    (
        world_points,
        _,
        _,
        variances,
        relative_points,
        sensor_origin,
    ) = lifecycle_numpy_map._prepare_lidar_arrays(
        raw,
        np.zeros(1, dtype=np.float32),
        world_from_lidar=rigid_pose,
        minimum_range=0.0,
        maximum_range=20.0,
        row_world_from_lidar=np.asarray([row_pose]),
        point_rows=np.array([0], dtype=np.int32),
    )

    np.testing.assert_array_equal(relative_points, raw)
    np.testing.assert_array_equal(sensor_origin, rigid_pose[:3, 3])
    np.testing.assert_array_equal(world_points, [[11.0, 0.0, 0.0]])
    expected = lifecycle_numpy_map.gaussian_config
    from roadrunner_mapping.geometry.gaussian_elevation import (
        point_height_variance,
    )

    np.testing.assert_allclose(
        variances,
        point_height_variance(np, np.array([1.0]), expected),
    )


def _empty_world_frame(grid, frame_id, timestamp, robot_xyz):
    return grid._transactional_ingest_world_arrays(
        np.asarray(robot_xyz, dtype=np.float64),
        frame_id,
        timestamp,
        np.empty((0, 3), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.bool_),
        robot_position_world=np.asarray(robot_xyz, dtype=np.float64),
    )


def test_overlap_clearance_removes_near_wrong_floor_but_preserves_other_cells(
    lifecycle_numpy_map,
):
    grid = lifecycle_numpy_map
    grid.rolling_grid.move_center(np.zeros(2))
    center = grid.cell_n // 2
    wrong_floor = (center, center)
    height_ok = (center, center + 1)
    far_wrong_floor = (0, 0)
    for cell, height in (
        (wrong_floor, 3.0),
        (height_ok, 1.0),
        (far_wrong_floor, 3.0),
    ):
        grid.elevation_valid[cell] = True
        grid.elevation_validity[cell] = 1.0
        grid.elevation_mean[cell] = height
        grid.geometry_support_count[cell] = 5
        grid.risk_weight[cell] = 2.0
        for histogram in grid.risk_histogram:
            histogram[cell] = 1.0
        grid.upper_bound[cell] = height
        grid.upper_bound_valid[cell] = True
        grid.upper_bound_last_update[cell] = 0.0

    _empty_world_frame(grid, "overlap", 1.0, [0.0, 0.0, 0.0])

    assert not grid.elevation_valid[wrong_floor]
    assert grid.elevation_validity[wrong_floor] == 0.0
    assert grid.geometry_support_count[wrong_floor] == 0
    assert grid.risk_weight[wrong_floor] == 0.0
    assert all(histogram[wrong_floor] == 0.0 for histogram in grid.risk_histogram)
    assert not grid.upper_bound_valid[wrong_floor]
    assert grid.elevation_valid[height_ok]
    assert grid.upper_bound_valid[height_ok]
    assert grid.elevation_valid[far_wrong_floor]
    assert grid.upper_bound_valid[far_wrong_floor]


def test_empty_frame_expires_old_upper_bound_without_touching_fresh_one(
    lifecycle_numpy_map,
):
    grid = lifecycle_numpy_map
    _empty_world_frame(grid, "origin", 1.0, [0.0, 0.0, 0.0])
    old_cell = (3, 3)
    fresh_cell = (4, 4)
    for cell, update_time in ((old_cell, 0.0), (fresh_cell, 5.0)):
        grid.upper_bound[cell] = -0.5
        grid.upper_bound_valid[cell] = True
        grid.upper_bound_last_update[cell] = update_time

    _empty_world_frame(grid, "expire", 12.0, [0.0, 0.0, 0.0])

    assert not grid.upper_bound_valid[old_cell]
    assert grid.upper_bound[old_cell] == 0.0
    assert grid.upper_bound_last_update[old_cell] == -np.inf
    assert grid.upper_bound_valid[fresh_cell]
    assert grid.upper_bound_last_update[fresh_cell] == 5.0


def _write_layer_sentinels(grid):
    expected_layers = {
        id(layer) for layer, _ in _registered_layer_contract(grid)
    } | {id(histogram) for histogram in grid.risk_histogram}
    registered_layers = {id(layer) for layer, _ in grid.rolling_grid._layers}
    assert expected_layers == registered_layers
    for layer, _ in _registered_layer_contract(grid):
        layer.fill(True if layer.dtype.kind == 'b' else 7)
    for histogram in grid.risk_histogram:
        histogram.fill(7)


def _map_state_snapshot(grid):
    world_fields = (
        None
        if grid._last_world_geometry_fields is None
        else {
            name: np.asarray(values).copy()
            for name, values in grid._last_world_geometry_fields.items()
        }
    )
    return {
        'center': grid.rolling_grid.center,
        'initialized': grid.rolling_grid.initialized,
        'time_origin': grid._time_origin,
        'latest_timestamp': grid._latest_timestamp,
        'height_drift_bias_z': grid.height_drift_bias_z,
        'height_drift_last_update': grid.height_drift_last_update,
        'last_lifecycle_stats': dict(grid.last_lifecycle_stats),
        'ingested_ids': tuple(grid._ingested_ids),
        'ingested_timestamps': tuple(grid._ingested_timestamps),
        'ingested_set': frozenset(grid._ingested_set),
        'layers': tuple(layer.copy() for layer, _ in grid.rolling_grid._layers),
        'world_fields': world_fields,
    }


def _assert_map_state_equal(grid, snapshot):
    np.testing.assert_array_equal(grid.rolling_grid.center, snapshot['center'])
    assert grid.rolling_grid.initialized == snapshot['initialized']
    assert grid._time_origin == snapshot['time_origin']
    assert grid._latest_timestamp == snapshot['latest_timestamp']
    assert grid.height_drift_bias_z == snapshot['height_drift_bias_z']
    assert grid.height_drift_last_update == snapshot['height_drift_last_update']
    assert grid.last_lifecycle_stats == snapshot['last_lifecycle_stats']
    assert tuple(grid._ingested_ids) == snapshot['ingested_ids']
    assert tuple(grid._ingested_timestamps) == snapshot['ingested_timestamps']
    assert frozenset(grid._ingested_set) == snapshot['ingested_set']
    for actual, expected in zip(
        (layer for layer, _ in grid.rolling_grid._layers), snapshot['layers']
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    if snapshot['world_fields'] is None:
        assert grid._last_world_geometry_fields is None
    else:
        assert set(grid._last_world_geometry_fields) == set(snapshot['world_fields'])
        for name, expected in snapshot['world_fields'].items():
            np.testing.assert_array_equal(
                grid._last_world_geometry_fields[name], expected
            )


def test_rolling_shift_clears_every_posterior_auxiliary_and_risk_layer(numpy_map):
    numpy_map.rolling_grid.move_center(np.zeros(2))
    _write_layer_sentinels(numpy_map)
    numpy_map.rolling_grid.move_center(np.array([numpy_map.resolution, 0.0]))
    for layer, fill in _registered_layer_contract(numpy_map):
        np.testing.assert_array_equal(layer[0, :], fill)
        expected = True if layer.dtype.kind == 'b' else 7
        np.testing.assert_array_equal(layer[1:, :], expected)
    for histogram in numpy_map.risk_histogram:
        np.testing.assert_array_equal(histogram[0, :], 0.0)
        np.testing.assert_array_equal(histogram[1:, :], 7.0)


def test_clear_resets_every_posterior_auxiliary_and_risk_layer(numpy_map):
    numpy_map.rolling_grid.move_center(np.zeros(2))
    _write_layer_sentinels(numpy_map)
    numpy_map.clear()
    for layer, fill in _registered_layer_contract(numpy_map):
        np.testing.assert_array_equal(layer, fill)
    for histogram in numpy_map.risk_histogram:
        np.testing.assert_array_equal(histogram, 0.0)


def test_expiry_reinitializes_every_cell_state_and_discards_old_risk(numpy_map):
    point = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    _advance_points(numpy_map, 'old', 1.0, point)
    rows, columns, valid = numpy_map.rolling_grid.world_indices(point)
    assert bool(valid[0])
    cell = (int(rows[0]), int(columns[0]))
    state_sentinels = {
        'elevation_mean': 5.0,
        'elevation_variance': 0.02,
        'elevation_valid': True,
        'geometry_support_count': 9,
        'accepted_frame_count': 5,
        'last_seen': 0.0,
        'variance_updated_at': 0.0,
        'geometry_conflict_score': 0.7,
        'conflict_updated_at': 0.0,
        'ground_sum': 8.0,
        'min_z': 4.0,
        'max_z': 6.0,
        'risk_pending_observation': True,
        'risk_sum': 7.0,
        'risk_sq_sum': 7.0,
        'risk_weight': 7.0,
        'risk_last_update': 1.0,
    }
    for name, value in state_sentinels.items():
        getattr(numpy_map, name)[cell] = value
    for histogram in numpy_map.risk_histogram:
        histogram[cell] = 7.0

    _advance_points(numpy_map, 'fresh', 200.0, point)
    assert numpy_map.elevation_mean[cell] == pytest.approx(1.0)
    assert numpy_map.elevation_variance[cell] == pytest.approx(
        numpy_map.gaussian_config.single_point_initial_variance
    )
    assert bool(numpy_map.elevation_valid[cell])
    assert numpy_map.geometry_support_count[cell] == 1
    assert numpy_map.accepted_frame_count[cell] == 1
    assert numpy_map.last_seen[cell] == pytest.approx(199.0)
    assert numpy_map.variance_updated_at[cell] == pytest.approx(199.0)
    assert numpy_map.geometry_conflict_score[cell] == pytest.approx(0.0)
    assert numpy_map.conflict_updated_at[cell] == pytest.approx(199.0)
    assert numpy_map.ground_sum[cell] == pytest.approx(1.0)
    assert numpy_map.min_z[cell] == pytest.approx(1.0)
    assert numpy_map.max_z[cell] == pytest.approx(1.0)
    assert not bool(numpy_map.risk_pending_observation[cell])
    assert numpy_map.risk_sum[cell] == pytest.approx(0.0)
    assert numpy_map.risk_sq_sum[cell] == pytest.approx(0.0)
    assert numpy_map.risk_weight[cell] == pytest.approx(0.0)
    assert np.isneginf(numpy_map.risk_last_update[cell])
    for histogram in numpy_map.risk_histogram:
        assert histogram[cell] == pytest.approx(0.0)


def test_three_consistent_frames_raise_confidence_and_reduce_uncertainty(numpy_map):
    from roadrunner_mapping.geometry.gpu_rolling_elevation import (
        GpuRollingElevationMap,
    )
    from roadrunner_mapping.geometry.schema import MapSpec

    numpy_map = GpuRollingElevationMap(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 0.2),
        world_map_length=4.0,
        gaussian_config=GaussianElevationConfig(pose_vertical_variance=0.005),
    )
    points = _dense_points()
    confidences = []
    uncertainty = []
    frame_counts = []
    rows, columns, valid = numpy_map.rolling_grid.world_indices(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    )
    assert bool(valid[0])
    cell = (int(rows[0]), int(columns[0]))
    for index, timestamp in enumerate((1.0, 2.0, 3.0), start=1):
        result = _advance_points(numpy_map, 'frame-{}'.format(index), timestamp, points)
        valid_map = result['geometry_valid_mask']
        confidences.append(float(np.mean(result['geometry_confidence'][valid_map])))
        uncertainty.append(float(np.mean(result['uncertainty_risk'][valid_map])))
        frame_counts.append(int(numpy_map.accepted_frame_count[cell]))
    assert frame_counts == [1, 2, 3]
    assert confidences[0] < confidences[1] < confidences[2]
    assert uncertainty[0] > uncertainty[1] > uncertainty[2]


def test_duplicate_and_nonincreasing_new_frame_time_contract(numpy_map):
    points = _dense_points()
    first = _advance_points(numpy_map, "first", 1.0, points)
    variance = numpy_map.elevation_variance.copy()
    histogram = [values.copy() for values in numpy_map.risk_histogram]
    duplicate = _advance_points(numpy_map, "first", 1.0, points + [0, 0, 2])
    np.testing.assert_array_equal(numpy_map.elevation_variance, variance)
    for before, after in zip(histogram, numpy_map.risk_histogram):
        np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(duplicate["geometry_risk"], first["geometry_risk"])
    with pytest.raises(ValueError, match="strictly increasing"):
        _advance_points(numpy_map, "same-time", 1.0, points)
    with pytest.raises(ValueError, match="strictly increasing"):
        _advance_points(numpy_map, "older", 0.5, points)


@pytest.mark.parametrize('timestamp', [1.0, 0.5])
def test_invalid_pointcloud_time_with_far_pose_is_fully_atomic(numpy_map, timestamp):
    points = _dense_points()
    _advance_points(numpy_map, 'first', 1.0, points)
    before = _map_state_snapshot(numpy_map)
    far_pose = np.eye(4)
    far_pose[0, 3] = 100.0
    with pytest.raises(ValueError, match='strictly increasing'):
        numpy_map.advance_pointcloud(
            'invalid-time-{}'.format(timestamp),
            timestamp,
            points,
            np.zeros(len(points), dtype=np.float32),
            world_from_lidar=np.eye(4),
            minimum_range=0.0,
            maximum_range=10.0,
            target_world_from_body=far_pose,
        )
    _assert_map_state_equal(numpy_map, before)


def test_invalid_offline_time_with_far_pose_is_fully_atomic(numpy_map):
    from roadrunner_mapping.geometry.pointcloud import PointObservation

    points = _dense_points()
    _advance_points(numpy_map, 'first', 1.0, points)
    observation = PointObservation(
        xyz=points,
        intensity=np.zeros(len(points)),
        semantic_probabilities=np.ones((len(points), 1)),
        ground_probability=np.ones(len(points)),
        dynamic=np.zeros(len(points), dtype=bool),
    )
    invalid = SimpleNamespace(
        timestamp=1.0,
        world_from_body=np.eye(4),
        observation_world=observation,
    )
    before = _map_state_snapshot(numpy_map)
    far_pose = np.eye(4)
    far_pose[1, 3] = 100.0
    with pytest.raises(ValueError, match='strictly increasing'):
        numpy_map.advance(
            ('invalid-offline-time',),
            lambda _: invalid,
            target_world_from_body=far_pose,
        )
    _assert_map_state_equal(numpy_map, before)


def test_invalid_first_pointcloud_input_with_far_pose_does_not_initialize_map(
    numpy_map,
):
    before = _map_state_snapshot(numpy_map)
    far_pose = np.eye(4)
    far_pose[0, 3] = 100.0
    with pytest.raises(ValueError, match='shape'):
        numpy_map.advance_pointcloud(
            'invalid-first-pointcloud',
            1.0,
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            world_from_lidar=np.eye(4),
            minimum_range=0.0,
            maximum_range=10.0,
            target_world_from_body=far_pose,
        )
    _assert_map_state_equal(numpy_map, before)


def test_invalid_first_offline_input_with_far_pose_does_not_initialize_map(
    numpy_map,
):
    observation = SimpleNamespace(
        xyz=np.zeros((2, 3), dtype=np.float32),
        ground_probability=np.ones(1, dtype=np.float32),
        dynamic=np.zeros(2, dtype=bool),
    )
    frame = SimpleNamespace(
        timestamp=1.0,
        world_from_body=np.eye(4),
        observation_world=observation,
    )
    before = _map_state_snapshot(numpy_map)
    far_pose = np.eye(4)
    far_pose[1, 3] = 100.0
    with pytest.raises(ValueError, match='metadata'):
        numpy_map.advance(
            ('invalid-first-offline',),
            lambda _: frame,
            target_world_from_body=far_pose,
        )
    _assert_map_state_equal(numpy_map, before)


def test_duplicate_id_bypasses_time_and_input_validation_without_moving(numpy_map):
    points = _dense_points()
    _advance_points(numpy_map, 'first', 1.0, points)
    before = _map_state_snapshot(numpy_map)
    far_pose = np.eye(4)
    far_pose[0, 3] = 100.0
    numpy_map.advance_pointcloud(
        'first',
        np.nan,
        np.empty((3, 2), dtype=np.float32),
        np.empty(99, dtype=np.float32),
        world_from_lidar=np.eye(4),
        minimum_range=10.0,
        maximum_range=1.0,
        target_world_from_body=far_pose,
    )
    _assert_map_state_equal(numpy_map, before)


@pytest.mark.parametrize(
    ('points', 'ground', 'dynamic', 'variance'),
    [
        (np.zeros((2, 2)), np.ones(2), np.zeros(2, dtype=bool), None),
        (np.zeros((2, 3)), np.ones(1), np.zeros(2, dtype=bool), None),
        (np.zeros((2, 3)), np.ones(2), np.zeros(2, dtype=bool), np.ones(1)),
    ],
)
def test_invalid_first_world_frame_leaves_all_state_uninitialized(
    numpy_map, points, ground, dynamic, variance
):
    before = _map_state_snapshot(numpy_map)
    with pytest.raises(ValueError):
        numpy_map._ingest_world_arrays(
            'invalid-first',
            1.0,
            points,
            ground,
            dynamic,
            point_variances=variance,
        )
    _assert_map_state_equal(numpy_map, before)


def test_first_frame_fusion_exception_does_not_commit_origin_or_metadata(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    def fail_fusion(*args, **kwargs):
        raise RuntimeError('synthetic fusion failure')

    monkeypatch.setattr(module, 'update_gaussian_elevation', fail_fusion)
    before = _map_state_snapshot(numpy_map)
    with pytest.raises(RuntimeError, match='synthetic fusion failure'):
        numpy_map._ingest_world_arrays(
            'fusion-failure',
            1.0,
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=bool),
        )
    _assert_map_state_equal(numpy_map, before)


def test_existing_map_center_move_and_fusion_exception_are_fully_atomic(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    _advance_points(numpy_map, 'seed', 1.0, _dense_points())
    before = _map_state_snapshot(numpy_map)

    def fail_fusion(*args, **kwargs):
        raise RuntimeError('synthetic fusion failure after rolling')

    monkeypatch.setattr(module, 'update_gaussian_elevation', fail_fusion)
    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.4
    shifted_pose[1, 3] = -0.6
    with pytest.raises(RuntimeError, match='failure after rolling'):
        numpy_map.advance_pointcloud(
            'failed-shift',
            2.0,
            _dense_points(0.1),
            np.zeros(len(_dense_points()), dtype=np.float32),
            world_from_lidar=np.eye(4),
            minimum_range=0.0,
            maximum_range=10.0,
            target_world_from_body=shifted_pose,
        )
    _assert_map_state_equal(numpy_map, before)


def test_expiry_cleanup_and_fusion_exception_are_fully_atomic(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    points = _dense_points()
    _advance_points(numpy_map, 'seed', 1.0, points)
    before = _map_state_snapshot(numpy_map)

    def fail_fusion(*args, **kwargs):
        raise RuntimeError('synthetic fusion failure after expiry')

    monkeypatch.setattr(module, 'update_gaussian_elevation', fail_fusion)
    with pytest.raises(RuntimeError, match='failure after expiry'):
        _advance_points(
            numpy_map,
            'failed-expiry',
            1.0 + numpy_map.max_geometry_age_sec + 1.0,
            points,
        )
    _assert_map_state_equal(numpy_map, before)


def test_lifecycle_cleanup_and_drift_failure_are_fully_atomic(
    lifecycle_numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    grid = lifecycle_numpy_map
    points = _dense_points()
    _advance_points(grid, "seed", 1.0, points)
    center = grid.cell_n // 2
    wrong_floor = (center, center)
    grid.elevation_mean[wrong_floor] = 3.0
    grid.elevation_valid[wrong_floor] = True
    grid.elevation_validity[wrong_floor] = 0.75
    grid.upper_bound[wrong_floor] = 3.5
    grid.upper_bound_valid[wrong_floor] = True
    grid.upper_bound_last_update[wrong_floor] = 0.0
    grid.risk_weight[wrong_floor] = 2.0
    for histogram in grid.risk_histogram:
        histogram[wrong_floor] = 1.0
    grid.height_drift_bias_z = 0.15
    grid.height_drift_last_update = 1.0
    before = _map_state_snapshot(grid)

    def fail_fusion(*args, **kwargs):
        grid.height_drift_bias_z = 0.9
        grid.height_drift_last_update = 2.0
        raise RuntimeError("synthetic lifecycle fusion failure")

    monkeypatch.setattr(module, "update_gaussian_elevation", fail_fusion)
    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.4
    with pytest.raises(RuntimeError, match="lifecycle fusion failure"):
        grid.advance_pointcloud(
            "failed-lifecycle",
            2.0,
            points,
            np.zeros(len(points), dtype=np.float32),
            world_from_lidar=np.eye(4),
            minimum_range=0.0,
            maximum_range=10.0,
            target_world_from_body=shifted_pose,
        )
    _assert_map_state_equal(grid, before)


def test_frame_metadata_transaction_does_not_copy_full_history(numpy_map):
    class _NoIterationList(list):
        def __iter__(self):
            raise AssertionError('metadata transaction copied the full list')

    class _NoIterationSet(set):
        def __iter__(self):
            raise AssertionError('metadata transaction copied the full set')

    history_size = 10_000
    numpy_map._ingested_ids = _NoIterationList(
        'history-{}'.format(index) for index in range(history_size)
    )
    numpy_map._ingested_timestamps = _NoIterationList(
        float(index + 1) for index in range(history_size)
    )
    numpy_map._ingested_set = _NoIterationSet(
        'history-{}'.format(index) for index in range(history_size)
    )
    numpy_map._time_origin = 1.0
    numpy_map._latest_timestamp = float(history_size)

    numpy_map._ingest_world_arrays(
        'new-frame',
        float(history_size + 1),
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=bool),
    )

    assert len(numpy_map._ingested_ids) == history_size + 1
    assert len(numpy_map._ingested_timestamps) == history_size + 1
    assert list.__getitem__(numpy_map._ingested_ids, -1) == 'new-frame'
    assert set.__contains__(numpy_map._ingested_set, 'new-frame')


def test_touched_backup_skips_semantic_layer_but_rolling_rollback_restores_it(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    class _SemanticSentinel(np.ndarray):
        def __new__(cls, values):
            instance = np.asarray(values).view(cls)
            instance.advanced_index_reads = 0
            return instance

        def __array_finalize__(self, source):
            self.advanced_index_reads = getattr(
                source, 'advanced_index_reads', 0
            )

        def __getitem__(self, key):
            if isinstance(key, np.ndarray):
                self.advanced_index_reads += 1
                raise AssertionError(
                    'geometry touched backup read the semantic layer'
                )
            return super().__getitem__(key)

    _advance_points(numpy_map, 'seed', 1.0, _dense_points())
    semantic = _SemanticSentinel(
        np.arange(numpy_map.cell_n**2, dtype=np.float32).reshape(
            numpy_map.cell_n, numpy_map.cell_n
        )
    )
    numpy_map.rolling_grid.register_layer(semantic, -1.0)
    semantic_before = np.asarray(semantic).copy()
    state_before = _map_state_snapshot(numpy_map)

    def fail_fusion(*args, **kwargs):
        raise RuntimeError('synthetic geometry fusion failure')

    monkeypatch.setattr(module, 'update_gaussian_elevation', fail_fusion)
    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.4
    shifted_pose[1, 3] = -0.6
    with pytest.raises(RuntimeError, match='geometry fusion failure'):
        numpy_map.advance_pointcloud(
            'failed-with-semantic-layer',
            2.0,
            _dense_points(0.1),
            np.zeros(len(_dense_points()), dtype=np.float32),
            world_from_lidar=np.eye(4),
            minimum_range=0.0,
            maximum_range=10.0,
            target_world_from_body=shifted_pose,
        )

    assert semantic.advanced_index_reads == 0
    np.testing.assert_array_equal(np.asarray(semantic), semantic_before)
    _assert_map_state_equal(numpy_map, state_before)


def _offline_test_frame(timestamp, height=0.0):
    points = _dense_points(height)
    observation = SimpleNamespace(
        xyz=points,
        ground_probability=np.ones(len(points), dtype=np.float32),
        dynamic=np.zeros(len(points), dtype=bool),
    )
    return SimpleNamespace(
        timestamp=timestamp,
        world_from_body=np.eye(4),
        observation_world=observation,
    )


def test_advance_loads_and_commits_each_frame_before_loading_the_next(numpy_map):
    frames = {
        'first': _offline_test_frame(1.0),
        'second': _offline_test_frame(2.0, 0.01),
        'third': _offline_test_frame(3.0, 0.02),
    }
    states_seen_by_loader = []

    def loader(frame_id):
        states_seen_by_loader.append(numpy_map.ingested_frame_ids)
        return frames[frame_id]

    numpy_map.advance(
        ('first', 'second', 'third'),
        loader,
        target_world_from_body=np.eye(4),
    )

    assert states_seen_by_loader == [(), ('first',), ('first', 'second')]


def test_single_advance_computes_world_geometry_once(numpy_map, monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    original = module._compute_world_geometry_fields
    calls = []

    def counted(**kwargs):
        calls.append(kwargs["latest_timestamp"])
        return original(**kwargs)

    monkeypatch.setattr(module, "_compute_world_geometry_fields", counted)
    numpy_map.advance(
        ("first",),
        lambda _: _offline_test_frame(1.0),
        target_world_from_body=np.eye(4),
    )

    assert calls == [1.0]


def test_lifecycle_height_drift_computes_full_geometry_once_per_frame(
    lifecycle_numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    assert lifecycle_numpy_map.lifecycle_config.enable_height_drift_compensation
    original = module._compute_world_geometry_fields
    calls = []

    def counted(**kwargs):
        calls.append(kwargs["latest_timestamp"])
        return original(**kwargs)

    monkeypatch.setattr(module, "_compute_world_geometry_fields", counted)
    _advance_points(lifecycle_numpy_map, "first", 1.0, _dense_points())
    _advance_points(lifecycle_numpy_map, "second", 2.0, _dense_points(0.01))

    assert calls == [1.0, 2.0]


def test_disabled_height_drift_does_not_compute_lightweight_slope(monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    grid = _new_numpy_map(
        monkeypatch,
        lifecycle_config=ElevationMapLifecycleConfig(
            enabled=True,
            enable_height_drift_compensation=False,
        ),
    )
    _advance_points(grid, "first", 1.0, _dense_points())

    def fail():
        raise AssertionError("disabled height drift must not compute slope")

    monkeypatch.setattr(grid, "_compute_current_world_slope", fail)
    _advance_points(grid, "second", 2.0, _dense_points(0.01))


def test_advance_pointcloud_computes_world_geometry_once_per_new_frame(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    original = module._compute_world_geometry_fields
    calls = []

    def counted(**kwargs):
        calls.append(kwargs["latest_timestamp"])
        return original(**kwargs)

    monkeypatch.setattr(module, "_compute_world_geometry_fields", counted)
    _advance_points(numpy_map, "first", 1.0, _dense_points())

    assert calls == [1.0]


def test_batch_advance_computes_geometry_once_per_frame_and_not_for_final_render(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    frames = {
        "first": _offline_test_frame(1.0),
        "second": _offline_test_frame(2.0, 0.01),
        "third": _offline_test_frame(3.0, 0.02),
    }
    original = module._compute_world_geometry_fields
    calls = []

    def counted(**kwargs):
        calls.append(kwargs["latest_timestamp"])
        return original(**kwargs)

    monkeypatch.setattr(module, "_compute_world_geometry_fields", counted)
    numpy_map.advance(
        tuple(frames),
        frames.__getitem__,
        target_world_from_body=np.eye(4),
    )

    assert calls == [1.0, 2.0, 3.0]


def test_advance_keeps_committed_prefix_when_second_frame_is_invalid(numpy_map):
    frames = {
        'first': _offline_test_frame(1.0),
        'invalid-second': _offline_test_frame(1.0, 0.5),
    }
    state_before_second = None
    states_seen_by_loader = []

    def loader(frame_id):
        nonlocal state_before_second
        states_seen_by_loader.append(numpy_map.ingested_frame_ids)
        if frame_id == 'invalid-second':
            state_before_second = _map_state_snapshot(numpy_map)
        return frames[frame_id]

    with pytest.raises(ValueError, match='strictly increasing'):
        numpy_map.advance(
            ('first', 'invalid-second'),
            loader,
            target_world_from_body=np.eye(4),
        )

    assert states_seen_by_loader == [(), ('first',)]
    assert state_before_second is not None
    _assert_map_state_equal(numpy_map, state_before_second)


def test_empty_frame_advances_render_time_without_mutating_posterior(numpy_map):
    _advance_points(numpy_map, "first", 1.0, _dense_points())
    stored = numpy_map.elevation_variance.copy()
    rendered = _advance_points(
        numpy_map, "empty", 3.0, np.empty((0, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(numpy_map.elevation_variance, stored)
    assert numpy_map._latest_timestamp == 3.0
    valid = rendered["geometry_valid_mask"]
    assert np.all(rendered["elevation_variance"][valid] >= stored[stored < 0.25].min())


def test_repeated_render_is_read_only_and_recomputes_without_frame_cache(
    numpy_map, monkeypatch
):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module

    _advance_points(numpy_map, "first", 1.0, _dense_points())
    variance = numpy_map.elevation_variance.copy()
    histogram = [values.copy() for values in numpy_map.risk_histogram]
    original = module._compute_world_geometry_fields
    calls = 0

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(module, "_compute_world_geometry_fields", counted)
    numpy_map._geometry_maps(np.eye(4))
    numpy_map._geometry_maps(np.eye(4))
    assert calls == 2
    np.testing.assert_array_equal(numpy_map.elevation_variance, variance)
    for before, after in zip(histogram, numpy_map.risk_histogram):
        np.testing.assert_array_equal(after, before)


def test_batch_advance_matches_two_single_advances_for_cvar(numpy_map, monkeypatch):
    import roadrunner_mapping.geometry.gpu_rolling_elevation as module
    from roadrunner_mapping.geometry.pointcloud import PointObservation

    def frame(frame_id, timestamp, height):
        points = _dense_points(height)
        observation = PointObservation(
            xyz=points,
            intensity=np.zeros(len(points)),
            semantic_probabilities=np.ones((len(points), 1)),
            ground_probability=np.ones(len(points)),
            dynamic=np.zeros(len(points), dtype=bool),
        )
        return SimpleNamespace(
            timestamp=timestamp,
            world_from_body=np.eye(4),
            observation_world=observation,
        )

    frames = {"f1": frame("f1", 1.0, 0.0), "f2": frame("f2", 2.0, 0.02)}
    separate = numpy_map
    separate.advance(("f1",), frames.__getitem__, target_world_from_body=np.eye(4))
    separate.advance(("f2",), frames.__getitem__, target_world_from_body=np.eye(4))

    backend = _NumpyCupy()
    monkeypatch.setattr(
        module,
        "_load_gpu_modules",
        lambda: module._GpuModules(cp=backend, ndimage=ndimage),
    )
    from roadrunner_mapping.geometry.schema import MapSpec
    batch = module.GpuRollingElevationMap(
        map_spec=MapSpec(-1.0, 1.0, -1.0, 1.0, 0.2),
        world_map_length=4.0,
    )
    batch_result = batch.advance(
        ("f1", "f2"), frames.__getitem__, target_world_from_body=np.eye(4)
    )
    separate_result = separate._geometry_maps(np.eye(4))
    for before, after in zip(separate.risk_histogram, batch.risk_histogram):
        np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        batch_result["geometry_risk_cvar"],
        separate_result["geometry_risk_cvar"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_clear_shift_expiry_and_snapshot_cover_new_layers(numpy_map):
    result = _advance_points(numpy_map, "first", 1.0, _dense_points())
    for name in ("elevation_variance", "elevation_std", "uncertainty_risk"):
        assert name in result
    snapshot = numpy_map.world_geometry_snapshot()
    for name in (
        "elevation_variance",
        "elevation_std",
        "uncertainty_risk",
        "elevation_validity",
        "elevation_upper_bound",
        "elevation_upper_bound_valid",
        "height_drift_bias_z",
    ):
        assert name in snapshot
        assert name in result
        assert snapshot[name].shape == numpy_map.elevation_mean.shape
        assert result[name].shape == numpy_map.map_spec.shape
    numpy_map.rolling_grid.move_center(np.array([10.0, 0.0]))
    assert not np.any(numpy_map.elevation_valid)
    assert not np.any(numpy_map.risk_pending_observation)
    numpy_map.clear()
    assert np.all(numpy_map.elevation_variance == numpy_map.gaussian_config.max_variance)
    assert not np.any(numpy_map.accepted_frame_count)


def test_offline_ingestion_prefers_cached_variance_and_falls_back_to_body_range(
    numpy_map,
):
    from roadrunner_mapping.geometry.pointcloud import PointObservation

    observation = PointObservation(
        xyz=np.array([[0.0, 0.0, 0.0]]),
        intensity=np.zeros(1),
        semantic_probabilities=np.ones((1, 1)),
        ground_probability=np.ones(1),
        dynamic=np.zeros(1, dtype=bool),
    )
    explicit = SimpleNamespace(
        timestamp=1.0,
        world_from_body=np.eye(4),
        observation_world=observation,
        height_variance=np.array([0.02], dtype=np.float32),
    )
    numpy_map.advance(("explicit",), lambda _: explicit, target_world_from_body=np.eye(4))
    assert numpy_map.elevation_variance[numpy_map.elevation_valid][0] == pytest.approx(
        0.02 + numpy_map.gaussian_config.pose_vertical_variance
    )

    numpy_map.clear()
    far_pose = np.eye(4)
    far_pose[0, 3] = 100.0
    fallback = SimpleNamespace(
        timestamp=2.0,
        world_from_body=far_pose,
        observation_world=observation,
    )
    numpy_map.advance(("fallback",), lambda _: fallback, target_world_from_body=np.eye(4))
    expected_point_variance = min(
        numpy_map.gaussian_config.sensor_std_m**2
        + numpy_map.gaussian_config.range_variance_factor * 100.0**2,
        numpy_map.gaussian_config.measurement_variance_max,
    )
    assert numpy_map.elevation_variance[numpy_map.elevation_valid][0] == pytest.approx(
        expected_point_variance + numpy_map.gaussian_config.pose_vertical_variance
    )
