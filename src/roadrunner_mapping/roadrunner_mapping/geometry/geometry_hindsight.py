"""SLAM-aligned point accumulation for RoadRunner geometry supervision."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from itertools import chain
from typing import Callable, Iterable, Mapping

import numpy as np

from .coordinates import validate_transform
from .elevation_grid_geometry import fit_packed_geometry_elevation_grid
from .geometry_batch import pack_geometry_frames_batched
from .geometry_cuda import fit_packed_geometry_cuda
from .gpu_rolling_elevation import GpuRollingElevationMap
from .geometry_snapshot import (
    DEFAULT_GRASS_GEOMETRY_CONFIG,
    GrassGeometryConfig,
    apply_grass_geometry_risk,
    fit_packed_geometry_cpu,
)
from .pointcloud import PointObservation
from .schema import MapSpec
from .sliding_voxel_geometry import SlidingVoxelWindow


@dataclass(frozen=True)
class AccumulatedGeometryFrame:
    timestamp: float
    sequence_id: str
    split: str
    world_from_body: np.ndarray
    observation_world: PointObservation

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("geometry frame timestamp must be finite")
        if not self.sequence_id or not self.split:
            raise ValueError("geometry frame sequence and split are required")
        object.__setattr__(
            self,
            "world_from_body",
            validate_transform(self.world_from_body).copy(),
        )


def _select_window(
    frames: Iterable[AccumulatedGeometryFrame],
    target: AccumulatedGeometryFrame,
    half_window: float,
) -> list[AccumulatedGeometryFrame]:
    if not np.isfinite(half_window) or half_window < 0.0:
        raise ValueError("half_window must be finite and non-negative")
    selected = [
        frame
        for frame in frames
        if frame.sequence_id == target.sequence_id
        and frame.split == target.split
        and abs(frame.timestamp - target.timestamp) <= half_window
    ]
    return sorted(selected, key=lambda frame: frame.timestamp)


def _build_accumulated_geometry_teacher_ordered(
    frames: Iterable[AccumulatedGeometryFrame],
    target: AccumulatedGeometryFrame,
    map_spec: MapSpec,
    *,
    half_window: float = 30.0,
    obstacle_height: float = 0.35,
    backend: str = "cpu",
    geometry_method: str = "irls",
    cuda_device: str = "cuda",
    cuda_solver_backend: str = "normal-equation",
    cuda_cell_chunk_size: int = 2048,
    cuda_point_chunk_size: int = 2_000_000,
    profiler=None,
    profile_fields: Mapping[str, object] | None = None,
) -> dict:
    """Accumulate raw points first, then estimate geometry once at target pose."""
    if backend not in {"cpu", "cuda"}:
        raise ValueError("geometry backend must be 'cpu' or 'cuda'")
    if geometry_method not in {"irls", "elevation-grid"}:
        raise ValueError(
            "geometry method must be 'irls' or 'elevation-grid'"
        )
    if not np.isfinite(obstacle_height) or obstacle_height <= 0.0:
        raise ValueError("obstacle_height must be finite and positive")
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("geometry hindsight window contains no frames")
    class_count = first.observation_world.semantic_probabilities.shape[1]
    timestamps = []
    previous_timestamp = -np.inf

    def packed_frames():
        nonlocal previous_timestamp
        for frame in chain((first,), iterator):
            if frame.sequence_id != target.sequence_id or frame.split != target.split:
                raise ValueError("geometry frame group does not match target")
            if abs(frame.timestamp - target.timestamp) > half_window + 1e-6:
                raise ValueError("geometry frame lies outside hindsight window")
            if frame.timestamp <= previous_timestamp:
                raise ValueError("geometry frames must be time ordered")
            previous_timestamp = frame.timestamp
            if frame.observation_world.semantic_probabilities.shape[1] != class_count:
                raise ValueError("geometry frame semantic class counts differ")
            timestamps.append(frame.timestamp)
            yield frame.timestamp, frame.observation_world

    profile_fields = {} if profile_fields is None else dict(profile_fields)
    pack_stage = (
        profiler.stage("geometry_pack", **profile_fields)
        if profiler is not None
        else nullcontext()
    )
    with pack_stage:
        packed = pack_geometry_frames_batched(
            packed_frames(),
            target_world_from_body=target.world_from_body,
            map_spec=map_spec,
            voxel_size=map_spec.resolution,
            max_count=8,
            static_decay_tau=max(120.0, 4.0 * half_window),
            dynamic_decay_tau=1.0,
            max_geometry_age_sec=None,
        )
    actual_span = float(max(timestamps) - min(timestamps))
    snapshot_timestamp = float(max(timestamps))
    geometry_max_age_sec = max(actual_span + 1.0, 1.0)
    if geometry_method == "elevation-grid":
        fit_stage = (
            profiler.stage("geometry_fit_elevation_grid", **profile_fields)
            if profiler is not None
            else nullcontext()
        )
        with fit_stage:
            maps = fit_packed_geometry_elevation_grid(
                packed,
                max_geometry_age_sec=geometry_max_age_sec,
            )
    elif backend == "cuda":
        fit_stage = (
            profiler.stage("geometry_fit_cuda", **profile_fields)
            if profiler is not None
            else nullcontext()
        )
        with fit_stage:
            maps = fit_packed_geometry_cuda(
                packed,
                device=cuda_device,
                solver_backend=cuda_solver_backend,
                cell_chunk_size=cuda_cell_chunk_size,
                point_chunk_size=cuda_point_chunk_size,
                max_geometry_age_sec=geometry_max_age_sec,
                profiler=profiler,
                profile_fields=profile_fields,
            )
    else:
        fit_stage = (
            profiler.stage("geometry_fit_cpu", **profile_fields)
            if profiler is not None
            else nullcontext()
        )
        with fit_stage:
            maps = fit_packed_geometry_cpu(
                packed,
                max_geometry_age_sec=geometry_max_age_sec,
            )
    fallback_ratio = float(maps.get("cuda_fallback_ratio", 0.0))
    fallback_cells = np.asarray(
        maps.get("cuda_fallback_cell_ids", np.empty((0, 2), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1, 2)
    valid = np.asarray(maps["geometry_valid_mask"], dtype=bool)
    result = {
        "geometry_hindsight_target": np.asarray(
            maps["geometry_risk"], dtype=np.float32
        ),
        "geometry_hindsight_valid_mask": valid,
        "geometry_hindsight_confidence": np.asarray(
            maps["geometry_confidence"], dtype=np.float32
        ),
        "geometry_hindsight_observation_count": np.asarray(
            maps["geometry_support_count"], dtype=np.int32
        ),
        "geometry_hindsight_elevation": np.asarray(
            maps["elevation"], dtype=np.float32
        ),
        "geometry_hindsight_frame_count": len(timestamps),
        "geometry_hindsight_window_start": float(min(timestamps)),
        "geometry_hindsight_window_end": float(max(timestamps)),
        "geometry_hindsight_actual_span_sec": actual_span,
        "geometry_hindsight_snapshot_timestamp": snapshot_timestamp,
        "geometry_hindsight_backend": (
            "cpu" if geometry_method == "elevation-grid" else backend
        ),
        "geometry_hindsight_method": geometry_method,
        "geometry_hindsight_cuda_solver_backend": str(
            maps.get("cuda_solver_backend", "")
        ),
        "geometry_hindsight_cuda_fallback_cells": fallback_cells,
        "geometry_hindsight_cuda_fallback_ratio": fallback_ratio,
        "geometry_hindsight_cuda_cell_chunk_size": int(
            maps.get("cuda_cell_chunk_size", 0)
        ),
        "geometry_hindsight_cuda_point_chunk_size": int(
            maps.get("cuda_point_chunk_size", 0)
        ),
    }
    for name in (
        "slope",
        "longitudinal_slope",
        "lateral_slope",
        "roughness",
        "positive_step_height",
        "negative_step_height",
        "non_ground_ratio",
        "ground_fit_residual",
        "slope_risk",
        "roughness_risk",
        "positive_step_risk",
        "negative_step_risk",
        "geometry_age",
        "geometry_conflict_score",
        "ground_fit_valid",
    ):
        result[f"geometry_hindsight_{name}"] = np.asarray(maps[name])
    return result


def build_accumulated_geometry_teacher(
    frames: Iterable[AccumulatedGeometryFrame],
    target: AccumulatedGeometryFrame,
    map_spec: MapSpec,
    **options,
) -> dict:
    """Build from an in-memory frame iterable while enforcing the strict window."""

    half_window = float(options.get("half_window", 30.0))
    selected = _select_window(frames, target, half_window)
    return _build_accumulated_geometry_teacher_ordered(
        selected, target, map_spec, **options
    )


def build_accumulated_geometry_teacher_from_ids(
    frame_ids: Iterable[str],
    frame_loader: Callable[[str], AccumulatedGeometryFrame],
    target: AccumulatedGeometryFrame,
    map_spec: MapSpec,
    **options,
) -> dict:
    """Stream a strict geometry window without retaining frame point arrays."""

    frames = (frame_loader(frame_id) for frame_id in frame_ids)
    return _build_accumulated_geometry_teacher_ordered(
        frames, target, map_spec, **options
    )


def build_sliding_accumulated_geometry_teacher_from_ids(
    window: SlidingVoxelWindow,
    frame_ids: Iterable[str],
    frame_loader: Callable[[str], AccumulatedGeometryFrame],
    target: AccumulatedGeometryFrame,
    map_spec: MapSpec,
    *,
    half_window: float = 30.0,
    profiler=None,
    profile_fields: Mapping[str, object] | None = None,
) -> dict:
    """Advance a persistent voxel window and derive one elevation-grid teacher."""

    if not np.isfinite(half_window) or half_window < 0.0:
        raise ValueError("half_window must be finite and non-negative")
    profile_fields = {} if profile_fields is None else dict(profile_fields)

    def checked_loader(frame_id: str) -> AccumulatedGeometryFrame:
        frame = frame_loader(frame_id)
        if frame.sequence_id != target.sequence_id or frame.split != target.split:
            raise ValueError("geometry frame group does not match target")
        if abs(frame.timestamp - target.timestamp) > half_window + 1e-6:
            raise ValueError("geometry frame lies outside hindsight window")
        return frame

    update_stage = (
        profiler.stage("geometry_sliding_voxel_update", **profile_fields)
        if profiler is not None
        else nullcontext()
    )
    with update_stage:
        packed = window.advance(
            tuple(frame_ids),
            checked_loader,
            target_world_from_body=target.world_from_body,
            map_spec=map_spec,
        )
    timestamps = window.active_timestamps
    if not timestamps:
        raise ValueError("geometry hindsight window contains no frames")
    if any(
        abs(timestamp - target.timestamp) > half_window + 1e-6
        for timestamp in timestamps
    ):
        raise ValueError("retained geometry frame lies outside hindsight window")
    actual_span = float(max(timestamps) - min(timestamps))
    geometry_max_age_sec = max(actual_span + 1.0, 1.0)
    fit_stage = (
        profiler.stage("geometry_fit_elevation_grid", **profile_fields)
        if profiler is not None
        else nullcontext()
    )
    with fit_stage:
        maps = fit_packed_geometry_elevation_grid(
            packed,
            max_geometry_age_sec=geometry_max_age_sec,
        )
    valid = np.asarray(maps["geometry_valid_mask"], dtype=bool)
    result = {
        "geometry_hindsight_target": np.asarray(
            maps["geometry_risk"], dtype=np.float32
        ),
        "geometry_hindsight_valid_mask": valid,
        "geometry_hindsight_confidence": np.asarray(
            maps["geometry_confidence"], dtype=np.float32
        ),
        "geometry_hindsight_observation_count": np.asarray(
            maps["geometry_support_count"], dtype=np.int32
        ),
        "geometry_hindsight_elevation": np.asarray(
            maps["elevation"], dtype=np.float32
        ),
        "geometry_hindsight_frame_count": len(timestamps),
        "geometry_hindsight_window_start": float(min(timestamps)),
        "geometry_hindsight_window_end": float(max(timestamps)),
        "geometry_hindsight_actual_span_sec": actual_span,
        "geometry_hindsight_snapshot_timestamp": float(max(timestamps)),
        "geometry_hindsight_backend": "cpu",
        "geometry_hindsight_method": "sliding-voxel",
        "geometry_hindsight_cuda_solver_backend": "",
        "geometry_hindsight_cuda_fallback_cells": np.empty(
            (0, 2), dtype=np.int32
        ),
        "geometry_hindsight_cuda_fallback_ratio": 0.0,
        "geometry_hindsight_cuda_cell_chunk_size": 0,
        "geometry_hindsight_cuda_point_chunk_size": 0,
    }
    for name in (
        "slope",
        "longitudinal_slope",
        "lateral_slope",
        "roughness",
        "positive_step_height",
        "negative_step_height",
        "non_ground_ratio",
        "ground_fit_residual",
        "slope_risk",
        "roughness_risk",
        "positive_step_risk",
        "negative_step_risk",
        "geometry_age",
        "geometry_conflict_score",
        "ground_fit_valid",
    ):
        result[f"geometry_hindsight_{name}"] = np.asarray(maps[name])
    return result


def build_gpu_rolling_geometry_teacher_from_ids(
    rolling_map: GpuRollingElevationMap,
    frame_ids: Iterable[str],
    frame_loader: Callable[[str], AccumulatedGeometryFrame],
    target: AccumulatedGeometryFrame,
    map_spec: MapSpec,
    *,
    half_window: float = 30.0,
    profiler=None,
    profile_fields: Mapping[str, object] | None = None,
) -> dict:
    """Add unseen frames once and render a persistent GPU elevation teacher."""

    if rolling_map.map_spec != map_spec:
        raise ValueError("GPU rolling map specification does not match target")
    if not np.isfinite(half_window) or half_window < 0.0:
        raise ValueError("half_window must be finite and non-negative")
    profile_fields = {} if profile_fields is None else dict(profile_fields)

    def checked_loader(frame_id: str) -> AccumulatedGeometryFrame:
        frame = frame_loader(frame_id)
        if frame.sequence_id != target.sequence_id or frame.split != target.split:
            raise ValueError("geometry frame group does not match target")
        if abs(frame.timestamp - target.timestamp) > half_window + 1e-6:
            raise ValueError("geometry frame lies outside current hindsight window")
        return frame

    stage = (
        profiler.stage("geometry_gpu_rolling_grid", **profile_fields)
        if profiler is not None
        else nullcontext()
    )
    with stage:
        maps = rolling_map.advance(
            tuple(frame_ids),
            checked_loader,
            target_world_from_body=target.world_from_body,
        )
    timestamps = rolling_map.ingested_timestamps
    if not timestamps:
        raise ValueError("GPU rolling geometry map contains no frames")
    valid = np.asarray(maps["geometry_valid_mask"], dtype=bool)
    result = {
        "geometry_hindsight_target": np.asarray(
            maps["geometry_risk"], dtype=np.float32
        ),
        "geometry_hindsight_valid_mask": valid,
        "geometry_hindsight_confidence": np.asarray(
            maps["geometry_confidence"], dtype=np.float32
        ),
        "geometry_hindsight_observation_count": np.asarray(
            maps["geometry_support_count"], dtype=np.int32
        ),
        "geometry_hindsight_elevation": np.asarray(
            maps["elevation"], dtype=np.float32
        ),
        "geometry_hindsight_frame_count": len(timestamps),
        "geometry_hindsight_window_start": float(min(timestamps)),
        "geometry_hindsight_window_end": float(max(timestamps)),
        "geometry_hindsight_actual_span_sec": float(
            max(timestamps) - min(timestamps)
        ),
        "geometry_hindsight_snapshot_timestamp": float(max(timestamps)),
        "geometry_hindsight_backend": "cuda",
        "geometry_hindsight_method": "gpu-rolling-grid",
        "geometry_hindsight_cuda_solver_backend": "cupy",
        "geometry_hindsight_cuda_fallback_cells": np.empty(
            (0, 2), dtype=np.int32
        ),
        "geometry_hindsight_cuda_fallback_ratio": 0.0,
        "geometry_hindsight_cuda_cell_chunk_size": 0,
        "geometry_hindsight_cuda_point_chunk_size": 0,
    }
    for name in (
        "slope",
        "longitudinal_slope",
        "lateral_slope",
        "roughness",
        "positive_step_height",
        "negative_step_height",
        "non_ground_ratio",
        "ground_fit_residual",
        "slope_risk",
        "roughness_risk",
        "positive_step_risk",
        "negative_step_risk",
        "geometry_age",
        "geometry_conflict_score",
        "ground_fit_valid",
    ):
        result[f"geometry_hindsight_{name}"] = np.asarray(maps[name])
    return result


def apply_grass_rule_to_accumulated_geometry(
    geometry: Mapping[str, np.ndarray],
    semantic: Mapping[str, np.ndarray],
    *,
    config: GrassGeometryConfig = DEFAULT_GRASS_GEOMETRY_CONFIG,
) -> dict:
    """Apply the shared grass rule to either CPU or CUDA geometry output."""

    grass = apply_grass_geometry_risk(
        grass_mask=semantic["grass_mask"],
        slope_risk=geometry["geometry_hindsight_slope_risk"],
        positive_step_risk=geometry[
            "geometry_hindsight_positive_step_risk"
        ],
        negative_step_risk=geometry[
            "geometry_hindsight_negative_step_risk"
        ],
        non_ground_ratio=geometry[
            "geometry_hindsight_non_ground_ratio"
        ],
        ground_confidence=geometry["geometry_hindsight_confidence"],
        ground_support_count=geometry[
            "geometry_hindsight_observation_count"
        ],
        config=config,
    )
    result = dict(geometry)
    result["geometry_hindsight_target"] = grass.pop("geometry_risk")
    for name in (
        "grass_probability",
        "grass_observation_weight",
        "grass_mask",
    ):
        result[name] = np.asarray(semantic[name])
    result.update(grass)
    return result


def geometry_cvar_risk(
    geometry: Mapping[str, np.ndarray],
    *,
    use_cvar: bool = True,
) -> dict:
    """Construct only the conservative geometry-risk branch."""

    cvar = dict(geometry)
    legacy_risk = np.asarray(
        geometry["geometry_hindsight_target"], dtype=np.float32
    )
    cvar_risk = np.asarray(
        geometry.get("geometry_hindsight_risk_cvar", legacy_risk),
        dtype=np.float32,
    )
    distribution_valid = np.asarray(
        geometry.get(
            "geometry_hindsight_risk_distribution_valid",
            geometry["geometry_hindsight_valid_mask"],
        ),
        dtype=bool,
    )
    cvar["geometry_hindsight_target"] = np.where(
        use_cvar & distribution_valid,
        np.maximum(legacy_risk, cvar_risk),
        legacy_risk,
    ).astype(np.float32)
    return cvar


def geometry_risk_variants(
    geometry: Mapping[str, np.ndarray],
    *,
    use_cvar: bool = True,
) -> tuple[dict, dict]:
    """Return independent legacy and conservative geometry-risk branches.

    The legacy branch is unchanged.  The CVaR branch uses the accumulated
    upper-tail estimate when it is available, but never lowers the legacy
    risk (including grass-rule priors or other hard geometry rules).
    """

    legacy = dict(geometry)
    return legacy, geometry_cvar_risk(geometry, use_cvar=use_cvar)


def semantic_cvar_risk(
    semantic: Mapping[str, np.ndarray],
    *,
    use_cvar: bool = True,
) -> dict:
    """Construct only the conservative semantic-risk branch."""

    cvar = dict(semantic)
    legacy_risk = np.asarray(
        semantic["semantic_hindsight_target"], dtype=np.float32
    )
    cvar_risk = np.asarray(
        semantic.get("semantic_hindsight_risk_cvar", legacy_risk),
        dtype=np.float32,
    )
    distribution_valid = np.asarray(
        semantic.get(
            "semantic_hindsight_risk_distribution_valid",
            semantic["semantic_hindsight_valid_mask"],
        ),
        dtype=bool,
    )
    cvar["semantic_hindsight_target"] = np.where(
        use_cvar & distribution_valid,
        np.maximum(legacy_risk, cvar_risk),
        legacy_risk,
    ).astype(np.float32)
    return cvar


def semantic_risk_variants(
    semantic: Mapping[str, np.ndarray],
    *,
    use_cvar: bool = True,
) -> tuple[dict, dict]:
    """Return independent legacy and conservative semantic-risk branches."""

    legacy = dict(semantic)
    return legacy, semantic_cvar_risk(semantic, use_cvar=use_cvar)


def fuse_accumulated_geometry_semantics(
    geometry: Mapping[str, np.ndarray],
    semantic: Mapping[str, np.ndarray],
) -> dict:
    """Fuse independent accumulated geometry and semantic teachers."""
    geometry_risk = np.asarray(geometry["geometry_hindsight_target"], dtype=np.float32)
    semantic_risk = np.asarray(semantic["semantic_hindsight_target"], dtype=np.float32)
    geometry_confidence = np.asarray(
        geometry["geometry_hindsight_confidence"], dtype=np.float32
    )
    semantic_confidence = np.asarray(
        semantic["semantic_hindsight_confidence"], dtype=np.float32
    )
    geometry_valid = np.asarray(
        geometry["geometry_hindsight_valid_mask"], dtype=bool
    )
    semantic_valid = np.asarray(
        semantic["semantic_hindsight_valid_mask"], dtype=bool
    )
    shape = geometry_risk.shape
    arrays = (
        semantic_risk,
        geometry_confidence,
        semantic_confidence,
        geometry_valid,
        semantic_valid,
    )
    if len(shape) != 2 or any(array.shape != shape for array in arrays):
        raise ValueError("accumulated teacher maps must share a 2D shape")

    both = geometry_valid & semantic_valid
    geometry_only = geometry_valid & ~semantic_valid
    semantic_only = semantic_valid & ~geometry_valid
    risk = np.zeros(shape, dtype=np.float32)
    risk[geometry_only] = geometry_risk[geometry_only]
    risk[semantic_only] = semantic_risk[semantic_only]
    denominator = geometry_confidence + semantic_confidence
    alpha = np.zeros(shape, dtype=np.float32)
    usable_both = both & (denominator > 0.0)
    alpha[usable_both] = geometry_confidence[usable_both] / denominator[usable_both]
    risk[usable_both] = (
        alpha[usable_both] * geometry_risk[usable_both]
        + (1.0 - alpha[usable_both]) * semantic_risk[usable_both]
    )
    risk[both & ~usable_both] = np.maximum(
        geometry_risk[both & ~usable_both], semantic_risk[both & ~usable_both]
    )
    hard_semantic = semantic_valid & (semantic_risk >= 0.95)
    risk[hard_semantic] = semantic_risk[hard_semantic]
    valid = geometry_valid | semantic_valid
    reliability = np.maximum(
        np.where(geometry_valid, geometry_confidence, 0.0),
        np.where(semantic_valid, semantic_confidence, 0.0),
    ).astype(np.float32)
    geometry_count = np.asarray(
        geometry["geometry_hindsight_observation_count"], dtype=np.int32
    )
    semantic_count = np.asarray(
        semantic["semantic_hindsight_observation_count"], dtype=np.int32
    )
    return {
        "traversability_risk": np.clip(risk, 0.0, 1.0),
        "valid_mask": valid,
        "reliability_map": reliability,
        "hindsight_observation_count": np.maximum(
            geometry_count, semantic_count
        ).astype(np.int32),
    }
