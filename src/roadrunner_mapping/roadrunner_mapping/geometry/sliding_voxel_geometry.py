"""Incremental fixed-window voxel state for GOOSE hindsight geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .coordinates import body_xy_to_grid, transform_points, validate_transform
from .geometry_batch import _decode_voxel_keys, _encode_voxel_keys
from .geometry_snapshot import PackedGeometrySnapshot
from .schema import MapSpec


@dataclass(frozen=True)
class FrameVoxelSummary:
    frame_id: str
    timestamp: float
    encoded_keys: np.ndarray
    counts: np.ndarray
    elevation_sums: np.ndarray
    ground_sums: np.ndarray
    dynamic_counts: np.ndarray


class _VoxelState:
    __slots__ = (
        "count",
        "elevation_sum",
        "ground_sum",
        "dynamic_count",
        "last_observed",
    )

    def __init__(
        self,
        count: int,
        elevation_sum: float,
        ground_sum: float,
        dynamic_count: int,
        last_observed: float,
    ) -> None:
        self.count = int(count)
        self.elevation_sum = float(elevation_sum)
        self.ground_sum = float(ground_sum)
        self.dynamic_count = int(dynamic_count)
        self.last_observed = float(last_observed)


def summarize_frame_voxels(
    frame_id: str,
    frame,
    *,
    voxel_size: float,
) -> FrameVoxelSummary:
    """Collapse one world-frame cloud into removable 3D voxel contributions."""

    if not frame_id:
        raise ValueError("sliding voxel frame_id is required")
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    timestamp = float(frame.timestamp)
    if not np.isfinite(timestamp):
        raise ValueError("sliding voxel timestamp must be finite")
    observation = frame.observation_world
    xyz = np.asarray(observation.xyz, dtype=np.float64)
    if xyz.size == 0:
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float64)
        return FrameVoxelSummary(
            frame_id=frame_id,
            timestamp=timestamp,
            encoded_keys=empty_int,
            counts=empty_int,
            elevation_sums=empty_float,
            ground_sums=empty_float,
            dynamic_counts=empty_int,
        )
    encoded = _encode_voxel_keys(
        np.floor(xyz / float(voxel_size)).astype(np.int64)
    )
    unique, inverse, counts = np.unique(
        encoded,
        return_inverse=True,
        return_counts=True,
    )
    return FrameVoxelSummary(
        frame_id=frame_id,
        timestamp=timestamp,
        encoded_keys=unique,
        counts=counts.astype(np.int64, copy=False),
        elevation_sums=np.bincount(
            inverse,
            weights=xyz[:, 2],
        ).astype(np.float64, copy=False),
        ground_sums=np.bincount(
            inverse,
            weights=np.asarray(
                observation.ground_probability,
                dtype=np.float64,
            ),
        ).astype(np.float64, copy=False),
        dynamic_counts=np.bincount(
            inverse,
            weights=np.asarray(observation.dynamic, dtype=np.int64),
        ).astype(np.int64, copy=False),
    )


class SlidingVoxelWindow:
    """Maintain exact voxel sufficient statistics for a monotonic time window."""

    def __init__(
        self,
        *,
        voxel_size: float,
        max_count: int = 8,
        static_decay_tau: float = 120.0,
        dynamic_decay_tau: float = 1.0,
        min_geometry_reliability: float = 0.10,
        min_observation_count: int = 1,
        min_elevation_m: float = -2.0,
        max_elevation_m: float = 3.0,
    ) -> None:
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxel_size must be finite and positive")
        if max_count <= 0 or min_observation_count <= 0:
            raise ValueError("voxel count thresholds must be positive")
        self.voxel_size = float(voxel_size)
        self.max_count = int(max_count)
        self.static_decay_tau = float(static_decay_tau)
        self.dynamic_decay_tau = float(dynamic_decay_tau)
        self.min_geometry_reliability = float(min_geometry_reliability)
        self.min_observation_count = int(min_observation_count)
        self.min_elevation_m = float(min_elevation_m)
        self.max_elevation_m = float(max_elevation_m)
        self._frames: dict[str, FrameVoxelSummary] = {}
        self._active_frame_ids: tuple[str, ...] = ()
        self._voxels: dict[int, _VoxelState] = {}

    @property
    def active_frame_ids(self) -> tuple[str, ...]:
        return self._active_frame_ids

    @property
    def active_timestamps(self) -> tuple[float, ...]:
        return tuple(
            self._frames[frame_id].timestamp
            for frame_id in self._active_frame_ids
        )

    def clear(self) -> None:
        self._frames.clear()
        self._active_frame_ids = ()
        self._voxels.clear()

    def _add(self, summary: FrameVoxelSummary) -> None:
        for key, count, elevation_sum, ground_sum, dynamic_count in zip(
            summary.encoded_keys,
            summary.counts,
            summary.elevation_sums,
            summary.ground_sums,
            summary.dynamic_counts,
        ):
            encoded = int(key)
            state = self._voxels.get(encoded)
            if state is None:
                self._voxels[encoded] = _VoxelState(
                    int(count),
                    float(elevation_sum),
                    float(ground_sum),
                    int(dynamic_count),
                    summary.timestamp,
                )
                continue
            state.count += int(count)
            state.elevation_sum += float(elevation_sum)
            state.ground_sum += float(ground_sum)
            state.dynamic_count += int(dynamic_count)
            state.last_observed = max(
                state.last_observed,
                summary.timestamp,
            )
        self._frames[summary.frame_id] = summary

    def _remove(self, frame_id: str) -> None:
        summary = self._frames.pop(frame_id)
        for key, count, elevation_sum, ground_sum, dynamic_count in zip(
            summary.encoded_keys,
            summary.counts,
            summary.elevation_sums,
            summary.ground_sums,
            summary.dynamic_counts,
        ):
            encoded = int(key)
            state = self._voxels[encoded]
            state.count -= int(count)
            state.elevation_sum -= float(elevation_sum)
            state.ground_sum -= float(ground_sum)
            state.dynamic_count -= int(dynamic_count)
            if state.count < 0 or state.dynamic_count < 0:
                raise RuntimeError("sliding voxel subtraction underflow")
            if state.count == 0:
                del self._voxels[encoded]

    def _update_frames(
        self,
        frame_ids: tuple[str, ...],
        frame_loader: Callable[[str], object],
    ) -> None:
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("sliding voxel frame ids must be unique")
        retained = tuple(
            frame_id
            for frame_id in self._active_frame_ids
            if frame_id in set(frame_ids)
        )
        if retained and (
            retained != self._active_frame_ids[-len(retained) :]
            or retained != frame_ids[: len(retained)]
        ):
            raise ValueError("sliding voxel windows must advance monotonically")
        for frame_id in self._active_frame_ids[: len(self._active_frame_ids) - len(retained)]:
            self._remove(frame_id)
        last_timestamp = max(
            (summary.timestamp for summary in self._frames.values()),
            default=-np.inf,
        )
        for frame_id in frame_ids[len(retained) :]:
            frame = frame_loader(frame_id)
            summary = summarize_frame_voxels(
                frame_id,
                frame,
                voxel_size=self.voxel_size,
            )
            if summary.timestamp <= last_timestamp:
                raise ValueError(
                    "sliding voxel entering frames must be time ordered"
                )
            self._add(summary)
            last_timestamp = summary.timestamp
        self._active_frame_ids = frame_ids

    def advance(
        self,
        frame_ids: Iterable[str],
        frame_loader: Callable[[str], object],
        *,
        target_world_from_body: np.ndarray,
        map_spec: MapSpec,
    ) -> PackedGeometrySnapshot:
        desired = tuple(frame_ids)
        if not desired:
            raise ValueError("sliding voxel window contains no frames")
        self._update_frames(desired, frame_loader)
        return self.snapshot(
            target_world_from_body=target_world_from_body,
            map_spec=map_spec,
        )

    def snapshot(
        self,
        *,
        target_world_from_body: np.ndarray,
        map_spec: MapSpec,
    ) -> PackedGeometrySnapshot:
        if not self._frames or not self._voxels:
            raise ValueError("cannot snapshot an empty sliding voxel window")
        encoded = np.asarray(sorted(self._voxels), dtype=np.int64)
        states = [self._voxels[int(key)] for key in encoded]
        counts = np.asarray([state.count for state in states], dtype=np.int64)
        elevation_sum = np.asarray(
            [state.elevation_sum for state in states],
            dtype=np.float64,
        )
        ground_sum = np.asarray(
            [state.ground_sum for state in states],
            dtype=np.float64,
        )
        dynamic = np.asarray(
            [state.dynamic_count > 0 for state in states],
            dtype=bool,
        )
        last_observed = np.asarray(
            [state.last_observed for state in states],
            dtype=np.float64,
        )
        timestamp = max(
            summary.timestamp for summary in self._frames.values()
        )
        first_timestamp = min(
            summary.timestamp for summary in self._frames.values()
        )
        max_geometry_age_sec = max(
            timestamp - first_timestamp + 1.0,
            1.0,
        )
        keys = _decode_voxel_keys(encoded)
        centers_world = (
            keys.astype(np.float64) + 0.5
        ) * self.voxel_size
        centers_world[:, 2] = elevation_sum / counts
        target_from_world = np.linalg.inv(
            validate_transform(target_world_from_body)
        )
        centers_body = transform_points(centers_world, target_from_world)
        rows, columns, inside = body_xy_to_grid(
            centers_body[:, :2],
            map_spec,
        )
        inside &= (
            (centers_body[:, 2] >= self.min_elevation_m)
            & (centers_body[:, 2] <= self.max_elevation_m)
        )
        centers_body = centers_body[inside]
        rows = rows[inside]
        columns = columns[inside]
        counts = counts[inside]
        ground_sum = ground_sum[inside]
        dynamic = dynamic[inside]
        last_observed = last_observed[inside]
        age = np.maximum(0.0, timestamp - last_observed)
        tau = np.where(
            dynamic,
            self.dynamic_decay_tau,
            self.static_decay_tau,
        )
        reliability = np.minimum(
            counts / float(self.max_count),
            1.0,
        ) * np.exp(-age / tau)

        height, width = map_spec.shape
        flat_size = height * width
        flat = rows.astype(np.int64) * width + columns.astype(np.int64)
        observation_count = np.zeros(flat_size, dtype=np.int32)
        np.add.at(observation_count, flat, counts.astype(np.int32))
        elevation = np.full(flat_size, np.inf, dtype=np.float64)
        np.minimum.at(elevation, flat, centers_body[:, 2])
        cell_last_observed = np.full(flat_size, -1.0, dtype=np.float64)
        np.maximum.at(cell_last_observed, flat, last_observed)

        support = (
            (reliability >= self.min_geometry_reliability)
            & (age <= max_geometry_age_sec)
            & ~dynamic
            & (counts >= self.min_observation_count)
        )
        support_flat = flat[support]
        support_points = centers_body[support]
        mean_ground = ground_sum / counts
        support_weights = (
            reliability[support]
            * np.exp(-age[support] / max_geometry_age_sec)
            * np.maximum(mean_ground[support], 0.05)
            * np.maximum(counts[support], 1.0)
        )
        order = np.argsort(support_flat, kind="stable")
        support_points = support_points[order]
        support_weights = support_weights[order]
        support_per_cell = np.bincount(
            support_flat,
            minlength=flat_size,
        )
        cell_offsets = np.zeros(flat_size + 1, dtype=np.int64)
        cell_offsets[1:] = np.cumsum(
            support_per_cell,
            dtype=np.int64,
        )

        observed = observation_count > 0
        elevation_layer = np.zeros(flat_size, dtype=np.float32)
        elevation_layer[observed] = elevation[observed].astype(np.float32)
        geometry_age = np.zeros(flat_size, dtype=np.float32)
        geometry_age[observed] = np.maximum(
            0.0,
            timestamp - cell_last_observed[observed],
        ).astype(np.float32)
        shape = map_spec.shape
        return PackedGeometrySnapshot(
            timestamp=float(timestamp),
            map_spec=map_spec,
            elevation=elevation_layer.reshape(shape),
            observation_count=observation_count.reshape(shape),
            last_observed_time=cell_last_observed.reshape(shape),
            geometry_age=geometry_age.reshape(shape),
            conflict_score=np.zeros(shape, dtype=np.float32),
            semantic_sum=np.zeros((*shape, 1), dtype=np.float64),
            semantic_weight=np.zeros(shape, dtype=np.float64),
            cell_offsets=cell_offsets,
            support_points=support_points,
            support_weights=support_weights,
        )
