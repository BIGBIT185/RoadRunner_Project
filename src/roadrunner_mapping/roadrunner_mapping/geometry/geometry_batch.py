"""Vectorized accumulation for fixed-window geometry teachers."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .coordinates import body_xy_to_grid, transform_points, validate_transform
from .geometry_snapshot import PackedGeometrySnapshot
from .pointcloud import PointObservation
from .schema import MapSpec


_KEY_BITS = 20
_KEY_BIAS = 1 << (_KEY_BITS - 1)
_KEY_MASK = (1 << _KEY_BITS) - 1


def _encode_voxel_keys(keys: np.ndarray) -> np.ndarray:
    shifted = np.asarray(keys, dtype=np.int64) + _KEY_BIAS
    if np.any(shifted < 0) or np.any(shifted > _KEY_MASK):
        raise ValueError("voxel coordinates exceed the 20-bit batch key range")
    return (
        (shifted[:, 0] << (2 * _KEY_BITS))
        | (shifted[:, 1] << _KEY_BITS)
        | shifted[:, 2]
    )


def _decode_voxel_keys(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.int64)
    return np.column_stack(
        (
            (encoded >> (2 * _KEY_BITS)) - _KEY_BIAS,
            ((encoded >> _KEY_BITS) & _KEY_MASK) - _KEY_BIAS,
            (encoded & _KEY_MASK) - _KEY_BIAS,
        )
    ).astype(np.int64, copy=False)


def pack_geometry_frames_batched(
    frames: Iterable[tuple[float, PointObservation]],
    *,
    target_world_from_body: np.ndarray,
    map_spec: MapSpec,
    voxel_size: float,
    max_count: int,
    static_decay_tau: float,
    dynamic_decay_tau: float,
    max_geometry_age_sec: Optional[float],
    min_geometry_reliability: float = 0.10,
    min_observation_count: int = 1,
    min_elevation_m: float = -2.0,
    max_elevation_m: float = 3.0,
) -> PackedGeometrySnapshot:
    """Pack a complete hindsight window without per-voxel Python updates."""

    if voxel_size <= 0.0 or max_count <= 0:
        raise ValueError("voxel_size and max_count must be positive")

    encoded_chunks = []
    elevation_chunks = []
    ground_chunks = []
    dynamic_chunks = []
    timestamp_chunks = []
    timestamps = []
    previous_timestamp = -np.inf
    for timestamp, observation in frames:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp <= previous_timestamp:
            raise ValueError("geometry frames must have finite increasing timestamps")
        previous_timestamp = timestamp
        timestamps.append(timestamp)
        if observation.xyz.size == 0:
            continue
        keys = np.floor(observation.xyz / voxel_size).astype(np.int64)
        encoded_chunks.append(_encode_voxel_keys(keys))
        elevation_chunks.append(observation.xyz[:, 2])
        ground_chunks.append(observation.ground_probability)
        dynamic_chunks.append(observation.dynamic.astype(np.float64))
        timestamp_chunks.append(
            np.full(observation.xyz.shape[0], timestamp, dtype=np.float64)
        )
    if not timestamps:
        raise ValueError("geometry hindsight window contains no frames")
    if not encoded_chunks:
        raise ValueError("geometry hindsight window contains no points")

    encoded = np.concatenate(encoded_chunks)
    elevation_values = np.concatenate(elevation_chunks)
    ground_values = np.concatenate(ground_chunks)
    dynamic_values = np.concatenate(dynamic_chunks)
    point_timestamps = np.concatenate(timestamp_chunks)
    unique, inverse, counts = np.unique(
        encoded, return_inverse=True, return_counts=True
    )
    counts_float = counts.astype(np.float64)
    mean_elevation = (
        np.bincount(inverse, weights=elevation_values) / counts_float
    )
    mean_ground = np.bincount(inverse, weights=ground_values) / counts_float
    dynamic = np.bincount(inverse, weights=dynamic_values) > 0.0
    last_observed = np.full(unique.shape, -np.inf, dtype=np.float64)
    np.maximum.at(last_observed, inverse, point_timestamps)

    keys = _decode_voxel_keys(unique)
    centers_world = (keys.astype(np.float64) + 0.5) * voxel_size
    centers_world[:, 2] = mean_elevation
    target_from_world = np.linalg.inv(validate_transform(target_world_from_body))
    centers_body = transform_points(centers_world, target_from_world)
    rows, columns, inside = body_xy_to_grid(centers_body[:, :2], map_spec)
    inside &= (centers_body[:, 2] >= min_elevation_m) & (
        centers_body[:, 2] <= max_elevation_m
    )

    centers_body = centers_body[inside]
    rows = rows[inside]
    columns = columns[inside]
    counts = counts[inside]
    mean_ground = mean_ground[inside]
    dynamic = dynamic[inside]
    last_observed = last_observed[inside]
    timestamp = float(timestamps[-1])
    if max_geometry_age_sec is None:
        max_geometry_age_sec = max(timestamp - float(timestamps[0]) + 1.0, 1.0)
    if not np.isfinite(max_geometry_age_sec) or max_geometry_age_sec <= 0.0:
        raise ValueError("max_geometry_age_sec must be finite and positive")
    age = np.maximum(0.0, timestamp - last_observed)
    tau = np.where(dynamic, dynamic_decay_tau, static_decay_tau)
    reliability = np.minimum(counts / float(max_count), 1.0) * np.exp(-age / tau)

    height, width = map_spec.shape
    shape = (height, width)
    flat = rows.astype(np.int64) * width + columns.astype(np.int64)
    observation_count = np.zeros(height * width, dtype=np.int32)
    np.add.at(observation_count, flat, counts.astype(np.int32))
    elevation = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(elevation, flat, centers_body[:, 2])
    cell_last_observed = np.full(height * width, -1.0, dtype=np.float64)
    np.maximum.at(cell_last_observed, flat, last_observed)

    support = (
        (reliability >= min_geometry_reliability)
        & (age <= max_geometry_age_sec)
        & ~dynamic
        & (counts >= min_observation_count)
    )
    support_flat = flat[support]
    support_points = centers_body[support]
    support_weights = (
        reliability[support]
        * np.exp(-age[support] / max(max_geometry_age_sec, 1e-6))
        * np.maximum(mean_ground[support], 0.05)
        * np.maximum(counts[support], 1.0)
    )
    order = np.argsort(support_flat, kind="stable")
    support_points = support_points[order]
    support_weights = support_weights[order]
    support_per_cell = np.bincount(support_flat, minlength=height * width)
    cell_offsets = np.zeros(height * width + 1, dtype=np.int64)
    cell_offsets[1:] = np.cumsum(support_per_cell, dtype=np.int64)

    observed = observation_count > 0
    elevation_layer = np.zeros(height * width, dtype=np.float32)
    elevation_layer[observed] = elevation[observed].astype(np.float32)
    geometry_age = np.zeros(height * width, dtype=np.float32)
    geometry_age[observed] = (
        timestamp - cell_last_observed[observed]
    ).clip(min=0.0).astype(np.float32)
    return PackedGeometrySnapshot(
        timestamp=timestamp,
        map_spec=map_spec,
        elevation=elevation_layer.reshape(shape),
        observation_count=observation_count.reshape(shape),
        last_observed_time=cell_last_observed.reshape(shape),
        geometry_age=geometry_age.reshape(shape),
        conflict_score=np.zeros(shape, dtype=np.float32),
        semantic_sum=np.zeros((height, width, 1), dtype=np.float64),
        semantic_weight=np.zeros(shape, dtype=np.float64),
        cell_offsets=cell_offsets,
        support_points=support_points,
        support_weights=support_weights,
    )
