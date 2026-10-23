"""Canonical body, world and BEV coordinate transforms."""

from __future__ import annotations

import numpy as np

from .schema import MapSpec


def body_xy_to_grid(
    xy: np.ndarray,
    map_spec: MapSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"xy must be [N,2], got {points.shape}")

    rows = np.floor((map_spec.x_max - points[:, 0]) / map_spec.resolution).astype(np.int64)
    columns = np.floor((map_spec.y_max - points[:, 1]) / map_spec.resolution).astype(np.int64)
    height, width = map_spec.shape
    valid = (
        (rows >= 0)
        & (rows < height)
        & (columns >= 0)
        & (columns < width)
    )
    return rows, columns, valid


def grid_to_body_xy(
    rows: np.ndarray,
    columns: np.ndarray,
    map_spec: MapSpec,
) -> np.ndarray:
    row_values = np.asarray(rows, dtype=np.float64)
    column_values = np.asarray(columns, dtype=np.float64)
    if row_values.shape != column_values.shape:
        raise ValueError("rows and columns must have matching shapes")
    x = map_spec.x_max - (row_values + 0.5) * map_spec.resolution
    y = map_spec.y_max - (column_values + 0.5) * map_spec.resolution
    return np.stack((x, y), axis=-1)


def validate_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("transform has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("transform rotation determinant is not one")
    return matrix


def relative_transform(
    world_from_target: np.ndarray,
    world_from_source: np.ndarray,
) -> np.ndarray:
    target = validate_transform(world_from_target)
    source = validate_transform(world_from_source)
    return np.linalg.inv(target) @ source


def transform_points(points: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"points must be [N,C>=3], got {values.shape}")
    transform = validate_transform(target_from_source)
    xyz_h = np.concatenate((values[:, :3], np.ones((values.shape[0], 1))), axis=1)
    transformed = (transform @ xyz_h.T).T[:, :3]
    if values.shape[1] == 3:
        return transformed
    return np.concatenate((transformed, values[:, 3:]), axis=1)
