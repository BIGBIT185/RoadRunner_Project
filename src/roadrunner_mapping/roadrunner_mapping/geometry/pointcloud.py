"""Validated point observations used by the causal voxel teacher."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 轨迹实现归入 sensors；这里仅复用其内部四元数转换工具，不改变点云逻辑。
from ..sensors.trajectory import _matrix_quaternion
from .coordinates import transform_points, validate_transform


_POINT_FIELD_FORMATS = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}


def _pointcloud2_structured(message, *, fields):
    requested = tuple(fields)
    available = {field.name: field for field in message.fields}
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {missing}")
    byte_order = ">" if message.is_bigendian else "<"
    formats = []
    offsets = []
    for name in requested:
        field = available[name]
        if field.count != 1 or field.datatype not in _POINT_FIELD_FORMATS:
            raise ValueError(f"unsupported PointCloud2 field: {name}")
        formats.append(byte_order + _POINT_FIELD_FORMATS[field.datatype])
        offsets.append(int(field.offset))
    dtype = np.dtype(
        {
            "names": list(requested),
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(message.point_step),
        }
    )
    height = int(message.height)
    width = int(message.width)
    point_step = int(message.point_step)
    row_step = int(message.row_step)
    if height < 0 or width < 0 or point_step <= 0 or row_step < width * point_step:
        raise ValueError("invalid PointCloud2 dimensions or strides")
    required_bytes = height * row_step
    if len(message.data) < required_bytes:
        raise ValueError("PointCloud2 data is shorter than height * row_step")
    rows = [
        np.frombuffer(
            message.data,
            dtype=dtype,
            count=width,
            offset=row * row_step,
        )
        for row in range(height)
    ]
    structured = np.concatenate(rows) if rows else np.empty(0, dtype=dtype)
    row_indices = np.repeat(np.arange(height, dtype=np.int32), width)
    return structured, row_indices


def pointcloud2_to_array(message, *, fields) -> np.ndarray:
    structured, _ = _pointcloud2_structured(message, fields=fields)
    return np.column_stack(
        [structured[name].astype(np.float64, copy=False) for name in tuple(fields)]
    )


def pointcloud2_to_array_with_rows(message, *, fields) -> tuple[np.ndarray, np.ndarray]:
    """Decode fields and retain the organized PointCloud2 row for each point."""

    structured, row_indices = _pointcloud2_structured(message, fields=fields)
    values = np.column_stack(
        [structured[name].astype(np.float64, copy=False) for name in tuple(fields)]
    )
    return values, row_indices


def scan_row_relative_times(
    height: int,
    *,
    scan_period_sec: float,
    stamp_reference: str = "end",
    direction: int = 1,
) -> np.ndarray:
    """Return model-based VLS row offsets when ROS has no per-point time field.

    GOOSE VLS128 bags expose an organized cloud whose rows follow the horizontal
    firing order, but only the whole-cloud stamp is available.  This helper does
    not invent a timestamp per point; it makes that explicit as a configurable
    scan model and is only used when the online node opts into deskew.
    """

    height = int(height)
    scan_period_sec = float(scan_period_sec)
    if height <= 0:
        raise ValueError("organized cloud height must be positive")
    if not np.isfinite(scan_period_sec) or scan_period_sec <= 0.0:
        raise ValueError("scan period must be finite and positive")
    if stamp_reference not in {"start", "center", "end"}:
        raise ValueError("stamp_reference must be start, center or end")
    if int(direction) not in {-1, 1}:
        raise ValueError("scan row direction must be -1 or 1")
    if height == 1:
        return np.zeros(1, dtype=np.float64)
    fraction = np.linspace(0.0, 1.0, height, dtype=np.float64)
    if stamp_reference == "start":
        offsets = fraction
    elif stamp_reference == "center":
        offsets = fraction - 0.5
    else:
        offsets = fraction - 1.0
    return offsets * scan_period_sec * int(direction)


def interpolate_scan_transforms(
    start_transform: np.ndarray,
    end_transform: np.ndarray,
    ratios,
) -> np.ndarray:
    """Interpolate a short SE(3) scan trajectory with translation + SLERP."""

    start = validate_transform(np.asarray(start_transform, dtype=np.float64))
    end = validate_transform(np.asarray(end_transform, dtype=np.float64))
    values = np.asarray(ratios, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("scan interpolation ratios must be finite")
    values = np.clip(values, 0.0, 1.0)
    q0 = _matrix_quaternion(start[:3, :3])
    q1 = _matrix_quaternion(end[:3, :3])
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        quaternions = q0[None, :] + values[:, None] * (q1 - q0)[None, :]
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    else:
        angle = float(np.arccos(dot))
        sine = float(np.sin(angle))
        first = np.sin((1.0 - values) * angle) / sine
        second = np.sin(values * angle) / sine
        quaternions = first[:, None] * q0[None, :] + second[:, None] * q1[None, :]
    x, y, z, w = quaternions.T
    result = np.zeros((values.size, 4, 4), dtype=np.float64)
    result[:, 3, 3] = 1.0
    result[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[:, 0, 1] = 2.0 * (x * y - z * w)
    result[:, 0, 2] = 2.0 * (x * z + y * w)
    result[:, 1, 0] = 2.0 * (x * y + z * w)
    result[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[:, 1, 2] = 2.0 * (y * z - x * w)
    result[:, 2, 0] = 2.0 * (x * z - y * w)
    result[:, 2, 1] = 2.0 * (y * z + x * w)
    result[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    result[:, :3, 3] = start[:3, 3] + values[:, None] * (
        end[:3, 3] - start[:3, 3]
    )[None, :]
    return result


def apply_row_transforms(
    pointcloud: np.ndarray,
    row_indices: np.ndarray,
    row_transforms: np.ndarray,
) -> np.ndarray:
    """Apply one transform per organized scan row to a CPU point cloud."""

    points = np.asarray(pointcloud, dtype=np.float64)
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    transforms = np.asarray(row_transforms, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or rows.shape != (points.shape[0],):
        raise ValueError("pointcloud and row_indices shapes are inconsistent")
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("row_transforms must have shape [H,4,4]")
    if np.any(rows < 0) or np.any(rows >= transforms.shape[0]):
        raise ValueError("row index is outside row_transforms")
    selected = transforms[rows]
    output = points.copy()
    output[:, :3] = np.einsum(
        "nij,nj->ni", selected[:, :3, :3], points[:, :3]
    ) + selected[:, :3, 3]
    return output


def deskew_pointcloud_to_reference(
    pointcloud: np.ndarray,
    *,
    cloud_timestamp: float,
    reference_timestamp: float,
    trajectory,
    lidar_from_base: np.ndarray,
    point_time_scale: float = 1e-9,
    time_bin_sec: float = 0.005,
) -> np.ndarray:
    points = np.asarray(pointcloud, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 5:
        raise ValueError("deskew requires pointcloud columns x,y,z,intensity,t")
    if point_time_scale <= 0.0 or time_bin_sec <= 0.0:
        raise ValueError("deskew time scale and bin size must be positive")
    if not np.isfinite(points).all():
        raise ValueError("deskew pointcloud must be finite")
    lidar_from_base = np.asarray(lidar_from_base, dtype=np.float64).reshape(4, 4)
    base_from_lidar = np.linalg.inv(lidar_from_base)
    reference_pose = trajectory.interpolate(
        float(reference_timestamp),
        max_time_gap=15.0,
        max_translation_gap=2.0,
        max_rotation_gap=0.5,
    ).transform
    relative_time = points[:, 4] * float(point_time_scale)
    bins = np.floor(relative_time / float(time_bin_sec)).astype(np.int64)
    output = points.copy()
    for bin_index in np.unique(bins):
        selected = bins == bin_index
        point_timestamp = float(
            cloud_timestamp + np.mean(relative_time[selected])
        )
        point_pose = trajectory.interpolate(
            point_timestamp,
            max_time_gap=15.0,
            max_translation_gap=2.0,
            max_rotation_gap=0.5,
        ).transform
        reference_lidar_from_point_lidar = (
            lidar_from_base
            @ np.linalg.inv(reference_pose)
            @ point_pose
            @ base_from_lidar
        )
        xyz = points[selected, :3]
        output[selected, :3] = (
            xyz @ reference_lidar_from_point_lidar[:3, :3].T
            + reference_lidar_from_point_lidar[:3, 3]
        )
    return output


@dataclass(frozen=True)
class PointObservation:
    xyz: np.ndarray
    intensity: np.ndarray
    semantic_probabilities: np.ndarray
    ground_probability: np.ndarray
    dynamic: np.ndarray

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        intensity = np.asarray(self.intensity, dtype=np.float64)
        semantic = np.asarray(self.semantic_probabilities, dtype=np.float64)
        ground = np.asarray(self.ground_probability, dtype=np.float64)
        dynamic = np.asarray(self.dynamic, dtype=bool)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N,3]")
        count = xyz.shape[0]
        if (
            intensity.shape != (count,)
            or semantic.ndim != 2
            or semantic.shape[0] != count
            or ground.shape != (count,)
            or dynamic.shape != (count,)
        ):
            raise ValueError("all point fields must match the point count")
        if not (
            np.isfinite(xyz).all()
            and np.isfinite(intensity).all()
            and np.isfinite(semantic).all()
            and np.isfinite(ground).all()
        ):
            raise ValueError("point observation values must be finite")
        if np.any(semantic < 0.0) or np.any(ground < 0.0) or np.any(ground > 1.0):
            raise ValueError("probabilities must be non-negative and ground in [0,1]")
        semantic_sum = semantic.sum(axis=1, keepdims=True)
        if np.any(semantic_sum <= 0.0):
            raise ValueError("semantic probabilities must have positive row sums")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "semantic_probabilities", semantic / semantic_sum)
        object.__setattr__(self, "ground_probability", ground)
        object.__setattr__(self, "dynamic", dynamic)

    def transform(self, target_from_source: np.ndarray) -> "PointObservation":
        return PointObservation(
            xyz=transform_points(self.xyz, target_from_source),
            intensity=self.intensity.copy(),
            semantic_probabilities=self.semantic_probabilities.copy(),
            ground_probability=self.ground_probability.copy(),
            dynamic=self.dynamic.copy(),
        )
