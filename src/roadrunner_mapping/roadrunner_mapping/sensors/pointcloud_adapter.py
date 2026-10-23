"""GOOSE ROS 消息到原始帧契约的适配器。

这里集中处理 PointCloud2 解码、整帧时差对齐和 NPZ 原子写入；实时节点和
离线构建都通过这些小函数共享相同的点云/位姿语义。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from roadrunner_mapping.geometry.coordinates import relative_transform
from roadrunner_mapping.geometry.pointcloud import (
    pointcloud2_to_array,
    pointcloud2_to_array_with_rows,
)


TOPIC_IMAGE = "/sensor/camera/windshield/vis/image_rect_color"
TOPIC_CAMERA_INFO = "/sensor/camera/windshield/vis/camera_info"
TOPIC_CLOUD = "/sensor/lidar/vls128_roof/velodyne_points"
TOPIC_TF = "/tf"
TOPIC_TF_STATIC = "/tf_static"
TOPIC_GPS = "/sensor/ins/oxts_rt3000/gps/fix"


def message_stamp(message) -> float:
    """读取并验证 ROS header 时间戳。"""
    timestamp = float(message.header.stamp.to_sec())
    if not np.isfinite(timestamp):
        raise ValueError("message timestamp must be finite")
    return timestamp


def decode_goose_cloud(
    message,
    *,
    minimum_range: float = 1.5,
    maximum_range: float = 120.0,
    minimum_forward_x: float | None = None,
) -> np.ndarray:
    """提取点云，并按有限性、距离和 LiDAR 前向 x 轴过滤。"""
    if not 0.0 <= minimum_range < maximum_range:
        raise ValueError("point-cloud range limits are invalid")
    if minimum_forward_x is not None and not np.isfinite(minimum_forward_x):
        raise ValueError("minimum_forward_x must be finite when provided")
    points = pointcloud2_to_array(
        message,
        fields=("x", "y", "z", "intensity"),
    )
    finite = np.isfinite(points).all(axis=1)
    ranges = np.linalg.norm(points[:, :3], axis=1)
    selected = finite & (ranges >= minimum_range) & (ranges <= maximum_range)
    if minimum_forward_x is not None:
        selected &= points[:, 0] >= float(minimum_forward_x)
    return points[selected].astype(np.float32, copy=False)


def decode_goose_cloud_with_rows(
    message,
    *,
    minimum_range: float = 1.5,
    maximum_range: float = 120.0,
    minimum_forward_x: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a cloud and preserve organized row IDs for optional scan deskew."""

    if not 0.0 <= minimum_range < maximum_range:
        raise ValueError("point-cloud range limits are invalid")
    if minimum_forward_x is not None and not np.isfinite(minimum_forward_x):
        raise ValueError("minimum_forward_x must be finite when provided")
    points, row_indices = pointcloud2_to_array_with_rows(
        message,
        fields=("x", "y", "z", "intensity"),
    )
    finite = np.isfinite(points).all(axis=1)
    ranges = np.linalg.norm(points[:, :3], axis=1)
    selected = finite & (ranges >= minimum_range) & (ranges <= maximum_range)
    if minimum_forward_x is not None:
        selected &= points[:, 0] >= float(minimum_forward_x)
    return (
        points[selected].astype(np.float32, copy=False),
        row_indices[selected].astype(np.int32, copy=False),
    )


def rigid_align_cloud_to_image(
    pointcloud: np.ndarray,
    *,
    world_from_cloud_body: np.ndarray,
    world_from_image_body: np.ndarray,
    lidar_from_base: np.ndarray,
) -> np.ndarray:
    """把 cloud timestamp 的整帧 LiDAR 点变换到 image timestamp。

    GOOSE 没有可靠逐点时间字段，不能执行扫描内 deskew；这里仅利用两个时刻
    的机器人位姿计算一个刚体变换，用于图像语义投影。
    """
    points = np.asarray(pointcloud, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points).all():
        raise ValueError("pointcloud must be a finite [N,C>=3] array")
    lidar_from_base = np.asarray(lidar_from_base, dtype=np.float64).reshape(4, 4)
    base_from_lidar = np.linalg.inv(lidar_from_base)
    image_lidar_from_cloud_lidar = (
        lidar_from_base
        @ np.linalg.inv(np.asarray(world_from_image_body, dtype=np.float64).reshape(4, 4))
        @ np.asarray(world_from_cloud_body, dtype=np.float64).reshape(4, 4)
        @ base_from_lidar
    )
    output = points.copy()
    output[:, :3] = (
        points[:, :3] @ image_lidar_from_cloud_lidar[:3, :3].T
        + image_lidar_from_cloud_lidar[:3, 3]
    )
    return output.astype(np.float32)


def raw_frame_payload(
    *,
    image: np.ndarray,
    pointcloud: np.ndarray,
    calibration: dict,
    world_from_body: np.ndarray,
    previous_world_from_body: np.ndarray | None,
    timestamp_image: float,
    timestamp_cloud: float,
    camera_info_timestamp: float,
    trajectory_source_stamps: tuple[float, float],
    trajectory_interp_ratio: float,
    sequence_id: str,
    split: str,
    source_bag: str,
) -> dict:
    """构造离线原始帧字典并记录时间、位姿和标定来源。"""
    world_from_body = np.asarray(world_from_body, dtype=np.float32).reshape(4, 4)
    if previous_world_from_body is None:
        relative_pose = np.eye(4, dtype=np.float32)
    else:
        relative_pose = relative_transform(
            world_from_body,
            np.asarray(previous_world_from_body, dtype=np.float64),
        ).astype(np.float32)
    return {
        "image": np.asarray(image, dtype=np.uint8),
        "pointcloud": np.asarray(pointcloud, dtype=np.float32),
        "pointcloud_view": np.asarray("360"),
        "intrinsics": np.asarray(calibration["intrinsics"], dtype=np.float32),
        "T_camera_base": np.asarray(calibration["T_camera_base"], dtype=np.float32),
        "T_lidar_base": np.asarray(calibration["T_lidar_base"], dtype=np.float32),
        "world_from_body": world_from_body,
        "relative_pose": relative_pose,
        "timestamp_image": np.float64(timestamp_image),
        "timestamp_cloud": np.float64(timestamp_cloud),
        "dt_cloud_image": np.float64(timestamp_cloud - timestamp_image),
        "camera_info_timestamp": np.float64(camera_info_timestamp),
        "trajectory_timestamp_left": np.float64(trajectory_source_stamps[0]),
        "trajectory_timestamp_right": np.float64(trajectory_source_stamps[1]),
        "trajectory_interp_ratio": np.float32(trajectory_interp_ratio),
        "pointcloud_deskewed": np.asarray(False),
        "pointcloud_rigid_aligned": np.asarray(True),
        "deskew_reference_timestamp": np.float64(timestamp_image),
        "sequence_id": np.asarray(sequence_id),
        "split": np.asarray(split),
        "source_bag": np.asarray(source_bag),
    }


def write_npz_atomic(path: Path | str, payload: dict) -> None:
    """先写临时 NPZ、fsync，再原子替换目标文件，避免中断留下半文件。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp.npz",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.chmod(temporary, 0o644)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
