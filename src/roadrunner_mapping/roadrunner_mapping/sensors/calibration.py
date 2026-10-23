"""从 GOOSE 记录的静态 TF 图解析相机和 LiDAR 标定。"""

from __future__ import annotations

import numpy as np

from roadrunner_mapping.geometry.calibration import (
    deduplicate_static_transforms,
    resolve_transform,
)


CAMERA_FRAME = "sensor/camera/windshield/vis"
LIDAR_FRAME = "sensor/lidar/vls128_roof"
BASE_FRAME = "base_link"


def rectified_intrinsics(
    camera_info,
    *,
    image_shape,
    camera_frame: str = CAMERA_FRAME,
) -> np.ndarray:
    """根据 CameraInfo.P、ROI 和 binning 生成与图像尺寸一致的内参矩阵。"""
    if str(camera_info.header.frame_id).lstrip("/") != camera_frame.lstrip("/"):
        raise ValueError("CameraInfo frame does not match configured camera frame")
    height, width = int(image_shape[0]), int(image_shape[1])
    bin_x = max(1, int(camera_info.binning_x))
    bin_y = max(1, int(camera_info.binning_y))
    roi = camera_info.roi
    roi_width = int(roi.width) or width * bin_x
    roi_height = int(roi.height) or height * bin_y
    expected_width = roi_width // bin_x
    expected_height = roi_height // bin_y
    if (height, width) != (expected_height, expected_width):
        raise ValueError(
            "CameraInfo ROI/binning image dimensions do not match decoded image"
        )
    projection = np.asarray(camera_info.P, dtype=np.float64).reshape(3, 4)
    intrinsics = np.asarray(
        [
            [projection[0, 0] / bin_x, 0.0, (projection[0, 2] - int(roi.x_offset)) / bin_x],
            [0.0, projection[1, 1] / bin_y, (projection[1, 2] - int(roi.y_offset)) / bin_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(intrinsics).all() or np.any(np.diag(intrinsics) <= 0.0):
        raise ValueError("rectified camera intrinsics must be finite and positive")
    return intrinsics


def build_goose_calibration(
    transforms,
    camera_info,
    *,
    image_shape,
    camera_frame: str = CAMERA_FRAME,
    lidar_frame: str = LIDAR_FRAME,
    base_frame: str = BASE_FRAME,
) -> dict:
    """解析静态 TF 外参并返回统一的 GOOSE 标定字典。"""
    graph = deduplicate_static_transforms(transforms)
    base_from_lidar = resolve_transform(graph, base_frame, lidar_frame)
    camera_from_base = resolve_transform(graph, camera_frame, base_frame)
    return {
        "intrinsics": rectified_intrinsics(
            camera_info,
            image_shape=image_shape,
            camera_frame=camera_frame,
        ),
        "T_lidar_base": np.linalg.inv(base_from_lidar).astype(np.float32),
        "T_camera_base": camera_from_base.astype(np.float32),
    }
