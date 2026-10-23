"""Validated static-TF graph and calibration products."""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

from .coordinates import validate_transform


TransformKey = Tuple[str, str]


@dataclass(frozen=True)
class StaticTransform:
    """ROS static transform represented as parent_from_child."""

    parent_frame: str
    child_frame: str
    translation: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        if not self.parent_frame or not self.child_frame:
            raise ValueError("static TF frame names must not be empty")
        if self.parent_frame == self.child_frame:
            raise ValueError("static TF parent and child must differ")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("static TF translation must contain 3 finite values")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("static TF quaternion must contain 4 finite values")
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-3):
            raise ValueError("static TF quaternion must have unit length")
        object.__setattr__(self, "translation", translation)
        object.__setattr__(
            self,
            "quaternion_xyzw",
            quaternion / np.linalg.norm(quaternion),
        )

    def as_matrix(self) -> np.ndarray:
        x, y, z, w = self.quaternion_xyzw
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )
        matrix[:3, 3] = self.translation
        return validate_transform(matrix)


def static_transform_from_ros(message) -> StaticTransform:
    translation = message.transform.translation
    rotation = message.transform.rotation
    return StaticTransform(
        parent_frame=str(message.header.frame_id).lstrip("/"),
        child_frame=str(message.child_frame_id).lstrip("/"),
        translation=np.asarray(
            [translation.x, translation.y, translation.z],
            dtype=np.float64,
        ),
        quaternion_xyzw=np.asarray(
            [rotation.x, rotation.y, rotation.z, rotation.w],
            dtype=np.float64,
        ),
    )


def select_static_transforms(
    transforms: Iterable[StaticTransform],
    *,
    excluded_frames: Iterable[str] = (),
) -> list:
    excluded = {str(frame).lstrip("/") for frame in excluded_frames}
    return [
        transform
        for transform in transforms
        if transform.parent_frame not in excluded
        and transform.child_frame not in excluded
    ]


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first, second)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def deduplicate_static_transforms(
    transforms: Iterable[StaticTransform],
    *,
    translation_tolerance: float = 1e-4,
    rotation_tolerance: float = 1e-4,
) -> Dict[TransformKey, StaticTransform]:
    if translation_tolerance < 0.0 or rotation_tolerance < 0.0:
        raise ValueError("static TF tolerances must be non-negative")
    unique: Dict[TransformKey, StaticTransform] = {}
    for transform in transforms:
        key = (transform.parent_frame, transform.child_frame)
        existing = unique.get(key)
        if existing is None:
            unique[key] = transform
            continue
        translation_error = float(
            np.linalg.norm(existing.translation - transform.translation)
        )
        rotation_error = _rotation_distance(
            existing.quaternion_xyzw,
            transform.quaternion_xyzw,
        )
        if (
            translation_error > translation_tolerance
            or rotation_error > rotation_tolerance
        ):
            raise ValueError(
                f"conflicting static TF {key}: translation_error="
                f"{translation_error:.6g}, rotation_error={rotation_error:.6g}"
            )
    return unique


def resolve_transform(
    transforms: Mapping[TransformKey, StaticTransform],
    target_frame: str,
    source_frame: str,
) -> np.ndarray:
    """Return target_from_source by traversing the validated static TF graph."""
    if target_frame == source_frame:
        return np.eye(4, dtype=np.float64)

    graph = {}
    for transform in transforms.values():
        parent_from_child = transform.as_matrix()
        graph.setdefault(transform.child_frame, []).append(
            (transform.parent_frame, parent_from_child)
        )
        graph.setdefault(transform.parent_frame, []).append(
            (transform.child_frame, np.linalg.inv(parent_from_child))
        )

    queue = deque([(source_frame, np.eye(4, dtype=np.float64))])
    visited = {source_frame}
    while queue:
        current_frame, current_from_source = queue.popleft()
        for neighbor_frame, neighbor_from_current in graph.get(current_frame, ()):
            if neighbor_frame in visited:
                continue
            neighbor_from_source = neighbor_from_current @ current_from_source
            if neighbor_frame == target_frame:
                return validate_transform(neighbor_from_source)
            visited.add(neighbor_frame)
            queue.append((neighbor_frame, neighbor_from_source))
    raise ValueError(
        f"no static TF path from {source_frame} to {target_frame}"
    )


def liosam_extrinsics(
    transforms: Mapping[TransformKey, StaticTransform],
    *,
    lidar_frame: str,
    imu_frame: str,
) -> dict:
    lidar_from_imu = resolve_transform(transforms, lidar_frame, imu_frame)
    rotation = lidar_from_imu[:3, :3].reshape(-1).tolist()
    return {
        "extrinsicTrans": lidar_from_imu[:3, 3].tolist(),
        "extrinsicRot": rotation,
        "extrinsicRPY": list(rotation),
    }


def build_liosam_config(
    template: Mapping[str, object],
    transforms: Mapping[TransformKey, StaticTransform],
    *,
    lidar_frame: str,
    imu_frame: str,
    imu_convention_approved: bool,
) -> dict:
    if not imu_convention_approved:
        raise ValueError("IMU convention must be explicitly approved")
    config = copy.deepcopy(dict(template))
    metadata = config.setdefault("rellis_metadata", {})
    lio_sam = config.setdefault("lio_sam", {})
    if not isinstance(metadata, dict) or not isinstance(lio_sam, dict):
        raise ValueError("LIO-SAM template sections must be mappings")

    sensor_extrinsics = liosam_extrinsics(
        transforms,
        lidar_frame=lidar_frame,
        imu_frame=imu_frame,
    )
    identity = np.eye(3, dtype=np.float64).reshape(-1).tolist()
    lio_sam.update(
        {
            "imuTopic": "/rellis/imu_correct",
            "gpsTopic": "/rellis/gps/odom",
            "extrinsicTrans": sensor_extrinsics["extrinsicTrans"],
            "extrinsicRot": identity,
            "extrinsicRPY": list(identity),
        }
    )
    lio_sam["lidarFrame"] = lidar_frame
    # LIO-SAM queries this TF only once at startup, before bag /tf_static arrives.
    # Keep its internal trajectory in the LiDAR frame and convert it on export.
    lio_sam["baselinkFrame"] = lidar_frame
    config["rellis_imu_correction"] = {
        "input_topic": "/vectornav/IMU",
        "output_topic": "/rellis/imu_correct",
        "output_frame": lidar_frame,
        "lidar_from_imu_rotation": sensor_extrinsics["extrinsicRot"],
        "enu_from_sensor_world": [
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
        ],
    }
    config["rellis_gps_conversion"] = {
        "input_topic": "/vectornav/GPS",
        "output_topic": "/rellis/gps/odom",
        "output_frame": "map",
        "horizontal_variance": 4.0,
        "vertical_variance": 100.0,
        "min_interval_sec": 0.2,
        "duplicate_distance_m": 0.01,
    }
    metadata["calibration_validated"] = True
    fingerprint = {
        "lidar_frame": lidar_frame,
        "imu_frame": imu_frame,
        "sensor_extrinsics": sensor_extrinsics,
        "imu_correction": config["rellis_imu_correction"],
        "gps_conversion": config["rellis_gps_conversion"],
    }
    serialized = json.dumps(
        fingerprint,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata["calibration_version"] = hashlib.sha256(serialized).hexdigest()[:16]
    return config
