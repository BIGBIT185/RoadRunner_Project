"""记录的 GOOSE RT3000 轨迹校验、插值和位姿变换工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_QUATERNION_NORM_TOLERANCE = 1e-3
_LIMIT_COMPARISON_ATOL = 1e-12
_DEFAULT_MAX_GAP = 0.1
_DEFAULT_MAX_TRANSLATION_STEP = 1.0
_DEFAULT_MAX_ROTATION_STEP = 0.2
_DEFAULT_MAX_SPEED = 40.0
_DEFAULT_MAX_ANGULAR_SPEED = 2.0


def _exceeds_limit(values: np.ndarray | float, limit: float) -> np.ndarray | bool:
    return np.asarray(values) > float(limit) + _LIMIT_COMPARISON_ATOL


def _quaternion_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def _quaternion_slerp(q0: np.ndarray, q1: np.ndarray, ratio: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + ratio * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = float(np.arccos(dot))
    return (
        np.sin((1.0 - ratio) * angle) * q0 + np.sin(ratio * angle) * q1
    ) / np.sin(angle)


def _validated_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must have shape [4]")
    if not np.isfinite(quaternion).all():
        raise ValueError("rotation values must be finite")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        raise ValueError("quaternion norm error exceeds 1e-3")
    return quaternion / norm


def _stamp_seconds(stamp: object) -> float:
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    seconds = getattr(stamp, "secs", getattr(stamp, "sec", None))
    nanoseconds = getattr(stamp, "nsecs", getattr(stamp, "nanosec", 0))
    if seconds is None:
        raise ValueError("TF timestamp is missing seconds")
    return float(seconds) + float(nanoseconds) * 1e-9


def _inverse_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ transform[:3, 3])
    return result


def _rotation_distance(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def _motion_rates(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a centered difference so dense timestamp jitter is not a false spike."""

    if timestamps.size == 2:
        time_spans = np.diff(timestamps)
        translation_spans = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        rotation_spans = np.asarray(
            [_rotation_distance(quaternions_xyzw[0], quaternions_xyzw[1])]
        )
    else:
        time_spans = timestamps[2:] - timestamps[:-2]
        translation_spans = np.linalg.norm(positions[2:] - positions[:-2], axis=1)
        rotation_spans = np.asarray(
            [
                _rotation_distance(quaternions_xyzw[index], quaternions_xyzw[index + 2])
                for index in range(timestamps.size - 2)
            ]
        )
    return translation_spans / time_spans, rotation_spans / time_spans


@dataclass(frozen=True)
class Rt3000Sample:
    timestamp: float
    transform_utm_base: np.ndarray


@dataclass(frozen=True)
class Rt3000Interpolation:
    transform: np.ndarray
    source_stamps: tuple[float, float]
    interpolation_ratio: float
    nearest_dt: float
    interpolated: bool


@dataclass
class Rt3000Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    source_bag: str

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.quaternions_xyzw = np.asarray(self.quaternions_xyzw, dtype=np.float64)
        self.source_bag = str(self.source_bag)
        self._validate()

    def _validate(self) -> None:
        count = self.timestamps.size
        if self.timestamps.ndim != 1 or count < 2:
            raise ValueError("timestamps must contain at least two samples")
        if self.positions.shape != (count, 3):
            raise ValueError("positions must have shape [N, 3]")
        if self.quaternions_xyzw.shape != (count, 4):
            raise ValueError("quaternions_xyzw must have shape [N, 4]")
        if not (
            np.isfinite(self.timestamps).all()
            and np.isfinite(self.positions).all()
            and np.isfinite(self.quaternions_xyzw).all()
        ):
            raise ValueError("trajectory values must be finite")
        if np.any(np.diff(self.timestamps) <= 0.0):
            raise ValueError("timestamps must be strictly increasing")
        norms = np.linalg.norm(self.quaternions_xyzw, axis=1)
        if np.any(np.abs(norms - 1.0) > _QUATERNION_NORM_TOLERANCE):
            raise ValueError("quaternion norm error exceeds 1e-3")
        self.quaternions_xyzw = self.quaternions_xyzw / norms[:, None]
        time_steps = np.diff(self.timestamps)
        translation_steps = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        rotation_steps = np.asarray(
            [
                _rotation_distance(
                    self.quaternions_xyzw[index],
                    self.quaternions_xyzw[index + 1],
                )
                for index in range(count - 1)
            ]
        )
        speeds, angular_speeds = _motion_rates(
            self.timestamps,
            self.positions,
            self.quaternions_xyzw,
        )
        if np.any(_exceeds_limit(time_steps, _DEFAULT_MAX_GAP)):
            raise ValueError("trajectory time gap exceeds 0.1 s")
        if np.any(
            _exceeds_limit(translation_steps, _DEFAULT_MAX_TRANSLATION_STEP)
        ):
            raise ValueError("trajectory translation step exceeds 1 m")
        if np.any(_exceeds_limit(rotation_steps, _DEFAULT_MAX_ROTATION_STEP)):
            raise ValueError("trajectory rotation step exceeds 0.2 rad")
        if np.any(_exceeds_limit(speeds, _DEFAULT_MAX_SPEED)):
            raise ValueError("trajectory speed exceeds 40 m/s")
        if np.any(_exceeds_limit(angular_speeds, _DEFAULT_MAX_ANGULAR_SPEED)):
            raise ValueError("trajectory angular speed exceeds 2 rad/s")

    @classmethod
    def from_npz(cls, path: Path | str) -> "Rt3000Trajectory":
        with np.load(path, allow_pickle=False) as data:
            contract = {
                "trajectory_source": "rt3000",
                "trajectory_pose_frame": "base_link",
                "trajectory_reference_frame": "local",
            }
            for field, expected in contract.items():
                if field not in data:
                    raise ValueError(f"trajectory NPZ is missing {field}")
                if str(data[field]) != expected:
                    raise ValueError(f"{field} must be {expected}")
            return cls(
                timestamps=data["timestamps"],
                positions=data["positions"],
                quaternions_xyzw=data["quaternions_xyzw"],
                source_bag=str(data["source_bag"]),
            )

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "timestamps": self.timestamps,
            "positions": self.positions,
            "quaternions_xyzw": self.quaternions_xyzw,
            "source_bag": np.asarray(self.source_bag),
            "trajectory_source": np.asarray("rt3000"),
            "trajectory_pose_frame": np.asarray("base_link"),
            "trajectory_reference_frame": np.asarray("local"),
        }

    def interpolate(
        self,
        timestamp: float,
        *,
        max_time_gap: float = 0.1,
    ) -> Rt3000Interpolation:
        timestamp = float(timestamp)
        max_time_gap = float(max_time_gap)
        if not np.isfinite(timestamp):
            raise ValueError("interpolation timestamp must be finite")
        if not np.isfinite(max_time_gap) or max_time_gap <= 0.0:
            raise ValueError("max_time_gap must be finite and positive")
        if timestamp < self.timestamps[0] or timestamp > self.timestamps[-1]:
            raise ValueError(f"timestamp {timestamp} is outside trajectory")

        upper = int(np.searchsorted(self.timestamps, timestamp, side="left"))
        if upper < self.timestamps.size and self.timestamps[upper] == timestamp:
            lower = upper
        else:
            lower = upper - 1

        if lower == upper:
            ratio = 0.0
            position = self.positions[lower]
            quaternion = self.quaternions_xyzw[lower]
            source_stamps = (float(self.timestamps[lower]),) * 2
            nearest_dt = 0.0
            interpolated = False
        else:
            time_gap = float(self.timestamps[upper] - self.timestamps[lower])
            if _exceeds_limit(time_gap, max_time_gap):
                raise ValueError("trajectory time gap exceeds configured limit")
            ratio = float((timestamp - self.timestamps[lower]) / time_gap)
            position = self.positions[lower] + ratio * (
                self.positions[upper] - self.positions[lower]
            )
            quaternion = _quaternion_slerp(
                self.quaternions_xyzw[lower],
                self.quaternions_xyzw[upper],
                ratio,
            )
            source_stamps = (
                float(self.timestamps[lower]),
                float(self.timestamps[upper]),
            )
            nearest_dt = min(timestamp - source_stamps[0], source_stamps[1] - timestamp)
            interpolated = True

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _quaternion_matrix(quaternion)
        transform[:3, 3] = position
        return Rt3000Interpolation(
            transform=transform,
            source_stamps=source_stamps,
            interpolation_ratio=ratio,
            nearest_dt=float(nearest_dt),
            interpolated=interpolated,
        )


def rt3000_sample_from_tf(transform_stamped: object) -> Rt3000Sample:
    header = transform_stamped.header
    if header.frame_id != "base_link" or transform_stamped.child_frame_id != "utm":
        raise ValueError("RT3000 TF must have frame pair base_link -> utm")

    timestamp = _stamp_seconds(header.stamp)
    translation_message = transform_stamped.transform.translation
    translation = np.asarray(
        [translation_message.x, translation_message.y, translation_message.z],
        dtype=np.float64,
    )
    rotation_message = transform_stamped.transform.rotation
    quaternion = _validated_quaternion(
        np.asarray(
            [rotation_message.x, rotation_message.y, rotation_message.z, rotation_message.w],
            dtype=np.float64,
        )
    )
    if not np.isfinite(timestamp) or not np.isfinite(translation).all():
        raise ValueError("TF timestamp and translation values must be finite")

    transform_base_utm = np.eye(4, dtype=np.float64)
    transform_base_utm[:3, :3] = _quaternion_matrix(quaternion)
    transform_base_utm[:3, 3] = translation
    return Rt3000Sample(
        timestamp=timestamp,
        transform_utm_base=_inverse_transform(transform_base_utm),
    )


def build_local_trajectory(
    samples: Iterable[Rt3000Sample],
    *,
    source_bag: str = "",
    max_gap: float = 0.1,
    max_translation_step: float = 1.0,
    max_rotation_step: float = 0.2,
    max_speed: float = 40.0,
    max_angular_speed: float = 2.0,
) -> Rt3000Trajectory:
    limits = np.asarray(
        [max_gap, max_translation_step, max_rotation_step, max_speed, max_angular_speed],
        dtype=np.float64,
    )
    if not np.isfinite(limits).all() or np.any(limits <= 0.0):
        raise ValueError("trajectory limits must be finite and positive")

    ordered = sorted(samples, key=lambda sample: sample.timestamp)
    unique: list[Rt3000Sample] = []
    for sample in ordered:
        if not np.isfinite(sample.timestamp) or not np.isfinite(sample.transform_utm_base).all():
            raise ValueError("RT3000 sample values must be finite")
        if np.asarray(sample.transform_utm_base).shape != (4, 4):
            raise ValueError("RT3000 sample transform must have shape [4, 4]")
        if unique and sample.timestamp == unique[-1].timestamp:
            if np.array_equal(sample.transform_utm_base, unique[-1].transform_utm_base):
                continue
            raise ValueError("conflicting duplicate RT3000 timestamp")
        unique.append(sample)
    if len(unique) < 2:
        raise ValueError("RT3000 trajectory must contain at least two unique samples")

    transform_local_utm = _inverse_transform(unique[0].transform_utm_base)
    local_transforms = np.stack(
        [transform_local_utm @ sample.transform_utm_base for sample in unique]
    )
    timestamps = np.asarray([sample.timestamp for sample in unique], dtype=np.float64)
    positions = local_transforms[:, :3, 3]
    quaternions = np.stack(
        [_matrix_quaternion(transform[:3, :3]) for transform in local_transforms]
    )

    time_steps = np.diff(timestamps)
    translation_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    rotation_steps = np.asarray(
        [_rotation_distance(quaternions[index], quaternions[index + 1]) for index in range(len(unique) - 1)]
    )
    speeds, angular_speeds = _motion_rates(timestamps, positions, quaternions)
    if np.any(_exceeds_limit(time_steps, max_gap)):
        raise ValueError("trajectory time gap exceeds 0.1 s")
    if np.any(_exceeds_limit(translation_steps, max_translation_step)):
        raise ValueError("trajectory translation step exceeds 1 m")
    if np.any(_exceeds_limit(rotation_steps, max_rotation_step)):
        raise ValueError("trajectory rotation step exceeds 0.2 rad")
    if np.any(_exceeds_limit(speeds, max_speed)):
        raise ValueError("trajectory speed exceeds 40 m/s")
    if np.any(_exceeds_limit(angular_speeds, max_angular_speed)):
        raise ValueError("trajectory angular speed exceeds 2 rad/s")

    return Rt3000Trajectory(
        timestamps=timestamps,
        positions=positions,
        quaternions_xyzw=quaternions,
        source_bag=source_bag,
    )
