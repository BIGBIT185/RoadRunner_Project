"""Versioned training-sample contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


SCHEMA_VERSION = "rellis-roadrunner-v1"
RISK_SEMANTICS = "0_safe_1_risky"


@dataclass(frozen=True)
class MapSpec:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    @property
    def shape(self) -> tuple[int, int]:
        height = int(round((self.x_max - self.x_min) / self.resolution))
        width = int(round((self.y_max - self.y_min) / self.resolution))
        return height, width


def make_online_map_spec(
    *,
    length_x_m: float = 40.0,
    length_y_m: float | None = None,
    resolution_m: float = 0.2,
) -> MapSpec:
    """Build a runtime BEV map contract from externally supplied dimensions."""

    length_x_m = float(length_x_m)
    length_y_m = length_x_m if length_y_m is None else float(length_y_m)
    resolution_m = float(resolution_m)
    if (
        not np.isfinite(length_x_m)
        or not np.isfinite(length_y_m)
        or not np.isfinite(resolution_m)
        or length_x_m <= 0.0
        or length_y_m <= 0.0
        or resolution_m <= 0.0
    ):
        raise ValueError("map lengths and resolution must be finite and positive")
    if int(round(length_x_m / resolution_m)) <= 0 or int(
        round(length_y_m / resolution_m)
    ) <= 0:
        raise ValueError("map dimensions must contain at least one cell")
    return MapSpec(
        x_min=-0.5 * length_x_m,
        x_max=0.5 * length_x_m,
        y_min=-0.5 * length_y_m,
        y_max=0.5 * length_y_m,
        resolution=resolution_m,
    )


DEFAULT_MAP_SPEC = MapSpec(
    x_min=-20.0,
    x_max=20.0,
    y_min=-20.0,
    y_max=20.0,
    resolution=0.2,
)

# 实时 GOOSE 链使用独立的车体系 BEV 范围；保持 DEFAULT_MAP_SPEC 不变，
# 以免改变 RELLIS 离线样本和既有训练数据的尺寸契约。
ONLINE_MAP_SPEC = make_online_map_spec()

REQUIRED_FIELDS = {
    "schema_version",
    "risk_semantics",
    "pointcloud",
    "pointcloud_view",
    "image",
    "intrinsics",
    "T_camera_base",
    "T_lidar_base",
    "world_from_body",
    "previous_elevation",
    "previous_elevation_valid",
    "relative_pose",
    "traversability_risk_target",
    "elevation_target",
    "confidence_target",
    "geometry_target",
    "semantic_target",
    "clearance_target",
    "reliability_target",
    "valid_mask",
    "observation_count",
    "last_observed_time",
    "timestamp_image",
    "timestamp_cloud",
    "dt_cloud_image",
    "sequence_id",
    "split",
    "prior_source",
    "prior_timestamp",
    "prior_valid_ratio",
    "prior_file",
}

OPTIONAL_MAP_FIELDS = {
    "geometry_hindsight_target",
    "geometry_hindsight_valid_mask",
    "geometry_hindsight_confidence",
    "geometry_hindsight_observation_count",
    "semantic_hindsight_target",
    "semantic_hindsight_valid_mask",
    "semantic_hindsight_confidence",
    "semantic_hindsight_observation_count",
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
    "geometry_confidence",
    "geometry_valid_mask",
    "geometry_support_count",
    "geometry_age",
    "geometry_conflict_score",
    "ground_fit_valid",
    "geometry_hindsight_slope",
    "geometry_hindsight_longitudinal_slope",
    "geometry_hindsight_lateral_slope",
    "geometry_hindsight_roughness",
    "geometry_hindsight_positive_step_height",
    "geometry_hindsight_negative_step_height",
    "geometry_hindsight_non_ground_ratio",
    "geometry_hindsight_ground_fit_residual",
    "geometry_hindsight_slope_risk",
    "geometry_hindsight_roughness_risk",
    "geometry_hindsight_positive_step_risk",
    "geometry_hindsight_negative_step_risk",
    "geometry_hindsight_geometry_age",
    "geometry_hindsight_geometry_conflict_score",
    "geometry_hindsight_ground_fit_valid",
    "grass_probability",
    "grass_observation_weight",
    "grass_mask",
    "grass_density_risk",
    "grass_prior_risk",
    "grass_positive_step_risk",
    "grass_negative_step_risk",
    "grass_negative_step_valid",
}


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar text, got shape {array.shape}")
    return str(array.item())


def _require_finite(name: str, value: object) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def validate_sample(
    sample: Mapping[str, object],
    map_spec: MapSpec = DEFAULT_MAP_SPEC,
) -> None:
    missing = sorted(REQUIRED_FIELDS.difference(sample))
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    if _scalar_text(sample["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("schema_version does not match")
    if _scalar_text(sample["risk_semantics"]) != RISK_SEMANTICS:
        raise ValueError("risk_semantics does not match")
    if _scalar_text(sample["pointcloud_view"]) != "360":
        raise ValueError("pointcloud_view must be 360")

    pointcloud = _require_finite("pointcloud", sample["pointcloud"])
    if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
        raise ValueError(f"pointcloud must be [N,C>=3], got {pointcloud.shape}")

    image = np.asarray(sample["image"])
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got {image.shape}")

    expected_shape = map_spec.shape
    map_fields = (
        "previous_elevation",
        "previous_elevation_valid",
        "traversability_risk_target",
        "elevation_target",
        "confidence_target",
        "geometry_target",
        "semantic_target",
        "clearance_target",
        "reliability_target",
        "valid_mask",
        "observation_count",
        "last_observed_time",
    )
    for name in map_fields:
        array = _require_finite(name, sample[name])
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    for name in OPTIONAL_MAP_FIELDS.intersection(sample):
        array = _require_finite(name, sample[name])
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")

    for name in (
        "traversability_risk_target",
        "confidence_target",
        "geometry_target",
        "semantic_target",
        "clearance_target",
        "reliability_target",
        "valid_mask",
    ):
        array = np.asarray(sample[name])
        if np.any((array < 0.0) | (array > 1.0)):
            raise ValueError(f"{name} must be in [0, 1]")
    for name in (
        "slope_risk",
        "roughness_risk",
        "positive_step_risk",
        "negative_step_risk",
        "geometry_confidence",
        "geometry_valid_mask",
        "ground_fit_valid",
        "semantic_hindsight_target",
        "semantic_hindsight_valid_mask",
        "semantic_hindsight_confidence",
        "geometry_hindsight_target",
        "geometry_hindsight_valid_mask",
        "geometry_hindsight_confidence",
        "geometry_hindsight_slope_risk",
        "geometry_hindsight_roughness_risk",
        "geometry_hindsight_positive_step_risk",
        "geometry_hindsight_negative_step_risk",
        "geometry_hindsight_ground_fit_valid",
        "grass_probability",
        "grass_mask",
        "grass_density_risk",
        "grass_prior_risk",
        "grass_positive_step_risk",
        "grass_negative_step_risk",
        "grass_negative_step_valid",
    ):
        if name in sample:
            array = np.asarray(sample[name])
            if np.any((array < 0.0) | (array > 1.0)):
                raise ValueError(f"{name} must be in [0, 1]")

    intrinsics = _require_finite("intrinsics", sample["intrinsics"])
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {intrinsics.shape}")
    for name in (
        "T_camera_base",
        "T_lidar_base",
        "world_from_body",
        "relative_pose",
    ):
        transform = _require_finite(name, sample[name])
        if transform.shape != (4, 4):
            raise ValueError(f"{name} must be 4x4, got {transform.shape}")

    for name in (
        "timestamp_image",
        "timestamp_cloud",
        "dt_cloud_image",
        "prior_timestamp",
        "prior_valid_ratio",
    ):
        value = _require_finite(name, sample[name])
        if value.size != 1:
            raise ValueError(f"{name} must be scalar")
    if float(np.asarray(sample["timestamp_cloud"])) > float(
        np.asarray(sample["timestamp_image"])
    ):
        raise ValueError("timestamp_cloud must be causal")
    if float(np.asarray(sample["prior_timestamp"])) > float(
        np.asarray(sample["timestamp_image"])
    ):
        raise ValueError("prior_timestamp must be causal")
    prior_valid_ratio = float(np.asarray(sample["prior_valid_ratio"]))
    if not 0.0 <= prior_valid_ratio <= 1.0:
        raise ValueError("prior_valid_ratio must be in [0,1]")
    if "hindsight" in _scalar_text(sample["prior_source"]).lower():
        raise ValueError("prior_source must not reference hindsight")
    _scalar_text(sample["prior_file"])
    if not _scalar_text(sample["sequence_id"]) or not _scalar_text(sample["split"]):
        raise ValueError("sequence_id and split must not be empty")
