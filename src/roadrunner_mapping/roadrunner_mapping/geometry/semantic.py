"""RUGD semantic definitions and calibrated image-to-point projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinates import transform_points, validate_transform


RUGD_CLASSES = (
    "void",
    "dirt",
    "sand",
    "grass",
    "tree",
    "pole",
    "water",
    "sky",
    "vehicle",
    "container",
    "asphalt",
    "gravel",
    "building",
    "mulch",
    "rock-bed",
    "log",
    "bicycle",
    "person",
    "fence",
    "bush",
    "sign",
    "rock",
    "bridge",
    "concrete",
    "picnic-table",
)

SEMANTIC_RISK = {
    1: 0.30,
    2: 0.50,
    3: 0.25,
    4: 1.00,
    5: 1.00,
    6: 1.00,
    8: 1.00,
    9: 1.00,
    10: 0.05,
    11: 0.35,
    12: 1.00,
    13: 0.40,
    14: 0.70,
    15: 1.00,
    16: 1.00,
    17: 1.00,
    18: 1.00,
    19: 0.70,
    20: 1.00,
    21: 1.00,
    22: 0.10,
    23: 0.10,
    24: 1.00,
}
HARD_OBSTACLE_IDS = frozenset(
    {4, 5, 6, 8, 9, 12, 15, 16, 17, 18, 20, 21, 24}
)
DYNAMIC_CLASS_IDS = frozenset({8, 16, 17})
GROUND_CLASS_IDS = frozenset({1, 2, 3, 10, 11, 13, 14, 22, 23})


@dataclass(frozen=True)
class ProjectedSemantics:
    probabilities: np.ndarray
    ground_probability: np.ndarray
    dynamic: np.ndarray
    valid: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray


def project_semantics_to_points(
    points_lidar: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    *,
    camera_from_lidar: np.ndarray,
    class_count: int,
    ground_class_ids=GROUND_CLASS_IDS,
    dynamic_class_ids=DYNAMIC_CLASS_IDS,
    invalid_class_ids=frozenset(),
) -> ProjectedSemantics:
    points = np.asarray(points_lidar, dtype=np.float64)
    label_image = np.asarray(labels)
    confidence_image = np.asarray(confidence, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    transform = validate_transform(camera_from_lidar)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_lidar must have shape [N,3]")
    if label_image.ndim != 2 or confidence_image.shape != label_image.shape:
        raise ValueError("semantic label and confidence images must match")
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3,3]")
    if class_count <= 0:
        raise ValueError("class_count must be positive")
    if np.any((confidence_image < 0.0) | (confidence_image > 1.0)):
        raise ValueError("semantic confidence must be in [0,1]")

    count = points.shape[0]
    probabilities = np.zeros((count, class_count), dtype=np.float32)
    probabilities[:, 0] = 1.0
    point_labels = np.zeros(count, dtype=np.int32)
    point_confidence = np.zeros(count, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)

    camera_points = transform_points(points, transform)
    positive_depth = camera_points[:, 2] > 1e-6
    projected = camera_points @ camera_matrix.T
    u = np.zeros(count, dtype=np.int64)
    v = np.zeros(count, dtype=np.int64)
    u[positive_depth] = np.rint(
        projected[positive_depth, 0] / projected[positive_depth, 2]
    ).astype(np.int64)
    v[positive_depth] = np.rint(
        projected[positive_depth, 1] / projected[positive_depth, 2]
    ).astype(np.int64)
    height, width = label_image.shape
    valid = (
        positive_depth
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    indices = np.flatnonzero(valid)
    if indices.size:
        sampled_labels = label_image[v[indices], u[indices]].astype(np.int64)
        if np.any((sampled_labels < 0) | (sampled_labels >= class_count)):
            raise ValueError("semantic label exceeds class_count")
        sampled_confidence = confidence_image[v[indices], u[indices]]
        probabilities[indices] = 0.0
        probabilities[indices, 0] = 1.0 - sampled_confidence
        probabilities[indices, sampled_labels] += sampled_confidence
        point_labels[indices] = sampled_labels
        point_confidence[indices] = sampled_confidence
        invalid = np.isin(sampled_labels, tuple(invalid_class_ids))
        if np.any(invalid):
            invalid_indices = indices[invalid]
            valid[invalid_indices] = False
            probabilities[invalid_indices] = 0.0
            probabilities[invalid_indices, 0] = 1.0
            point_confidence[invalid_indices] = 0.0

    ground_probability = np.isin(
        point_labels,
        tuple(ground_class_ids),
    ).astype(np.float32) * valid.astype(np.float32)
    dynamic = valid & np.isin(point_labels, tuple(dynamic_class_ids))
    return ProjectedSemantics(
        probabilities=probabilities,
        ground_probability=ground_probability,
        dynamic=dynamic,
        valid=valid,
        labels=point_labels,
        confidence=point_confidence,
    )
