"""Fast geometry teacher derived from one downsampled BEV elevation grid."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry_snapshot import PackedGeometrySnapshot


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (np.asarray(values) >= edge1).astype(np.float32)
    scaled = np.clip(
        (np.asarray(values, dtype=np.float32) - edge0) / (edge1 - edge0),
        0.0,
        1.0,
    )
    return (scaled * scaled * (3.0 - 2.0 * scaled)).astype(np.float32)


def _normalized_box(
    values: np.ndarray,
    valid: np.ndarray,
    kernel_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(valid, dtype=np.float32)
    numerator = cv2.boxFilter(
        np.asarray(values, dtype=np.float32) * mask,
        -1,
        (kernel_size, kernel_size),
        normalize=False,
        borderType=cv2.BORDER_REFLECT,
    )
    count = cv2.boxFilter(
        mask,
        -1,
        (kernel_size, kernel_size),
        normalize=False,
        borderType=cv2.BORDER_REFLECT,
    )
    mean = np.zeros_like(numerator)
    np.divide(numerator, count, out=mean, where=count > 0.0)
    return mean, count


def _non_ground_ratio(
    packed: PackedGeometrySnapshot,
    ground_elevation: np.ndarray,
    threshold: float,
) -> np.ndarray:
    shape = packed.map_spec.shape
    counts = np.diff(np.asarray(packed.cell_offsets, dtype=np.int64))
    if packed.support_points.size == 0 or not np.any(counts):
        return np.zeros(shape, dtype=np.float32)
    flat_cell_ids = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    ground = ground_elevation.reshape(-1)[flat_cell_ids]
    residual = np.asarray(packed.support_points[:, 2]) - ground
    weights = np.asarray(packed.support_weights, dtype=np.float64)
    total = np.bincount(
        flat_cell_ids,
        weights=weights,
        minlength=counts.size,
    )
    non_ground = np.bincount(
        flat_cell_ids,
        weights=weights * (np.abs(residual) > threshold),
        minlength=counts.size,
    )
    ratio = np.zeros(counts.size, dtype=np.float32)
    np.divide(non_ground, total, out=ratio, where=total > 0.0)
    return ratio.reshape(shape)


def fit_packed_geometry_elevation_grid(
    packed: PackedGeometrySnapshot,
    *,
    neighborhood_cells: int = 7,
    min_support_cells: int = 4,
    max_geometry_age_sec: float = 31.0,
    ground_fit_residual_scale: float = 0.10,
    slope_safe_deg: float = 8.0,
    slope_critical_deg: float = 20.0,
    lateral_slope_safe_deg: float = 6.0,
    lateral_slope_critical_deg: float = 15.0,
    roughness_safe_m: float = 0.03,
    roughness_critical_m: float = 0.12,
    positive_step_safe_m: float = 0.05,
    positive_step_critical_m: float = 0.25,
    negative_step_safe_m: float = 0.05,
    negative_step_critical_m: float = 0.25,
) -> dict[str, np.ndarray]:
    """Aggregate elevation once, then derive all geometry risks vectorially."""

    if neighborhood_cells <= 0 or neighborhood_cells % 2 == 0:
        raise ValueError("neighborhood_cells must be a positive odd integer")
    if min_support_cells <= 0:
        raise ValueError("min_support_cells must be positive")
    if max_geometry_age_sec <= 0.0:
        raise ValueError("max_geometry_age_sec must be positive")

    elevation_raw = np.asarray(packed.elevation, dtype=np.float32)
    observed = np.asarray(packed.observation_count) > 0
    local_mean, local_count = _normalized_box(
        elevation_raw,
        observed,
        neighborhood_cells,
    )
    filled = np.where(observed, elevation_raw, local_mean).astype(np.float32)
    ground = cv2.medianBlur(filled, 5)

    resolution = float(packed.map_spec.resolution)
    longitudinal_gradient = cv2.Sobel(
        ground,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
        scale=1.0 / (8.0 * resolution),
        borderType=cv2.BORDER_REFLECT,
    )
    lateral_gradient = cv2.Sobel(
        ground,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        scale=1.0 / (8.0 * resolution),
        borderType=cv2.BORDER_REFLECT,
    )
    longitudinal_slope = np.arctan(np.abs(longitudinal_gradient)).astype(
        np.float32
    )
    lateral_slope = np.arctan(np.abs(lateral_gradient)).astype(np.float32)
    slope = np.arctan(
        np.hypot(longitudinal_gradient, lateral_gradient)
    ).astype(np.float32)

    step_kernel = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(
        np.where(observed, ground, -1e6).astype(np.float32),
        step_kernel,
        borderType=cv2.BORDER_REPLICATE,
    )
    local_min = cv2.erode(
        np.where(observed, ground, 1e6).astype(np.float32),
        step_kernel,
        borderType=cv2.BORDER_REPLICATE,
    )
    positive_step = np.maximum(local_max - ground, 0.0).astype(np.float32)
    negative_step = np.maximum(ground - local_min, 0.0).astype(np.float32)
    positive_step[~observed] = 0.0
    negative_step[~observed] = 0.0

    squared_mean, _ = _normalized_box(
        elevation_raw * elevation_raw,
        observed,
        neighborhood_cells,
    )
    roughness = np.sqrt(
        np.maximum(squared_mean - local_mean * local_mean, 0.0)
    ).astype(np.float32)
    residual = np.abs(elevation_raw - ground).astype(np.float32)
    residual[~observed] = 0.0

    slope_safe, slope_critical = np.deg2rad(
        [slope_safe_deg, slope_critical_deg]
    )
    lateral_safe, lateral_critical = np.deg2rad(
        [lateral_slope_safe_deg, lateral_slope_critical_deg]
    )
    slope_risk = np.maximum(
        _smoothstep(slope_safe, slope_critical, slope),
        _smoothstep(lateral_safe, lateral_critical, lateral_slope),
    )
    roughness_risk = _smoothstep(
        roughness_safe_m,
        roughness_critical_m,
        roughness,
    )
    positive_step_risk = _smoothstep(
        positive_step_safe_m,
        positive_step_critical_m,
        positive_step,
    )
    negative_step_risk = _smoothstep(
        negative_step_safe_m,
        negative_step_critical_m,
        negative_step,
    )

    support_count = np.rint(local_count).astype(np.int32)
    valid = observed & (support_count >= min_support_cells)
    freshness = np.exp(
        -np.asarray(packed.geometry_age, dtype=np.float32)
        / max(max_geometry_age_sec, 1e-6)
    )
    density_confidence = 1.0 - np.exp(-support_count / 12.0)
    residual_confidence = np.exp(
        -residual / max(ground_fit_residual_scale, 1e-6)
    )
    confidence = (
        density_confidence
        * freshness
        * residual_confidence
        / (1.0 + np.asarray(packed.conflict_score, dtype=np.float32))
    )
    confidence = np.where(valid, np.clip(confidence, 0.0, 1.0), 0.0).astype(
        np.float32
    )

    for layer in (
        slope,
        longitudinal_slope,
        lateral_slope,
        roughness,
        positive_step,
        negative_step,
        slope_risk,
        roughness_risk,
        positive_step_risk,
        negative_step_risk,
    ):
        layer[~valid] = 0.0
    geometry_risk = np.maximum.reduce(
        (slope_risk, positive_step_risk, negative_step_risk)
    ).astype(np.float32)

    semantic_ids = np.full(packed.map_spec.shape, -1, dtype=np.int32)
    semantic_confidence = np.zeros(packed.map_spec.shape, dtype=np.float32)
    semantic_valid = np.asarray(packed.semantic_weight) > 0.0
    if np.any(semantic_valid):
        probabilities = np.zeros_like(packed.semantic_sum)
        probabilities[semantic_valid] = (
            packed.semantic_sum[semantic_valid]
            / packed.semantic_weight[semantic_valid, None]
        )
        semantic_ids[semantic_valid] = np.argmax(
            probabilities[semantic_valid],
            axis=1,
        )
        semantic_confidence[semantic_valid] = np.max(
            probabilities[semantic_valid],
            axis=1,
        )

    return {
        "geometry_risk": geometry_risk,
        "geometry_confidence": confidence,
        "slope": slope,
        "longitudinal_slope": longitudinal_slope,
        "lateral_slope": lateral_slope,
        "roughness": roughness,
        "positive_step_height": positive_step,
        "negative_step_height": negative_step,
        "non_ground_ratio": _non_ground_ratio(
            packed,
            ground,
            positive_step_safe_m,
        ),
        "ground_fit_residual": residual,
        "slope_risk": slope_risk,
        "roughness_risk": roughness_risk,
        "positive_step_risk": positive_step_risk,
        "negative_step_risk": negative_step_risk,
        "elevation": np.where(observed, ground, 0.0).astype(np.float32),
        "geometry_reliability": confidence,
        "semantic_ids": semantic_ids,
        "semantic_confidence": semantic_confidence,
        "observation_count": np.asarray(
            packed.observation_count,
            dtype=np.int32,
        ),
        "last_observed_time": np.asarray(
            packed.last_observed_time,
            dtype=np.float32,
        ),
        "observed_mask": observed,
        "geometry_valid_mask": valid.astype(np.float32),
        "geometry_support_count": support_count,
        "geometry_age": np.asarray(packed.geometry_age, dtype=np.float32),
        "geometry_conflict_score": np.asarray(
            packed.conflict_score,
            dtype=np.float32,
        ),
        "ground_fit_valid": valid.astype(np.float32),
        "geometry_version": np.asarray("v3_elevation_grid"),
    }
