"""Versioned numerical contract for robust local-ground plane fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundFitContract:
    """Parameters shared by the CPU reference and future CUDA implementation."""

    version: str = "svd-irls-v1"
    dtype: str = "float64"
    max_iterations: int = 10
    residual_scale: float = 0.10
    convergence_rtol: float = 1e-3
    convergence_atol: float = 1e-6
    min_weight: float = 1e-6

    def __post_init__(self) -> None:
        if self.dtype != "float64":
            raise ValueError("ground fit dtype must be float64")
        if self.max_iterations <= 0:
            raise ValueError("ground fit max_iterations must be positive")
        finite_non_negative = (
            self.residual_scale,
            self.convergence_rtol,
            self.convergence_atol,
            self.min_weight,
        )
        if not np.isfinite(finite_non_negative).all() or any(
            value < 0.0 for value in finite_non_negative
        ):
            raise ValueError("ground fit tolerances must be finite and non-negative")
        if self.residual_scale == 0.0:
            raise ValueError("ground fit residual_scale must be positive")


DEFAULT_GROUND_FIT_CONTRACT = GroundFitContract()


def _invalid_result(point_count: int) -> tuple[np.ndarray, np.ndarray, float, bool]:
    return (
        np.zeros(3, dtype=np.float64),
        np.zeros(max(0, int(point_count)), dtype=np.float64),
        0.0,
        False,
    )


def _svd_rank(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return u, singular_values, vh, 0
    tolerance = (
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    return u, singular_values, vh, rank


def _svd_solve(matrix: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        u, singular_values, vh, rank = _svd_rank(matrix)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float64), False
    if rank < 3:
        return np.zeros(3, dtype=np.float64), False
    projected = u.T @ values
    coefficients = vh.T @ (projected / singular_values)
    if not np.isfinite(coefficients).all():
        return np.zeros(3, dtype=np.float64), False
    return coefficients.astype(np.float64, copy=False), True


def fit_ground_plane_irls_cpu(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    contract: GroundFitContract = DEFAULT_GROUND_FIT_CONTRACT,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit ``z = ax + by + c`` using the frozen float64 SVD/IRLS contract."""

    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points must have shape [N,3]")
    if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
        raise ValueError("weights must have shape [N] matching points")
    if not np.isfinite(points).all() or not np.isfinite(weights).all():
        raise ValueError("ground fit inputs must be finite")
    if np.any(weights < 0.0):
        raise ValueError("ground fit weights must be non-negative")
    if points.shape[0] < 3:
        return _invalid_result(points.shape[0])

    design = np.column_stack(
        [points[:, 0], points[:, 1], np.ones(points.shape[0], dtype=np.float64)]
    )
    try:
        if _svd_rank(design)[3] < 3:
            return _invalid_result(points.shape[0])
    except np.linalg.LinAlgError:
        return _invalid_result(points.shape[0])

    base_weights = np.clip(weights, contract.min_weight, None)
    robust_weights = base_weights.copy()
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(contract.max_iterations):
        sqrt_weights = np.sqrt(robust_weights)
        weighted_design = design * sqrt_weights[:, None]
        weighted_z = points[:, 2] * sqrt_weights
        coefficients, solved = _svd_solve(weighted_design, weighted_z)
        if not solved:
            return _invalid_result(points.shape[0])
        residuals = points[:, 2] - design @ coefficients
        huber = np.minimum(
            1.0,
            contract.residual_scale / np.maximum(np.abs(residuals), 1e-6),
        )
        updated = np.clip(base_weights * huber, contract.min_weight, None)
        if np.allclose(
            updated,
            robust_weights,
            rtol=contract.convergence_rtol,
            atol=contract.convergence_atol,
        ):
            robust_weights = updated
            break
        robust_weights = updated

    residuals = points[:, 2] - design @ coefficients
    weight_sum = float(np.sum(robust_weights))
    if weight_sum <= 0.0:
        return _invalid_result(points.shape[0])
    mean_abs = float(np.sum(np.abs(residuals) * robust_weights) / weight_sum)
    if not np.isfinite(mean_abs):
        return _invalid_result(points.shape[0])
    return coefficients, residuals, mean_abs, True
