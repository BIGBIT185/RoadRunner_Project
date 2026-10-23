"""CUDA batched SVD/IRLS solver for packed local-ground neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from .ground_fit import DEFAULT_GROUND_FIT_CONTRACT, GroundFitContract
from .ground_fit import fit_ground_plane_irls_cpu
from .geometry_snapshot import (
    PackedGeometrySnapshot,
    _combine_geometry_risks,
    _smoothstep,
    _weighted_quantile,
)


@dataclass(frozen=True)
class CudaGroundFitResult:
    coefficients: torch.Tensor
    residuals: torch.Tensor
    mean_abs_residual: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class CudaChunkPolicy:
    cell_chunk_size: int = 2048
    point_chunk_size: int = 2_000_000
    min_cell_chunk_size: int = 128
    min_point_chunk_size: int = 50_000
    memory_limit_bytes: int = 20 * 1024 ** 3

    def __post_init__(self) -> None:
        values = (
            self.cell_chunk_size,
            self.point_chunk_size,
            self.min_cell_chunk_size,
            self.min_point_chunk_size,
            self.memory_limit_bytes,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError("CUDA chunk and memory limits must be positive")
        if self.cell_chunk_size < self.min_cell_chunk_size:
            raise ValueError("cell chunk cannot be below its minimum")
        if self.point_chunk_size < self.min_point_chunk_size:
            raise ValueError("point chunk cannot be below its minimum")

    def after_oom(self) -> "CudaChunkPolicy":
        if self.cell_chunk_size > self.min_cell_chunk_size:
            return CudaChunkPolicy(
                cell_chunk_size=max(
                    self.min_cell_chunk_size, self.cell_chunk_size // 2
                ),
                point_chunk_size=self.point_chunk_size,
                min_cell_chunk_size=self.min_cell_chunk_size,
                min_point_chunk_size=self.min_point_chunk_size,
                memory_limit_bytes=self.memory_limit_bytes,
            )
        if self.point_chunk_size > self.min_point_chunk_size:
            return CudaChunkPolicy(
                cell_chunk_size=self.cell_chunk_size,
                point_chunk_size=max(
                    self.min_point_chunk_size, self.point_chunk_size // 2
                ),
                min_cell_chunk_size=self.min_cell_chunk_size,
                min_point_chunk_size=self.min_point_chunk_size,
                memory_limit_bytes=self.memory_limit_bytes,
            )
        raise RuntimeError("CUDA OOM at minimum cell and point chunk sizes")


def estimate_cuda_batch_bytes(cell_count: int, padded_point_count: int) -> int:
    """Conservatively estimate float64 solver storage with a 20% margin."""

    if cell_count <= 0 or padded_point_count <= 0:
        raise ValueError("CUDA batch dimensions must be positive")
    # Inputs, design matrices, IRLS double buffers, residual workspaces and SVD.
    per_point_float64_values = 16
    point_storage = (
        int(cell_count)
        * int(padded_point_count)
        * per_point_float64_values
        * np.dtype(np.float64).itemsize
    )
    per_cell_storage = int(cell_count) * 64 * np.dtype(np.float64).itemsize
    return int(np.ceil((point_storage + per_cell_storage) * 1.20))


def run_with_cuda_oom_retry(operation, policy: CudaChunkPolicy):
    """Retry one complete target after CUDA OOM using deterministic chunk steps."""

    current = policy
    while True:
        try:
            return operation(current), current
        except RuntimeError as error:
            if "cuda" not in str(error).lower() or "out of memory" not in str(error).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            current = current.after_oom()


def _as_tensor(value, *, device, dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _batched_svd_solve(
    matrix: torch.Tensor,
    values: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    scale = torch.maximum(counts, torch.full_like(counts, 3)).to(torch.float64)
    tolerance = scale * torch.finfo(torch.float64).eps * singular[:, 0]
    retained = singular > tolerance[:, None]
    rank = retained.sum(dim=1)
    inverse = torch.where(retained, singular.reciprocal(), torch.zeros_like(singular))
    projected = torch.matmul(u.transpose(1, 2), values.unsqueeze(-1)).squeeze(-1)
    coefficients = torch.matmul(
        vh.transpose(1, 2), (projected * inverse).unsqueeze(-1)
    ).squeeze(-1)
    valid = (rank == 3) & torch.isfinite(coefficients).all(dim=1)
    return coefficients, valid


def _batched_normal_equation_solve(
    matrix: torch.Tensor,
    values: torch.Tensor,
    counts: torch.Tensor,
    *,
    condition_limit: float = 1.0e10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve 3-variable least squares cheaply and fall back to SVD if unsafe."""

    gram = torch.matmul(matrix.transpose(1, 2), matrix)
    rhs = torch.matmul(matrix.transpose(1, 2), values.unsqueeze(-1)).squeeze(-1)
    eigenvalues = torch.linalg.eigvalsh(gram)
    largest = eigenvalues[:, -1]
    smallest = eigenvalues[:, 0]
    well_conditioned = (
        torch.isfinite(eigenvalues).all(dim=1)
        & (largest > 0.0)
        & (smallest > largest / condition_limit)
        & (counts >= 3)
    )
    cholesky, info = torch.linalg.cholesky_ex(gram)
    well_conditioned &= info == 0
    coefficients = torch.zeros(
        (matrix.shape[0], matrix.shape[2]), dtype=matrix.dtype, device=matrix.device
    )
    if bool(torch.any(well_conditioned)):
        fast_indices = torch.nonzero(well_conditioned, as_tuple=False).squeeze(1)
        solved = torch.cholesky_solve(
            rhs[fast_indices].unsqueeze(-1), cholesky[fast_indices]
        ).squeeze(-1)
        fast_finite = torch.isfinite(solved).all(dim=1)
        coefficients[fast_indices] = torch.where(
            fast_finite[:, None], solved, torch.zeros_like(solved)
        )
        well_conditioned[fast_indices] &= fast_finite

    fallback = ~well_conditioned
    valid = well_conditioned.clone()
    if bool(torch.any(fallback)):
        fallback_indices = torch.nonzero(fallback, as_tuple=False).squeeze(1)
        solved, solved_valid = _batched_svd_solve(
            matrix[fallback_indices], values[fallback_indices], counts[fallback_indices]
        )
        coefficients[fallback_indices] = solved
        valid[fallback_indices] = solved_valid
    return coefficients, valid


def fit_ground_planes_cuda(
    points,
    weights,
    counts,
    *,
    device: str = "cuda",
    contract: GroundFitContract = DEFAULT_GROUND_FIT_CONTRACT,
    solver_backend: str = "normal-equation",
) -> CudaGroundFitResult:
    """Fit a padded batch of local planes with the frozen solver contract."""

    target_device = torch.device(device)
    if solver_backend not in {"svd", "normal-equation"}:
        raise ValueError("solver_backend must be 'svd' or 'normal-equation'")
    solve = (
        _batched_svd_solve
        if solver_backend == "svd"
        else _batched_normal_equation_solve
    )
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA ground fitting requires an available CUDA device")
    points_tensor = _as_tensor(points, device=target_device, dtype=torch.float64)
    weights_tensor = _as_tensor(weights, device=target_device, dtype=torch.float64)
    counts_tensor = _as_tensor(counts, device=target_device, dtype=torch.int64)
    if points_tensor.ndim != 3 or points_tensor.shape[2] != 3:
        raise ValueError("points must have shape [B,N,3]")
    batch, maximum, _ = points_tensor.shape
    if weights_tensor.shape != (batch, maximum) or counts_tensor.shape != (batch,):
        raise ValueError("weights/counts must match padded points")
    if torch.any(counts_tensor < 0) or torch.any(counts_tensor > maximum):
        raise ValueError("counts must lie within the padded point dimension")
    if not torch.isfinite(points_tensor).all() or not torch.isfinite(weights_tensor).all():
        raise ValueError("CUDA ground fit inputs must be finite")
    if torch.any(weights_tensor < 0):
        raise ValueError("CUDA ground fit weights must be non-negative")

    indices = torch.arange(maximum, device=target_device)[None, :]
    mask = indices < counts_tensor[:, None]
    design = torch.stack(
        (
            points_tensor[:, :, 0],
            points_tensor[:, :, 1],
            mask.to(torch.float64),
        ),
        dim=2,
    )
    design = design * mask[:, :, None]
    _, unweighted_valid = solve(
        design, points_tensor[:, :, 2] * mask, counts_tensor
    )
    base_weights = torch.where(
        mask,
        torch.clamp(weights_tensor, min=contract.min_weight),
        torch.zeros_like(weights_tensor),
    )
    robust_weights = base_weights.clone()
    coefficients = torch.zeros((batch, 3), dtype=torch.float64, device=target_device)
    active = unweighted_valid & (counts_tensor >= 3)
    for _ in range(contract.max_iterations):
        sqrt_weights = torch.sqrt(robust_weights)
        weighted_design = design * sqrt_weights[:, :, None]
        weighted_z = points_tensor[:, :, 2] * sqrt_weights
        solved, weighted_valid = solve(
            weighted_design, weighted_z, counts_tensor
        )
        current_active = active & weighted_valid
        coefficients = torch.where(
            current_active[:, None], solved, coefficients
        )
        prediction = torch.sum(design * coefficients[:, None, :], dim=2)
        residuals = points_tensor[:, :, 2] - prediction
        huber = torch.minimum(
            torch.ones_like(residuals),
            contract.residual_scale / torch.clamp(torch.abs(residuals), min=1e-6),
        )
        updated = torch.where(
            mask,
            torch.clamp(base_weights * huber, min=contract.min_weight),
            torch.zeros_like(base_weights),
        )
        close = torch.abs(updated - robust_weights) <= (
            contract.convergence_atol
            + contract.convergence_rtol * torch.abs(robust_weights)
        )
        converged = torch.all(close | ~mask, dim=1)
        robust_weights = torch.where(
            current_active[:, None], updated, robust_weights
        )
        active = current_active & ~converged
        unweighted_valid = unweighted_valid & weighted_valid
        if not bool(torch.any(active)):
            break

    prediction = torch.sum(design * coefficients[:, None, :], dim=2)
    residuals = (points_tensor[:, :, 2] - prediction) * mask
    weight_sum = robust_weights.sum(dim=1)
    valid = unweighted_valid & (counts_tensor >= 3) & (weight_sum > 0.0)
    mean_abs = torch.where(
        valid,
        (torch.abs(residuals) * robust_weights).sum(dim=1)
        / torch.clamp(weight_sum, min=torch.finfo(torch.float64).tiny),
        torch.zeros_like(weight_sum),
    )
    coefficients = torch.where(valid[:, None], coefficients, torch.zeros_like(coefficients))
    residuals = torch.where(valid[:, None], residuals, torch.zeros_like(residuals))
    return CudaGroundFitResult(
        coefficients=coefficients,
        residuals=residuals,
        mean_abs_residual=mean_abs,
        valid=valid,
    )


def _cell_neighborhood(
    packed: PackedGeometrySnapshot,
    row: int,
    column: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = packed.map_spec.shape
    points, weights = [], []
    for neighbor_row in range(max(0, row - radius), min(height, row + radius + 1)):
        for neighbor_column in range(
            max(0, column - radius), min(width, column + radius + 1)
        ):
            cell_points = packed.cell_points(neighbor_row, neighbor_column)
            if cell_points.size == 0:
                continue
            distance = np.linalg.norm(
                [neighbor_row - row, neighbor_column - column]
            )
            factor = np.exp(-(distance / max(radius, 1)) ** 2)
            points.append(cell_points)
            weights.append(
                packed.cell_weights(neighbor_row, neighbor_column) * factor
            )
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(points), np.concatenate(weights)


def _fit_packed_geometry_cuda_once(
    packed: PackedGeometrySnapshot,
    *,
    device: str = "cuda",
    solver_backend: str = "normal-equation",
    cell_chunk_size: int = 2048,
    point_chunk_size: int = 2_000_000,
    memory_limit_bytes: int = 20 * 1024 ** 3,
    neighborhood_cells: int = 7,
    min_support_points: int = 4,
    ground_fit_max_iterations: int = 10,
    ground_fit_residual_scale: float = 0.10,
    max_fit_residual: float = 0.20,
    max_geometry_age_sec: float = 5.0,
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
    min_elevation_m: float = -2.0,
    max_elevation_m: float = 3.0,
    profiler=None,
    profile_fields: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    """Fit every observed packed BEV cell with chunked CUDA SVD/IRLS."""

    if cell_chunk_size <= 0 or point_chunk_size <= 0:
        raise ValueError("cell_chunk_size and point_chunk_size must be positive")
    height, width = packed.map_spec.shape
    shape = (height, width)
    observed = packed.observation_count > 0
    output = {
        name: np.zeros(shape, dtype=np.float32)
        for name in (
            "geometry_risk", "geometry_confidence", "slope",
            "longitudinal_slope", "lateral_slope", "roughness",
            "positive_step_height", "negative_step_height", "non_ground_ratio",
            "ground_fit_residual", "slope_risk", "roughness_risk",
            "positive_step_risk", "negative_step_risk",
        )
    }
    support_count = np.zeros(shape, dtype=np.int32)
    fit_valid = np.zeros(shape, dtype=bool)
    fitted_coefficients = np.zeros((height, width, 3), dtype=np.float64)
    fitted_residuals = np.zeros(shape, dtype=np.float64)
    solver_valid = np.zeros(shape, dtype=bool)
    fallback_cells: list[tuple[int, int]] = []
    radius = max(0, int(neighborhood_cells) // 2)
    contract = GroundFitContract(
        max_iterations=max(1, int(ground_fit_max_iterations)),
        residual_scale=max(float(ground_fit_residual_scale), 1e-6),
    )
    cells = [(int(row), int(column)) for row, column in zip(*np.nonzero(observed))]
    candidate_count = 0
    neighborhood_sec = 0.0
    batch_pack_sec = 0.0
    cuda_fit_and_transfer_sec = 0.0
    fallback_sec = 0.0
    for start in range(0, len(cells), cell_chunk_size):
        observed_chunk = cells[start:start + cell_chunk_size]
        neighborhoods = []
        chunk = []
        stage_started = time.perf_counter()
        for row, column in observed_chunk:
            points, weights = _cell_neighborhood(packed, row, column, radius)
            support_count[row, column] = points.shape[0]
            if points.shape[0] >= min_support_points:
                chunk.append((row, column))
                neighborhoods.append((points, weights))
        neighborhood_sec += time.perf_counter() - stage_started
        candidate_count += len(chunk)
        if not chunk:
            continue
        maximum = max(points.shape[0] for points, _ in neighborhoods)
        cells_per_point_batch = max(1, point_chunk_size // maximum)
        for batch_start in range(0, len(chunk), cells_per_point_batch):
            batch_cells = chunk[batch_start:batch_start + cells_per_point_batch]
            batch_neighborhoods = neighborhoods[
                batch_start:batch_start + cells_per_point_batch
            ]
            batch_maximum = max(
                points.shape[0] for points, _ in batch_neighborhoods
            )
            estimate = estimate_cuda_batch_bytes(
                len(batch_cells), batch_maximum
            )
            if estimate > memory_limit_bytes:
                raise RuntimeError(
                    "CUDA out of memory preflight: estimated "
                    f"{estimate} bytes exceeds {memory_limit_bytes}"
                )
            stage_started = time.perf_counter()
            point_batch = np.zeros(
                (len(batch_cells), batch_maximum, 3), dtype=np.float64
            )
            weight_batch = np.zeros(
                (len(batch_cells), batch_maximum), dtype=np.float64
            )
            count_batch = np.zeros(len(batch_cells), dtype=np.int64)
            for index, (points, weights) in enumerate(batch_neighborhoods):
                count = points.shape[0]
                point_batch[index, :count] = points
                weight_batch[index, :count] = weights
                count_batch[index] = count
            batch_pack_sec += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            result = fit_ground_planes_cuda(
                point_batch,
                weight_batch,
                count_batch,
                device=device,
                contract=contract,
                solver_backend=solver_backend,
            )
            coefficients = result.coefficients.cpu().numpy()
            residuals = result.mean_abs_residual.cpu().numpy()
            valid_batch = result.valid.cpu().numpy()
            cuda_fit_and_transfer_sec += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            for index, (row, column) in enumerate(batch_cells):
                if valid_batch[index]:
                    fitted_coefficients[row, column] = coefficients[index]
                    fitted_residuals[row, column] = residuals[index]
                    solver_valid[row, column] = True
                    continue
                points, weights = batch_neighborhoods[index]
                cpu_coefficients, _, cpu_residual, cpu_valid = (
                    fit_ground_plane_irls_cpu(points, weights, contract=contract)
                )
                if cpu_valid:
                    fitted_coefficients[row, column] = cpu_coefficients
                    fitted_residuals[row, column] = cpu_residual
                    solver_valid[row, column] = True
                    fallback_cells.append((row, column))
            fallback_sec += time.perf_counter() - stage_started

    risk_started = time.perf_counter()
    slope_safe, slope_critical = np.deg2rad([slope_safe_deg, slope_critical_deg])
    lateral_safe, lateral_critical = np.deg2rad(
        [lateral_slope_safe_deg, lateral_slope_critical_deg]
    )
    elevation = packed.elevation.copy()
    for row, column in cells:
        if not solver_valid[row, column]:
            continue
        residual = fitted_residuals[row, column]
        if residual > max_fit_residual:
            continue
        a, b, c = fitted_coefficients[row, column]
        center_x = packed.map_spec.x_max - (row + 0.5) * packed.map_spec.resolution
        center_y = packed.map_spec.y_max - (column + 0.5) * packed.map_spec.resolution
        fitted = float(a * center_x + b * center_y + c)
        if not min_elevation_m <= fitted <= max_elevation_m:
            continue
        cell_points = packed.cell_points(row, column)
        cell_weights = packed.cell_weights(row, column)
        if cell_points.size == 0:
            continue
        cell_residual = cell_points[:, 2] - (
            a * cell_points[:, 0] + b * cell_points[:, 1] + c
        )
        median = _weighted_quantile(cell_residual, cell_weights, 0.5)
        mad = 1.4826 * _weighted_quantile(
            np.abs(cell_residual - median), cell_weights, 0.5
        )
        positive_step = max(
            0.0, _weighted_quantile(cell_residual, cell_weights, 0.95)
        )
        negative_step = max(
            0.0, _weighted_quantile(-cell_residual, cell_weights, 0.95)
        )
        weight_sum = max(float(np.sum(cell_weights)), 1e-12)
        slope = float(np.arctan(np.sqrt(a * a + b * b)))
        longitudinal = float(np.arctan(abs(a)))
        lateral = float(np.arctan(abs(b)))
        risks = (
            max(
                _smoothstep(slope_safe, slope_critical, slope),
                _smoothstep(lateral_safe, lateral_critical, lateral),
            ),
            _smoothstep(roughness_safe_m, roughness_critical_m, mad),
            _smoothstep(
                positive_step_safe_m, positive_step_critical_m, positive_step
            ),
            _smoothstep(
                negative_step_safe_m, negative_step_critical_m, negative_step
            ),
        )
        elevation[row, column] = fitted
        values = (
            ("slope", slope), ("longitudinal_slope", longitudinal),
            ("lateral_slope", lateral), ("roughness", mad),
            ("positive_step_height", positive_step),
            ("negative_step_height", negative_step),
            ("non_ground_ratio", max(
                float(np.sum(cell_weights[cell_residual > positive_step_safe_m]))
                / weight_sum,
                float(np.sum(cell_weights[cell_residual < -negative_step_safe_m]))
                / weight_sum,
            )),
            ("ground_fit_residual", residual), ("slope_risk", risks[0]),
            ("roughness_risk", risks[1]), ("positive_step_risk", risks[2]),
            ("negative_step_risk", risks[3]),
            ("geometry_risk", _combine_geometry_risks(*risks)),
        )
        for name, value in values:
            output[name][row, column] = value
        confidence = (
            (1.0 - np.exp(-support_count[row, column] / 12.0))
            * np.exp(
                -packed.geometry_age[row, column]
                / max(max_geometry_age_sec, 1e-6)
            )
            * np.exp(-residual / max(ground_fit_residual_scale, 1e-6))
            / (1.0 + packed.conflict_score[row, column])
        )
        output["geometry_confidence"][row, column] = np.clip(
            confidence, 0.0, 1.0
        )
        fit_valid[row, column] = True

    semantic_ids = np.full(shape, -1, dtype=np.int32)
    semantic_confidence = np.zeros(shape, dtype=np.float32)
    semantic_valid = packed.semantic_weight > 0.0
    probabilities = np.zeros_like(packed.semantic_sum)
    probabilities[semantic_valid] = (
        packed.semantic_sum[semantic_valid]
        / packed.semantic_weight[semantic_valid, None]
    )
    semantic_ids[semantic_valid] = np.argmax(probabilities[semantic_valid], axis=1)
    semantic_confidence[semantic_valid] = np.max(
        probabilities[semantic_valid], axis=1
    )
    fallback_array = np.asarray(fallback_cells, dtype=np.int32).reshape(-1, 2)
    output.update({
        "elevation": elevation,
        "geometry_reliability": output["geometry_confidence"],
        "semantic_ids": semantic_ids,
        "semantic_confidence": semantic_confidence,
        "observation_count": packed.observation_count,
        "last_observed_time": packed.last_observed_time.astype(np.float32),
        "observed_mask": observed,
        "geometry_valid_mask": (output["geometry_confidence"] > 0).astype(np.float32),
        "geometry_support_count": support_count,
        "geometry_age": packed.geometry_age,
        "geometry_conflict_score": packed.conflict_score,
        "ground_fit_valid": fit_valid.astype(np.float32),
        "geometry_version": np.asarray("v2_local_ground_multi_factor"),
        "cuda_fallback_cell_ids": fallback_array,
        "cuda_fallback_ratio": float(len(fallback_cells) / max(candidate_count, 1)),
        "cuda_cell_chunk_size": int(cell_chunk_size),
        "cuda_point_chunk_size": int(point_chunk_size),
        "cuda_solver_backend": np.asarray(solver_backend),
    })
    if profiler is not None:
        fields = {} if profile_fields is None else dict(profile_fields)
        profiler.record(
            "geometry_cuda_breakdown",
            neighborhood_sec=neighborhood_sec,
            batch_pack_sec=batch_pack_sec,
            cuda_fit_and_transfer_sec=cuda_fit_and_transfer_sec,
            fallback_sec=fallback_sec,
            risk_postprocess_sec=time.perf_counter() - risk_started,
            candidate_cells=candidate_count,
            **fields,
        )
    return output


def fit_packed_geometry_cuda(
    packed: PackedGeometrySnapshot,
    *,
    device: str = "cuda",
    cell_chunk_size: int = 2048,
    point_chunk_size: int = 2_000_000,
    min_cell_chunk_size: int = 128,
    min_point_chunk_size: int = 50_000,
    memory_limit_bytes: int = 20 * 1024 ** 3,
    profiler=None,
    profile_fields: dict[str, object] | None = None,
    **geometry_options,
) -> dict[str, np.ndarray]:
    """Run one packed target with deterministic CUDA OOM retries."""

    initial_policy = CudaChunkPolicy(
        cell_chunk_size=cell_chunk_size,
        point_chunk_size=point_chunk_size,
        min_cell_chunk_size=min_cell_chunk_size,
        min_point_chunk_size=min_point_chunk_size,
        memory_limit_bytes=memory_limit_bytes,
    )

    def operation(policy: CudaChunkPolicy):
        return _fit_packed_geometry_cuda_once(
            packed,
            device=device,
            cell_chunk_size=policy.cell_chunk_size,
            point_chunk_size=policy.point_chunk_size,
            memory_limit_bytes=policy.memory_limit_bytes,
            profiler=profiler,
            profile_fields=profile_fields,
            **geometry_options,
        )

    output, _ = run_with_cuda_oom_retry(operation, initial_policy)
    return output
