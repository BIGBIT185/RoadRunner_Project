"""共享 GPU 滚动栅格状态。

几何高程层和语义层使用同一个世界系滚动窗口。窗口中心移动时，所有注册
图层一起执行 ``roll`` 和边缘清理；点云到栅格的索引也只由这个类计算，避免
两条流水线各自维护一套滚动索引。
"""

from __future__ import annotations

from threading import RLock
from typing import List, Tuple

import numpy as np


def bilinear_world_xy_samples(
    world_x,
    world_y,
    *,
    center: np.ndarray,
    world_map_length: float,
    resolution: float,
    cell_n: int,
    xp,
):
    """Return four world-grid neighbors and their valid interpolation mass.

    ``world_xy_indices`` intentionally keeps the historical nearest-neighbor
    behavior.  This companion uses cell-center coordinates so a sub-cell pose
    change changes the weights continuously instead of switching every output
    cell to a new integer key.  Neighbors outside the rolling map contribute
    zero weight; callers can reject samples whose remaining mass is too small.
    The function accepts NumPy or CuPy as ``xp`` so the indexing convention is
    tested without requiring a CUDA device.
    """

    if not np.isfinite(world_map_length) or world_map_length <= 0.0:
        raise ValueError("world_map_length must be finite and positive")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be finite and positive")
    if int(cell_n) <= 0:
        raise ValueError("cell_n must be positive")
    world_x = xp.asarray(world_x)
    world_y = xp.asarray(world_y)
    if world_x.shape != world_y.shape:
        raise ValueError("world_x and world_y must have the same shape")
    center = np.asarray(center, dtype=np.float64).reshape(2)
    if not np.isfinite(center).all():
        raise ValueError("rolling-grid center must be finite")

    # Convert world coordinates to floating cell-center coordinates.  The
    # existing row convention grows toward decreasing world x/y, hence the
    # subtraction from the positive map edge.
    row_coordinate = (
        center[0] + 0.5 * float(world_map_length) - world_x
    ) / float(resolution) - 0.5
    column_coordinate = (
        center[1] + 0.5 * float(world_map_length) - world_y
    ) / float(resolution) - 0.5
    row_base = xp.floor(row_coordinate).astype(xp.int64)
    column_base = xp.floor(column_coordinate).astype(xp.int64)
    row_fraction = row_coordinate - row_base
    column_fraction = column_coordinate - column_base

    rows = xp.stack(
        (row_base, row_base, row_base + 1, row_base + 1),
        axis=0,
    )
    columns = xp.stack(
        (column_base, column_base + 1, column_base, column_base + 1),
        axis=0,
    )
    weights = xp.stack(
        (
            (1.0 - row_fraction) * (1.0 - column_fraction),
            (1.0 - row_fraction) * column_fraction,
            row_fraction * (1.0 - column_fraction),
            row_fraction * column_fraction,
        ),
        axis=0,
    )
    finite = xp.isfinite(world_x) & xp.isfinite(world_y)
    valid_neighbors = (
        finite[None, ...]
        & (rows >= 0)
        & (rows < int(cell_n))
        & (columns >= 0)
        & (columns < int(cell_n))
    )
    weights = xp.where(valid_neighbors, weights, 0.0)
    valid_weight = xp.sum(weights, axis=0)
    rows = xp.clip(rows, 0, int(cell_n) - 1)
    columns = xp.clip(columns, 0, int(cell_n) - 1)
    return rows, columns, weights, valid_weight


def weighted_neighbor_sample(
    values,
    rows,
    columns,
    weights,
    *,
    support=None,
    min_valid_weight: float = 0.5,
    xp,
    fill_value=0.0,
):
    """Sample a grid using normalized weights over observed neighbors only."""

    if not np.isfinite(min_valid_weight) or not 0.0 <= min_valid_weight <= 1.0:
        raise ValueError("min_valid_weight must be finite and in [0,1]")
    values = xp.asarray(values)
    rows = xp.asarray(rows)
    columns = xp.asarray(columns)
    weights = xp.asarray(weights)
    if rows.shape != columns.shape or rows.shape != weights.shape:
        raise ValueError("neighbor indices and weights must have matching shapes")
    gathered = values[rows, columns]
    if support is None:
        supported = xp.ones_like(gathered, dtype=bool)
    else:
        support = xp.asarray(support)
        supported = support[rows, columns]
    effective_weights = xp.where(supported, weights, 0.0)
    valid_weight = xp.sum(effective_weights, axis=0)
    valid = valid_weight >= float(min_valid_weight)
    safe_values = xp.where(supported, gathered, 0.0)
    denominator = xp.maximum(valid_weight, xp.asarray(1e-12, dtype=valid_weight.dtype))
    sampled = xp.sum(safe_values * effective_weights, axis=0) / denominator
    sampled = xp.where(valid, sampled, fill_value)
    return sampled, valid_weight, valid


class GpuRollingGrid:
    """维护共享窗口中心、滚动图层和世界坐标索引。"""

    def __init__(
        self,
        *,
        world_map_length: float,
        resolution: float,
        device,
        cp,
    ) -> None:
        if not np.isfinite(world_map_length) or world_map_length <= 0.0:
            raise ValueError("world_map_length must be finite and positive")
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution must be finite and positive")
        self.cp = cp
        self.device = device
        self.world_map_length = float(world_map_length)
        self.resolution = float(resolution)
        self.cell_n = int(np.ceil(self.world_map_length / self.resolution))
        self.shape = (self.cell_n, self.cell_n)
        self._center = np.zeros(2, dtype=np.float64)
        self._initialized = False
        self._layers: List[Tuple[object, object]] = []
        # 几何线程会滚动同一批语义图层；锁覆盖滚动和读写，避免异步语义线程
        # 在索引计算与图层更新之间看到半滚动状态。
        self.lock = RLock()

    @property
    def center(self) -> np.ndarray:
        return self._center.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def register_layer(self, layer, fill_value) -> None:
        """注册一个与滚动窗口同形状的 GPU 图层。"""

        if tuple(layer.shape) != self.shape:
            raise ValueError(
                f"rolling layer shape must be {self.shape}, got {layer.shape}"
            )
        self._layers.append((layer, fill_value))

    def _clear_shifted_edges(self, layer, row_shift: int, column_shift: int, fill) -> None:
        if row_shift > 0:
            layer[:row_shift, :] = fill
        elif row_shift < 0:
            layer[row_shift:, :] = fill
        if column_shift > 0:
            layer[:, :column_shift] = fill
        elif column_shift < 0:
            layer[:, column_shift:] = fill

    def clear(self) -> None:
        """清空所有注册图层并重置窗口中心。"""

        with self.lock:
            with self.device:
                for layer, fill in self._layers:
                    layer.fill(fill)
            self._center.fill(0.0)
            self._initialized = False

    def move_center(self, target_xy: np.ndarray) -> None:
        """按分辨率移动窗口，并同步滚动所有注册图层。"""

        with self.lock:
            target = np.asarray(target_xy, dtype=np.float64).reshape(2)
            if not np.isfinite(target).all():
                raise ValueError("rolling-grid center must be finite")
            if not self._initialized:
                self._center = target.copy()
                self._initialized = True
                return

            shift = np.rint(
                (target - self._center) / self.resolution
            ).astype(np.int64)
            if not np.any(shift):
                return
            if np.any(np.abs(shift) >= self.cell_n):
                self.clear()
                self._center = target.copy()
                self._initialized = True
                return

            row_shift, column_shift = int(shift[0]), int(shift[1])
            with self.device:
                for layer, fill in self._layers:
                    layer[...] = self.cp.roll(
                        layer,
                        (row_shift, column_shift),
                        axis=(0, 1),
                    )
                    self._clear_shifted_edges(layer, row_shift, column_shift, fill)
            self._center += shift.astype(np.float64) * self.resolution

    def world_xy_indices(self, world_x, world_y):
        """将世界平面坐标转换为共享滚动窗口的行列索引。"""

        with self.lock:
            world_x = self.cp.asarray(world_x)
            world_y = self.cp.asarray(world_y)
            if world_x.shape != world_y.shape:
                raise ValueError("world_x and world_y must have the same shape")
            rows = self.cp.floor(
                (self._center[0] + 0.5 * self.world_map_length - world_x)
                / self.resolution
            ).astype(self.cp.int64)
            columns = self.cp.floor(
                (self._center[1] + 0.5 * self.world_map_length - world_y)
                / self.resolution
            ).astype(self.cp.int64)
            valid = (
                self.cp.isfinite(world_x)
                & self.cp.isfinite(world_y)
                & (rows >= 0)
                & (rows < self.cell_n)
                & (columns >= 0)
                & (columns < self.cell_n)
            )
            return rows, columns, valid

    def world_indices(self, points):
        """将 ``[N,3+]`` 世界系点云映射到共享行列索引。"""

        points = self.cp.asarray(points)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("world points must have shape [N,C>=2]")
        rows, columns, valid = self.world_xy_indices(points[:, 0], points[:, 1])
        valid &= self.cp.isfinite(points).all(axis=1)
        return rows, columns, valid
