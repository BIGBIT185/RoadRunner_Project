"""Small backend-agnostic helpers for conservative risk statistics."""

from __future__ import annotations


def histogram_cvar(xp, histogram, weight, alpha: float):
    """Compute an upper-tail CVaR from equally spaced risk histogram bins.

    The upper edge of each occupied bin is used, which intentionally makes the
    estimate conservative for both NumPy and CuPy backends.
    """

    tail_mass = (1.0 - float(alpha)) * weight
    remaining = tail_mass.copy()
    tail_sum = xp.zeros_like(weight, dtype=xp.float32)
    bin_count = len(histogram)
    for index in range(bin_count - 1, -1, -1):
        take = xp.minimum(histogram[index], remaining)
        tail_sum += take * (float(index + 1) / float(bin_count))
        remaining = xp.maximum(remaining - take, 0.0)
    return xp.where(
        tail_mass > 1e-12,
        tail_sum / xp.maximum(tail_mass, 1e-12),
        0.0,
    )


def weighted_risk_histogram(xp, cells, risks, weights, cell_count, bin_count):
    """Group qualified point-risk mass by (risk bin, cell), before averaging.

    Inputs are already filtered by the caller. A single bincount preserves
    within-cell multimodality and keeps CPU/GPU distributions consistent with
    their existing confidence-weighted first and second moments.
    """
    bins = xp.minimum((risks * bin_count).astype(xp.int32), bin_count - 1)
    keys = bins.astype(xp.int64) * cell_count + cells
    return xp.bincount(keys, weights=weights, minlength=bin_count * cell_count).reshape(
        bin_count, cell_count
    ).astype(xp.float32)
