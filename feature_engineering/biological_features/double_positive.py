"""Double-positive proportion (Zhang et al. 2024).

P_g1,g2 = mean(I(expr1 > T) * I(expr2 > T)). Threshold configurable.
"""
from __future__ import annotations

import numpy as np


def double_positive_proportion(
    expr1: np.ndarray, expr2: np.ndarray, threshold: float = 0.0
) -> float:
    a = np.asarray(expr1, dtype=float).ravel()
    b = np.asarray(expr2, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan")
    return float(np.mean((a > threshold) & (b > threshold)))


def single_proportions(
    expr1: np.ndarray, expr2: np.ndarray, threshold: float = 0.0
) -> dict:
    from .target_activity import positive_proportion

    p1 = positive_proportion(expr1, threshold)
    p2 = positive_proportion(expr2, threshold)
    return {
        "single_ratio_1": p1,
        "single_ratio_2": p2,
        "single_ratio_sum": float(np.nansum([p1, p2])),
        "single_ratio_min": float(np.nanmin([p1, p2])),
        "single_ratio_max": float(np.nanmax([p1, p2])),
    }
