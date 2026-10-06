"""Pearson correlation between target expressions (Zhang et al. 2024).

Guards: mismatched shapes -> ValueError; <3 valid cells -> NaN;
zero variance -> 0.0 (documented); NaNs dropped pairwise.
"""
from __future__ import annotations

import numpy as np


def pearson_correlation(expr1: np.ndarray, expr2: np.ndarray) -> float:
    a = np.asarray(expr1, dtype=float).ravel()
    b = np.asarray(expr2, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    c = np.corrcoef(a, b)
    return float(c[0, 1])
