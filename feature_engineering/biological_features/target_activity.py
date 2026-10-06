"""Target activity features (Zhang et al. 2024).

P_gene = mean(I(expr > T)) within a cell population; plus mean expression.
Cell-type-specific values computed by grouping on an obs column.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def positive_proportion(expr: np.ndarray, threshold: float = 0.0) -> float:
    expr = np.asarray(expr, dtype=float).ravel()
    expr = expr[~np.isnan(expr)]
    if expr.size == 0:
        return float("nan")
    return float(np.mean(expr > threshold))


def mean_expression(expr: np.ndarray) -> float:
    expr = np.asarray(expr, dtype=float).ravel()
    expr = expr[~np.isnan(expr)]
    if expr.size == 0:
        return float("nan")
    return float(np.mean(expr))


def celltype_activity(
    expr: np.ndarray,
    labels: Sequence[str],
    threshold: float = 0.0,
) -> dict:
    """Return {celltype: {'mean': f, 'positive_proportion': f, 'n_cells': n}}."""
    expr = np.asarray(expr, dtype=float).ravel()
    out: dict = {}
    for ct in sorted(set(labels)):
        mask = np.array([l == ct for l in labels])
        vals = expr[mask]
        out[str(ct)] = {
            "mean": mean_expression(vals),
            "positive_proportion": positive_proportion(vals, threshold),
            "n_cells": int(mask.sum()),
        }
    return out


def top_expressing_populations(
    expr: np.ndarray, labels: Sequence[str], k: int = 2
) -> list[str]:
    act = celltype_activity(expr, labels)
    ranked = sorted(act.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    return [ct for ct, _ in ranked[:k]]
