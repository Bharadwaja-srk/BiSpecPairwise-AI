"""Safety score (Zhang et al. 2024, verbatim).

d = expr_tumor - expr_normal (per target)
score_safe = dx * dy / (dx + dy)   (harmonic-mean form; higher = safer)

Assumptions: inputs are mean expressions (same units/scale) for tumor and
adjacent normal. Negative d (higher normal expression) lowers safety.
dx + dy == 0 -> returns 0.0 (documented edge case).
"""
from __future__ import annotations


def tumor_normal_diff(expr_tumor: float, expr_normal: float) -> float:
    return float(expr_tumor) - float(expr_normal)


def safety_score(
    expr_tumor_1: float,
    expr_normal_1: float,
    expr_tumor_2: float,
    expr_normal_2: float,
) -> float:
    dx = tumor_normal_diff(expr_tumor_1, expr_normal_1)
    dy = tumor_normal_diff(expr_tumor_2, expr_normal_2)
    denom = dx + dy
    if denom == 0:
        return 0.0
    return float(dx * dy / denom)
