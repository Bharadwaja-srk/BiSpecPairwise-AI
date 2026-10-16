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


def rank_margins(
    expr: np.ndarray, labels: Sequence[str], k: int = 2
) -> dict:
    """Top-k vs next-ranked expression-mean margins (P-A diagnostic, read-only).

    Uses the same per-population means as the selection rule but ranks only
    populations with finite means (NaN-mean populations are counted
    separately in ``n_nan_pops`` rather than silently ordered). Reports the
    top k+1 entries with means and cell counts plus absolute/relative
    top-1-vs-top-2 (``m1v2``) and top-k-vs-next (``m2v3``) margins and whether
    the cutoff is an exact tie. Changes nothing about selection or features.
    """
    act = celltype_activity(expr, labels)
    finite = {ct: v for ct, v in act.items()
              if isinstance(v["mean"], float) and np.isfinite(v["mean"])}
    ranked = sorted(finite.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    top = [{"population": ct, "mean": v["mean"], "n_cells": v["n_cells"]}
           for ct, v in ranked[: k + 1]]
    def _margin(a: float, b: float) -> dict:
        d = abs(a - b)
        return {"abs": float(d), "rel": float(d / max(abs(a), abs(b), 1e-12))}
    # the actual legacy selection, for fidelity checking (may include
    # NaN-mean populations, which finite ranking excludes)
    legacy = top_expressing_populations(expr, labels, k)
    finite_labels = [t["population"] for t in top[:k]]
    return {
        "k": k,
        "top": top,
        "m1v2": _margin(top[0]["mean"], top[1]["mean"]) if len(top) >= 2 else None,
        "m2v3": _margin(top[k - 1]["mean"], top[k]["mean"]) if len(top) >= k + 1 else None,
        "cutoff_tie": bool(len(top) >= k + 1 and top[k - 1]["mean"] == top[k]["mean"]),
        "n_pops": len(act),
        "n_nan_pops": len(act) - len(finite),
        "legacy_selection": [str(x) for x in legacy],
        "legacy_matches_finite_ranking": sorted(str(x) for x in legacy) == sorted(finite_labels),
    }


def overlap_best_match(
    ref_labels: Sequence[str], alt_labels: Sequence[str]
) -> dict:
    """Map each reference population to its best cell-overlap alternative.

    Pure cell counts, no ID-string comparison, no thresholds. Returns
    ``{ref_pop: {"alt_pop": str, "share": float, "n_cells": int}}`` where
    share is the fraction of the reference population's cells found in the
    best-matching alternative population.
    """
    ref = np.asarray([str(x) for x in ref_labels])
    alt = np.asarray([str(x) for x in alt_labels])
    assert len(ref) == len(alt), "labelings must cover the same cells"
    out: dict = {}
    for ct in sorted(set(ref)):
        mask = ref == ct
        sub = alt[mask]
        vals, counts = np.unique(sub, return_counts=True)
        best = int(np.argmax(counts))
        out[str(ct)] = {"alt_pop": str(vals[best]),
                        "share": float(counts[best] / mask.sum()),
                        "n_cells": int(mask.sum())}
    return out


def compare_selections(
    ref_labels: Sequence[str], alt_labels: Sequence[str] | None,
    ref_pops: Sequence[str], alt_pops: Sequence[str],
) -> dict:
    """P-B diagnostic flag: do overlap-matched selections differ?

    Maps every reference-selected population to its best-overlap alternative
    cluster and reports whether that matched set equals the alternatively
    selected set. Flag-only: never drops features or alters scores. When no
    alternative labeling is available, reports ``status: unavailable`` rather
    than stable.
    """
    if alt_labels is None or len(list(alt_labels)) == 0:
        return {"status": "unavailable (no alternative labeling)",
                "differ": None, "matches": {}}
    ref = [str(x) for x in ref_pops]
    alt = [str(x) for x in alt_pops]
    matches = overlap_best_match(ref_labels, alt_labels)
    detail = {ct: matches.get(ct, {"alt_pop": None, "share": 0.0,
                                   "n_cells": 0}) for ct in ref}
    matched_set = sorted({v["alt_pop"] for v in detail.values()
                          if v["alt_pop"] is not None})
    leaky = [ct for ct, v in detail.items() if v["share"] < 1.0]
    return {"status": "compared",
            "differ": matched_set != sorted(alt),
            "leaky_selected_pops": leaky,
            "ref_pops": ref, "alt_pops": alt,
            "overlap_matched_ref_pops": matched_set, "matches": detail}
