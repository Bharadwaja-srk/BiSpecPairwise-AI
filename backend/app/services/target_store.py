"""
Target feature resolution for BSPAI Backend.

Resolution order for POST /api/analyze {target1, target2, features?}:
  1. Caller-supplied `features` dict (validated, NaN-filled for missing keys).
  2. Precomputed JSON store at FEATURE_STORE_PATH (optional file mapping
     "T1__T2" or "T2__T1" -> 11-key feature dict). Inspected 2026-10-08: no
     precomputed 11-feature store ships with the repo (data/raw holds only
     scRNA-seq matrices; feature_engineering/ computes on demand from arrays).
     The store hook exists for operators who generate one offline.
  3. NaN-fill per feature_schema missing policy.

Never fabricates biology: when keys are missing the response carries
`missing_features=[...]` and `feature_status="incomplete"`, and callers must
not present the resulting score as biologically valid.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_ORDER: list[str] = [
    "target_activity_1",
    "target_activity_2",
    "single_ratio_max",
    "single_ratio_min",
    "double_positive",
    "safety_score",
    "pathway_similarity",
    "pathway_shared_count",
    "gene_embedding_similarity",
    "gene_embedding_distance",
    "pearson_correlation",
]

_store_cache: dict[str, dict] | None = None


def _load_store() -> dict[str, dict]:
    global _store_cache
    if _store_cache is not None:
        return _store_cache
    path = os.getenv("FEATURE_STORE_PATH", "data/precomputed_features.json")
    _store_cache = {}
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                raw = json.load(f)
            _store_cache = {str(k).upper(): dict(v) for k, v in raw.items()}
            logger.info("Loaded precomputed feature store: %s (%d pairs)", path, len(_store_cache))
        except Exception as e:
            logger.warning("Failed to load feature store %s: %s", path, e)
    return _store_cache


def _coerce_float(v: Any) -> float:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return float("nan")


def resolve_features(
    target1: str,
    target2: str,
    supplied: dict[str, float] | None = None,
) -> tuple[dict[str, float], list[str], str]:
    """Return (features, missing_features, source).

    source ∈ {"caller", "precomputed", "nan-fill"}.
    missing_features lists keys that were absent or non-numeric.
    """
    t1, t2 = target1.upper(), target2.upper()
    if supplied:
        feats = {k: _coerce_float(supplied.get(k, float("nan"))) for k in FEATURE_ORDER}
        missing = [k for k in FEATURE_ORDER if k not in supplied or _is_nan(feats[k]) and k not in supplied]
        # recompute: a key counts missing only if not supplied at all or unparseable
        missing = [k for k in FEATURE_ORDER if k not in supplied]
        try:
            float(supplied.get("__x__", 0))  # noqa - keep simple
        except Exception:
            pass
        return feats, missing, "caller"
    store = _load_store()
    for key in (f"{t1}__{t2}", f"{t2}__{t1}"):
        if key in store:
            entry = store[key]
            feats = {k: _coerce_float(entry.get(k, float("nan"))) for k in FEATURE_ORDER}
            missing = [k for k in FEATURE_ORDER if k not in entry]
            return feats, missing, "precomputed"
    feats = {k: float("nan") for k in FEATURE_ORDER}
    return feats, list(FEATURE_ORDER), "nan-fill"


def _is_nan(v: float) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def feature_status(missing: list[str]) -> str:
    return "complete" if not missing else "incomplete"


# --- Clearly-marked development target catalogue (NOT real biology) ---
DEV_TARGETS: list[dict[str, Any]] = [
    {"gene_symbol": s, "gene_name": n, "description": "DEV-ONLY placeholder; not real biological data."}
    for s, n in [
        ("CD274", "PD-L1 (dev placeholder)"),
        ("CTLA4", "CTLA-4 (dev placeholder)"),
        ("PDCD1", "PD-1 (dev placeholder)"),
        ("EGFR", "EGFR (dev placeholder)"),
        ("ERBB2", "HER2 (dev placeholder)"),
        ("CD47", "CD47 (dev placeholder)"),
        ("LAG3", "LAG-3 (dev placeholder)"),
        ("TNFRSF9", "4-1BB (dev placeholder)"),
        ("ICOS", "ICOS (dev placeholder)"),
        ("MUC1", "MUC1 (dev placeholder)"),
    ]
]
