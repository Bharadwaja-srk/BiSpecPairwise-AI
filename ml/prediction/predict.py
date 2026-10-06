"""Clean prediction interface (no hardcoded scores).

predict_target_pair(target1, target2, features) ->
 {target1, target2, prediction_score, rank, features, feature_importance, model}
Supports bundles: {"model","features","kind"[,"reference"]}.
Pairwise bundles score (x - reference); pointwise score x directly.
"""
from __future__ import annotations

import numpy as np

from ml.explainability.feature_importance import importance_dict


def _vector(features: dict | list, order: list[str]) -> np.ndarray:
    if isinstance(features, dict):
        return np.array([float(features.get(k, float("nan"))) for k in order],
                        dtype=float)
    return np.asarray(features, dtype=float).ravel()


def load_bundle(path: str) -> dict:
    from ml.data.datasets import load_artifact

    b = load_artifact(path)
    return b if isinstance(b, dict) and "model" in b else {"model": b,
            "features": None, "kind": "unknown"}


def predict_target_pair(target1: str, target2: str, features,
                        bundle: dict) -> dict:
    model = bundle["model"]
    order: list[str] = bundle.get("features") or (
        list(features.keys()) if isinstance(features, dict) else [])
    x = np.nan_to_num(_vector(features, order), nan=0.0).reshape(1, -1)
    if bundle.get("kind") == "xgboost_pairwise" and "reference" in bundle:
        ref = np.asarray(bundle["reference"], dtype=float).reshape(1, -1)
        x = x - ref
    score = float(model.predict_proba(x)[0, 1])
    feats = (dict(features) if isinstance(features, dict)
             else dict(zip(order, x.ravel().tolist())))
    return {
        "target1": target1, "target2": target2,
        "prediction_score": score, "rank": 0,
        "features": feats,
        "feature_importance": importance_dict(model, order),
        "model": bundle.get("kind", "unknown"),
    }
