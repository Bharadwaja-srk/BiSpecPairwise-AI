"""Leakage-aware 5-fold cross-validation (grouped by pair_id when given)."""
from __future__ import annotations

import numpy as np

from .evaluate import evaluate_proba


def cross_validate(
    X, y, make_fn, groups: list[str] | None = None,
    n_splits: int = 5, seed: int = 42,
) -> dict:
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
    y = np.asarray(y).ravel()
    if groups is not None:
        splitter = GroupKFold(
            n_splits=min(n_splits, len(set(groups)))).split(X, groups=groups)
    else:
        try:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed).split(X, y)
        except ValueError:
            from sklearn.model_selection import KFold

            splitter = KFold(n_splits=min(n_splits, len(X)), shuffle=True,
                             random_state=seed).split(X)
    aucs = []
    for tr, te in splitter:
        m = make_fn()
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        aucs.append(evaluate_proba(y[te], proba)["auc"])
    aucs = np.array(aucs, dtype=float)
    return {"auc_per_fold": aucs.tolist(),
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs))}
