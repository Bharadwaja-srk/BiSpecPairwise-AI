"""Pairwise learning dataset (Zhang et al. 2024).

NOT binary classification: learns relative ordering. For drugs i, j with
stage_rank[i] < stage_rank[j] (i more advanced), the pair (Xi - Xj) gets
label 1; the mirrored (Xj - Xi) gets label 0. Ties are skipped (invalid
comparisons). Splits are grouped by pair_id to prevent leakage.
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd


def build_pairwise(
    features: np.ndarray,
    ranks: np.ndarray,
    pair_ids: list[str] | None = None,
    max_pairs: int | None = None,
    seed: int = 42,
) -> dict:
    X = np.asarray(features, dtype=float)
    r = np.asarray(ranks).ravel()
    assert len(X) == len(r), "features/ranks length mismatch"
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    combos = [(i, j) for i, j in itertools.combinations(idx, 2) if r[i] != r[j]]
    if max_pairs is not None and len(combos) > max_pairs:
        sel = rng.choice(len(combos), max_pairs, replace=False)
        combos = [combos[k] for k in sorted(sel)]
    Xd, y = [], []
    for i, j in combos:
        diff = np.nan_to_num(X[i] - X[j], nan=0.0)
        if r[i] < r[j]:  # i more advanced
            Xd += [diff, -diff]
            y += [1, 0]
        else:
            Xd += [-diff, diff]
            y += [0, 1]
    return {"X_diff": np.array(Xd), "y": np.array(y, dtype=int),
            "n_source": len(X), "n_pairs": len(combos)}


def group_split(
    pair_ids: list[str], n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold over pair_ids (leakage-aware). Falls back to KFold."""
    from sklearn.model_selection import GroupKFold, KFold

    ids = np.array(pair_ids)
    try:
        gkf = GroupKFold(n_splits=min(n_splits, len(set(ids))))
        return [(tr, te) for tr, te in gkf.split(ids, groups=ids)]
    except ValueError:
        kf = KFold(n_splits=min(n_splits, len(ids)), shuffle=True,
                   random_state=seed)
        return [(tr, te) for tr, te in kf.split(ids)]


def build_pairwise_df(
    df: pd.DataFrame, feature_cols: list[str],
    rank_col: str = "stage_rank", max_pairs: int | None = None,
    seed: int = 42,
) -> dict:
    X = df[feature_cols].to_numpy(dtype=float)
    r = df[rank_col].to_numpy()
    return build_pairwise(X, r, df.get("pair_id", None),
                          max_pairs=max_pairs, seed=seed)
