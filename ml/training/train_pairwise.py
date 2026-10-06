"""XGBoost + pairwise learning (PRIMARY model, Zhang et al. 2024).

Trains an XGB classifier on difference vectors (Xi - Xj) so inference can
score single candidates directly: score(x) = P(x > reference). Compatible
with the same prediction/ranking pipeline as pointwise XGB.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ml.config import RANDOM_SEED, seed_everything
from ml.data.datasets import save_artifact
from ml.data.pairwise import build_pairwise_df
from ml.training.train_xgboost import DEFAULT_PARAMS, make_model


def train_pairwise_df(df, feature_cols: list[str],
                      params: dict | None = None,
                      max_pairs: int | None = None,
                      seed: int = RANDOM_SEED) -> dict:
    seed_everything(seed)
    built = build_pairwise_df(df, feature_cols, max_pairs=max_pairs,
                              seed=seed)
    Xd = np.nan_to_num(built["X_diff"], nan=0.0)
    model = make_model({**(params or {}), "random_state": seed})
    model.fit(Xd, built["y"])
    # reference = mean feature vector of most-advanced class (for scoring)
    ranks = df["stage_rank"].to_numpy()
    best = df.loc[ranks == ranks.min(), feature_cols].to_numpy(dtype=float)
    ref = np.nan_to_num(np.nanmean(best, axis=0), nan=0.0)
    return {"model": model, "features": feature_cols,
            "kind": "xgboost_pairwise", "reference": ref,
            "n_pairs": built["n_pairs"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", default=os.getenv("MODEL_PATH",
                                               "models/xgb_pairwise_v1.pkl"))
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()
    import pandas as pd

    df = pd.read_csv(args.csv)
    bundle = train_pairwise_df(df, args.features, max_pairs=args.max_pairs,
                               seed=args.seed)
    save_artifact(bundle, args.out)
    print(f"saved -> {args.out} (pairs={bundle['n_pairs']})")


if __name__ == "__main__":
    main()
