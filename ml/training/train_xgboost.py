"""XGBoost pointwise training (paper defaults: lr=0.1, 100 trees, depth=10).

Training logic lives here — never inside FastAPI. Reproducible via seed.
"""
from __future__ import annotations

import argparse
import os

from ml.config import RANDOM_SEED, seed_everything
from ml.data.datasets import load_artifact, save_artifact


DEFAULT_PARAMS = {
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
}


def make_model(params: dict | None = None):
    from xgboost import XGBClassifier

    p = {**DEFAULT_PARAMS, **(params or {})}
    return XGBClassifier(**p)


def train(df, feature_cols: list[str], label_col: str = "label",
          params: dict | None = None, seed: int = RANDOM_SEED):
    import numpy as np

    seed_everything(seed)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    model = make_model({**(params or {}), "random_state": seed})
    model.fit(X, y)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--label", default="label")
    ap.add_argument("--out", default=os.getenv("MODEL_PATH",
                                               "models/xgb_pointwise.pkl"))
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()
    import pandas as pd

    df = pd.read_csv(args.csv)
    model = train(df, args.features, args.label, seed=args.seed)
    save_artifact({"model": model, "features": args.features,
                   "kind": "xgboost_pointwise"}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
