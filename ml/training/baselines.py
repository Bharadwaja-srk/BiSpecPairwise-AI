"""Baseline models with a common fit/predict interface (paper §Experimental settings).

LogReg (L2 C=10 ~ 1/0.1, 1000 iter), DT (min_leaf 5, depth 10),
RF (100 trees), GBDT (lr 0.1, 100 trees), MLP baseline (DNN-lite).
All metrics computed by caller — never hardcoded.
"""
from __future__ import annotations

import numpy as np


def _clean(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)


def make_baseline(name: str, seed: int = 42):
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    if name == "logreg":
        return LogisticRegression(C=10.0, max_iter=1000, random_state=seed)
    if name == "tree":
        return DecisionTreeClassifier(min_samples_leaf=5, max_depth=10,
                                      random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      n_jobs=-1)
    if name == "gbdt":
        return GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,
                                          random_state=seed)
    if name == "dnn":
        return MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
                             activation="relu", max_iter=500,
                             random_state=seed)
    raise ValueError(f"unknown baseline: {name}")


BASELINES = ["logreg", "tree", "rf", "gbdt", "dnn"]


def train_baseline(X, y, name: str, seed: int = 42):
    m = make_baseline(name, seed)
    m.fit(_clean(X), np.asarray(y).ravel())
    return m


def compare(X_tr, y_tr, X_te, y_te, seed: int = 42) -> dict:
    from ml.evaluation.metrics import roc_auc_safe

    out: dict = {}
    for name in BASELINES:
        try:
            m = train_baseline(X_tr, y_tr, name, seed)
            proba = m.predict_proba(_clean(X_te))[:, 1]
            out[name] = {"auc": roc_auc_safe(y_te, proba)}
        except Exception as e:  # e.g. tiny synthetic data
            out[name] = {"auc": float("nan"), "error": str(e)}
    return out
