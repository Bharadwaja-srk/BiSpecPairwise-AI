"""Tabular helpers: feature matrix assembly + persistence helpers."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd


def df_to_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].to_numpy(dtype=float)


def save_artifact(obj, path: str) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    return joblib.load(path)
