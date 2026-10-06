"""Unified evaluation: proba -> AUC + classification report."""
from __future__ import annotations

import numpy as np

from .metrics import classification_report_dict, roc_auc_safe


def evaluate_proba(y_true, y_score, threshold: float = 0.5) -> dict:
    y_score = np.asarray(y_score, dtype=float).ravel()
    y_pred = (y_score >= threshold).astype(int)
    out = {"auc": roc_auc_safe(y_true, y_score), "threshold": threshold}
    out.update(classification_report_dict(y_true, y_pred))
    return out
