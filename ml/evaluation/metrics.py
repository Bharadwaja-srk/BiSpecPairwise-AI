"""Metrics: ROC-AUC (primary), accuracy/precision/recall/F1, confusion matrix,
ranking (NDCG, MRR). All computed — never hardcoded."""
from __future__ import annotations

import numpy as np


def roc_auc_safe(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).ravel()
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, np.asarray(y_score).ravel()))
    except Exception:
        return float("nan")


def classification_report_dict(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def dcg(relevances: list[float], k: int | None = None) -> float:
    rel = np.asarray(relevances[:k] if k else relevances, dtype=float)
    return float(np.sum((2 ** rel - 1) / np.log2(np.arange(2, len(rel) + 2))))


def ndcg(ranked_relevance: list[float], k: int | None = None) -> float:
    ideal = sorted(ranked_relevance, reverse=True)
    denom = dcg(ideal, k)
    return dcg(ranked_relevance, k) / denom if denom else 0.0


def mrr(first_relevant_ranks: list[int]) -> float:
    rr = [1.0 / r for r in first_relevant_ranks if r and r > 0]
    return float(np.mean(rr)) if rr else 0.0
