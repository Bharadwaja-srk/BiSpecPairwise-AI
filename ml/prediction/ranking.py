"""Candidate scoring / sorting / rank assignment / top-K (ranking, not classification)."""
from __future__ import annotations

from .predict import predict_target_pair


def rank_candidates(candidates: list[dict], bundle: dict,
                    top_k: int | None = None) -> list[dict]:
    scored = [predict_target_pair(c["target1"], c["target2"],
                                  c["features"], bundle)
              for c in candidates]
    scored.sort(key=lambda r: r["prediction_score"], reverse=True)
    for i, r in enumerate(scored, start=1):
        r["rank"] = i
    return scored[:top_k] if top_k else scored


def top_k(candidates: list[dict], bundle: dict, k: int = 10) -> list[dict]:
    return rank_candidates(candidates, bundle, top_k=k)
