"""Ranking routes: GET /api/ranking."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.ml_service import MLServiceError

router = APIRouter()

# Small dev candidate set (NOT real biology) used when caller supplies no pairs.
_DEV_CANDIDATES = [
    ("CD274", "CTLA4"),
    ("PDCD1", "CTLA4"),
    ("EGFR", "ERBB2"),
    ("CD47", "ERBB2"),
]


@router.get("/ranking")
async def get_ranking(
    top_k: int = Query(default=10, ge=1, le=100),
    candidate: Optional[list[str]] = Query(default=None, description="Repeatable 'T1:T2' pairs"),
):
    from app.services.ml_service import get_ml_service
    from app.services.target_store import resolve_features

    pairs: list[tuple[str, str]] = []
    for c in candidate or []:
        if ":" in c:
            a, b = c.split(":", 1)
            a, b = a.strip().upper(), b.strip().upper()
            if a and b and a != b:
                pairs.append((a, b))
    # Per docs/api/api.md: caller-supplied pairs are scored as given; with no
    # params a small clearly-marked dev set is used (never real biology).
    source = "caller-supplied"
    if not pairs:
        pairs = _DEV_CANDIDATES[:]
        source = "dev-candidates (NOT real biology)"

    candidates = []
    for a, b in pairs[:top_k]:
        feats, _, _ = resolve_features(a, b, None)
        candidates.append({"target1": a, "target2": b, "features": feats})

    ml = get_ml_service()
    try:
        ranked = await ml.rank_candidates(candidates, top_k=top_k)
    except MLServiceError as e:
        raise HTTPException(e.status_code, str(e)) from e
    return {
        "items": [
            {
                "rank": r.rank,
                "target1": r.target1,
                "target2": r.target2,
                "prediction_score": r.prediction_score,
                "model": r.model,
            }
            for r in ranked.results
        ],
        "total": len(ranked.results),
        "source": source,
    }
