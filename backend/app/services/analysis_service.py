"""
Analysis orchestration: Backend -> ML -> RAG -> structured response.

Route handlers stay thin; all logic lives here.
DB persistence is best-effort (degrade gracefully when Postgres is down).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _discretize_local(features: dict[str, float]) -> dict[str, str]:
    """Fallback discretization using ML Table-5 bins without importing ml internals."""
    try:
        from ml.explainability.feature_importance import discretize

        return {k: discretize(k, float(v)) for k, v in features.items()}
    except Exception:
        return {k: str(v) for k, v in features.items()}


def _json_safe(obj):
    """Recursively convert NaN/Inf floats to None so Starlette JSONResponse never crashes.

    Missing features are still explicitly listed in `missing_features` + `feature_status`.
    """
    import math

    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


async def run_analysis(
    target1: str,
    target2: str,
    supplied_features: Optional[dict[str, float]] = None,
    session=None,
) -> dict[str, Any]:
    from app.api.dependencies import normalize_pair
    from app.services.ml_service import get_ml_service
    from app.services.target_store import feature_status, resolve_features

    t1, t2 = normalize_pair(target1, target2)
    features, missing, source = resolve_features(t1, t2, supplied_features)

    ml = get_ml_service()
    ml_resp = await ml.predict(t1, t2, features)

    status = feature_status(missing)
    # If caller gave partial features but ML echoed NaNs, extend missing list
    import math

    nan_keys = [k for k, v in ml_resp.features.items() if isinstance(v, float) and math.isnan(v)]
    all_missing = sorted(set(missing) | set(nan_keys))
    status = feature_status(all_missing)

    discretized = _discretize_local(ml_resp.features)
    importance = ml_resp.sorted_importance()

    # RAG (degrade gracefully)
    report: Optional[dict[str, Any]] = None
    evidence: list[dict[str, Any]] = []
    rag_error: Optional[str] = None
    try:
        from app.services.rag_service import get_rag_service

        rag = get_rag_service()
        payload = {
            "target1": t1,
            "target2": t2,
            "prediction": {"prediction_score": ml_resp.prediction_score, "rank": ml_resp.rank, "model": ml_resp.model},
            "features": ml_resp.features,
            "feature_importance": ml_resp.feature_importance,
            "discretized_features": discretized,
        }
        report = await rag.generate_report(payload)
        evidence = report.get("retrieved_evidence", []) if isinstance(report, dict) else []
    except Exception as e:
        logger.warning("RAG report failed for %s-%s: %s", t1, t2, e)
        rag_error = f"RAG unavailable: {type(e).__name__}"

    now = datetime.now(timezone.utc)
    response: dict[str, Any] = {
        "target1": t1,
        "target2": t2,
        "status": "completed",
        "prediction": {
            "prediction_score": ml_resp.prediction_score,
            "rank": ml_resp.rank,
            "features": ml_resp.features,
            "discretized_features": discretized,
            "missing_features": all_missing,
            "feature_status": status,
            "feature_source": source,
            "feature_importance": importance,
            "model": ml_resp.model,
        },
        "retrieved_evidence": evidence,
        "report": report,
        "rag_error": rag_error,
        "persistence": "skipped",
        "created_at": now.isoformat(),
        "completed_at": now.isoformat(),
    }

    # Best-effort DB persistence
    if session is not None:
        try:
            from app.services.persistence import persist_analysis

            analysis_id = await persist_analysis(session, t1, t2, ml_resp, report, evidence)
            response["analysis_id"] = str(analysis_id)
            response["persistence"] = "persisted"
        except Exception as e:
            logger.warning("DB persistence skipped: %s", e)
            response["persistence"] = "skipped"

    return _json_safe(response)
