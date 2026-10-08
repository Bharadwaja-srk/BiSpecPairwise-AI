"""Report routes: GET /api/reports/{analysis_id}."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/reports/{analysis_id}")
async def get_report(analysis_id: UUID):
    from sqlalchemy import select

    from app.database.connection import db_manager
    from app.models.database import Analysis, Prediction, Report

    try:
        async with db_manager.session() as session:
            a = (await session.execute(select(Analysis).where(Analysis.id == analysis_id))).scalar_one_or_none()
            if not a:
                raise HTTPException(404, "analysis not found")
            p = (await session.execute(select(Prediction).where(Prediction.analysis_id == a.id))).scalar_one_or_none()
            r = (await session.execute(select(Report).where(Report.analysis_id == a.id))).scalar_one_or_none()
            return {
                "analysis_id": str(a.id),
                "status": a.status.value if hasattr(a.status, "value") else str(a.status),
                "prediction": (
                    {"score": p.prediction_score, "rank": p.rank, "model": p.model_version} if p else None
                ),
                "report": (
                    {
                        "summary": r.summary,
                        "biological_interpretation": r.biological_interpretation,
                        "mechanistic_rationale": r.mechanistic_rationale,
                        "safety_considerations": r.safety_considerations,
                        "clinical_evidence": r.clinical_evidence,
                        "retrieved_evidence": r.retrieved_evidence,
                        "limitations": r.limitations,
                        "confidence": r.confidence,
                    }
                    if r
                    else None
                ),
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"database unavailable: {e}") from e
