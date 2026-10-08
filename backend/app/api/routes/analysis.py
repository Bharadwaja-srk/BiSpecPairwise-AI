"""Analysis routes: POST /api/analyze."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import normalize_pair  # noqa: F401 (re-export for tests)
from app.schemas.analysis import AnalysisRequest
from app.services.ml_service import MLServiceError

router = APIRouter()


def _get_session_optional():
    # Optional DB: degrade gracefully when Postgres is down.
    async def _dep():
        try:
            from app.database.connection import db_manager

            async with db_manager.session() as session:
                yield session
        except Exception:
            yield None

    return _dep


@router.post("/analyze")
async def analyze(req: AnalysisRequest):
    try:
        t1, t2 = req.normalized_pair()
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    from app.services.analysis_service import run_analysis
    from app.database.connection import db_manager

    session = None
    try:
        # Try to enter a session; fall back to None (persistence skipped).
        cm = db_manager.session()
        session = await cm.__aenter__()
    except Exception:
        session = None

    try:
        result = await run_analysis(t1, t2, req.features, session=session)
        if session is not None:
            try:
                await session.commit()
            except Exception:
                pass
        return result
    except MLServiceError as e:
        raise HTTPException(e.status_code, str(e)) from e
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass
