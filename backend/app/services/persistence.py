"""Best-effort DB persistence for analyses (never fails the request)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID


async def persist_analysis(session, t1: str, t2: str, ml_resp, report, evidence) -> UUID:
    from app.models.database import Analysis, AnalysisStatus, Evidence, Prediction, Report, Target, TargetPair

    now = datetime.now(timezone.utc)

    async def _get_or_create_target(symbol: str) -> Target:
        from sqlalchemy import select

        r = await session.execute(select(Target).where(Target.gene_symbol == symbol))
        obj = r.scalar_one_or_none()
        if obj is None:
            obj = Target(gene_symbol=symbol)
            session.add(obj)
            await session.flush()
        return obj

    o1 = await _get_or_create_target(t1)
    o2 = await _get_or_create_target(t2)

    from sqlalchemy import select

    r = await session.execute(
        select(TargetPair).where(
            ((TargetPair.target1_id == o1.id) & (TargetPair.target2_id == o2.id))
            | ((TargetPair.target1_id == o2.id) & (TargetPair.target2_id == o1.id))
        )
    )
    pair = r.scalar_one_or_none()
    if pair is None:
        pair = TargetPair(target1_id=o1.id, target2_id=o2.id)
        session.add(pair)
        await session.flush()

    analysis = Analysis(
        target_pair_id=pair.id,
        status=AnalysisStatus.COMPLETED,
        request_data={"target1": t1, "target2": t2},
        started_at=now,
        completed_at=now,
    )
    session.add(analysis)
    await session.flush()

    session.add(
        Prediction(
            analysis_id=analysis.id,
            prediction_score=float(ml_resp.prediction_score),
            rank=int(ml_resp.rank or 0) or None,
            features=dict(ml_resp.features),
            feature_importance=dict(ml_resp.feature_importance),
            model_version=str(ml_resp.model),
        )
    )
    if isinstance(report, dict) and report.get("summary"):
        rep = Report(
            analysis_id=analysis.id,
            summary=str(report.get("summary", "")),
            biological_interpretation=str(report.get("biological_interpretation", "")),
            mechanistic_rationale=str(report.get("mechanistic_rationale", "")),
            safety_considerations=str(report.get("safety_considerations", "")),
            clinical_evidence=str(report.get("clinical_evidence", "")),
            limitations=list(report.get("limitations", []) or []),
            confidence=str(report.get("confidence", "low")),
            retrieved_evidence=list(evidence or []),
            model_name=str(report.get("model_name", "")) or None,
        )
        session.add(rep)
        await session.flush()
        for ev in evidence or []:
            if not isinstance(ev, dict):
                continue
            session.add(
                Evidence(
                    report_id=rep.id,
                    source_title=str(ev.get("source_title", ev.get("title", "unknown"))),
                    source_type=str(ev.get("source_type", "unknown")),
                    source_url=ev.get("source_url") or ev.get("url"),
                    source_id=ev.get("source_id"),
                    snippet=str(ev.get("snippet", ev.get("text", "")))[:4000],
                    relevance_score=float(ev.get("relevance_score", ev.get("score", 0.0)) or 0.0),
                    extra_metadata={k: v for k, v in ev.items() if k not in ("snippet", "text")},
                )
            )
    await session.commit()
    return analysis.id
