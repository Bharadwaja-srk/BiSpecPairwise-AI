"""Analysis e2e: mock ML + mock RAG -> stable structured response (no network, no DB)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.asyncio()
async def test_run_analysis_e2e_mocked():
    import app.services.ml_service as mls
    import app.services.rag_service as rags
    from app.services.analysis_service import run_analysis

    mls._ml_service = mls.MockMLService()
    rags._rag_service = rags.MockRAGService()
    try:
        out = await run_analysis("CD274", "CTLA4", session=None)
    finally:
        mls._ml_service = None
        rags._rag_service = None
    assert out["target1"] == "CD274" and out["target2"] == "CTLA4"
    pred = out["prediction"]
    assert pred["prediction_score"] == pytest.approx(0.93)
    assert len(pred["features"]) == 11  # full FEATURE_ORDER echo
    assert pred["feature_importance"][0]["rank"] == 1
    assert pred["feature_status"] == "incomplete"  # NaN-fill must be flagged
    assert len(pred["missing_features"]) == 11
    assert out["report"]["confidence"] in ("high", "medium", "low")
    assert out["persistence"] == "skipped"


@pytest.mark.asyncio()
async def test_run_analysis_reports_rag_failure_gracefully():
    import app.services.ml_service as mls
    import app.services.rag_service as rags
    from app.services.analysis_service import run_analysis

    class _BrokenRAG(rags.MockRAGService):
        async def generate_report(self, payload):
            raise ConnectionError("rag down")

    mls._ml_service = mls.MockMLService()
    rags._rag_service = _BrokenRAG()
    try:
        out = await run_analysis("EGFR", "ERBB2", session=None)
    finally:
        mls._ml_service = None
        rags._rag_service = None
    assert out["report"] is None
    assert out["rag_error"] is not None
    assert out["prediction"]["prediction_score"] == pytest.approx(0.57)


def test_target_store_nan_fill_flags_missing():
    from app.services.target_store import FEATURE_ORDER, resolve_features

    feats, missing, source = resolve_features("A", "B", None)
    assert set(feats.keys()) == set(FEATURE_ORDER)
    assert missing == FEATURE_ORDER and source == "nan-fill"
    feats2, missing2, src2 = resolve_features("A", "B", {"safety_score": 0.1})
    assert src2 == "caller" and "safety_score" not in missing2
    assert len(missing2) == len(FEATURE_ORDER) - 1
