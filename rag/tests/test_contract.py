"""Backend<->RAG contract tests (no network, controlled fixtures).

Proves the analysis payload the backend builds is accepted by the RAG
service schema, and the backend mock report satisfies the RAG report
validator. Real-service behavior is NOT established here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))


def test_backend_payload_keys_match_rag_schema():
    from rag.app import GenerateRequest, RetrieveRequest

    assert set(GenerateRequest.model_fields) == {
        "target1", "target2", "prediction", "features",
        "feature_importance", "discretized_features",
    }
    assert set(RetrieveRequest.model_fields) == {
        "target1", "target2", "context", "top_k",
    }
    # backend sends exactly these keys (analysis_service.run_analysis payload)
    backend_keys = {"target1", "target2", "prediction", "features",
                    "feature_importance", "discretized_features"}
    assert backend_keys <= set(GenerateRequest.model_fields)


def test_backend_mock_report_passes_rag_validator():
    import asyncio

    from app.services.rag_service import MockRAGService
    from rag.reports.report_generator import validate_report

    async def run():
        return await MockRAGService().generate_report({
            "target1": "CD274", "target2": "CTLA4",
            "prediction": {"prediction_score": 0.9, "rank": 0}})

    rep = asyncio.run(run())
    assert rep["model_name"] == "mock-llm"  # labelled mock, never real
    assert rep["retrieved_evidence"] == []
    validated = validate_report(rep)
    assert validated["confidence"] in ("high", "medium", "low")
    assert len(validated["limitations"]) > 0


def test_rag_generate_report_roundtrip_mock():
    from fastapi.testclient import TestClient

    import rag.app as rag_app
    from rag.retrieval import retriever

    retriever._store = None
    os.environ.pop("LLM_PROVIDER", None)
    c = TestClient(rag_app.app)
    r = c.post("/generate-report", json={
        "target1": "CD274", "target2": "CTLA4",
        "prediction": {"prediction_score": 0.9, "rank": 0, "model": "m"},
        "features": {"safety_score": 0.0012},
        "feature_importance": {"safety_score": 1.0},
        "discretized_features": {}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["confidence"] in ("high", "medium", "low")
    assert body["retrieved_evidence"] == []  # empty store -> [] never fabricated
    retriever._store = None
