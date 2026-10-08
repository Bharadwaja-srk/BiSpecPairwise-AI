"""
RAG Service HTTP client for BSPAI Backend.

Talks to the standalone RAG service (rag/app.py) over HTTP:
  POST /generate-report {target1,target2,prediction,features,feature_importance,discretized_features}
  POST /retrieve        {target1,target2,top_k}
  GET  /health
RAG failure must degrade gracefully (report=null + rag_error), never 500 the analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        response = await client.post("/generate-report", json=payload)
        response.raise_for_status()
        return response.json()

    async def retrieve(self, target1: str, target2: str, top_k: int = 5) -> list[dict[str, Any]]:
        client = await self._get_client()
        response = await client.post("/retrieve", json={"target1": target1, "target2": target2, "top_k": top_k})
        response.raise_for_status()
        data = response.json()
        return data.get("results", data.get("evidence", []))

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False


class MockRAGService(RAGService):
    """Deterministic mock — clearly labelled mock content, no fabricated citations."""

    def __init__(self):
        self.base_url = "mock"
        self.timeout = 0

    async def generate_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        t1, t2 = payload.get("target1", "?"), payload.get("target2", "?")
        pred = payload.get("prediction", {})
        return {
            "summary": f"Mock interpretation of {t1}+{t2} (score {pred.get('prediction_score')}).",
            "prediction": {"score": pred.get("prediction_score"), "rank": pred.get("rank", 0)},
            "biological_interpretation": "Mock: feature-driven interpretation placeholder — not real biology.",
            "mechanistic_rationale": "Mock: mechanistic rationale placeholder.",
            "safety_considerations": "Mock: safety must be assessed from real tumor/normal expression; no claim made.",
            "clinical_evidence": "Mock: no retrieved clinical trials; no efficacy claimed.",
            "retrieved_evidence": [],
            "limitations": ["Mock RAG — no real biomedical evidence retrieved.", "ML score is not a clinical probability."],
            "confidence": "low",
            "model_name": "mock-llm",
        }

    async def retrieve(self, target1: str, target2: str, top_k: int = 5) -> list[dict[str, Any]]:
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        settings = get_settings()
        if settings.ENABLE_MOCK_RAG or settings.ENABLE_MOCK_LLM:
            logger.info("Using mock RAG service")
            _rag_service = MockRAGService()
        else:
            _rag_service = RAGService(settings.RAG_SERVICE_URL, settings.RAG_SERVICE_TIMEOUT)
    return _rag_service


async def close_rag_service() -> None:
    global _rag_service
    if _rag_service:
        await _rag_service.close()
        _rag_service = None
