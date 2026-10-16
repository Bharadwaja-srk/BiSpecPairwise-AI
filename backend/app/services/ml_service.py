"""
ML Service HTTP client for BSPAI Backend.

Thin adapter over the existing ML service (ml/service/app.py, owned by
OpenCode — READ-ONLY). Backend must depend on the ML API contract, never on
ML implementation details.

Actual ML contract (see docs/api/ml-interface.md):
  GET  /health   -> {"status": "ok", "model_loaded": bool}
  POST /predict  -> req {"target1","target2","features":{11 FEATURE_ORDER keys}}
                     resp {"target1","target2","prediction_score","rank",
                           "features":echo,"feature_importance":{name:score},"model":kind}
  POST /rank     -> req {"candidates":[{...predict req}],"top_k":int|null}
                     resp {"results":[{...predict resp + rank 1..N}]}
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Canonical 11-feature order (mirrors feature_engineering/feature_schema.py).
# Imported lazily to avoid hard backend->ML dependency at module import time.
FEATURE_ORDER: list[str] = [
    "target_activity_1",
    "target_activity_2",
    "single_ratio_max",
    "single_ratio_min",
    "double_positive",
    "safety_score",
    "pathway_similarity",
    "pathway_shared_count",
    "gene_embedding_similarity",
    "gene_embedding_distance",
    "pearson_correlation",
]


class MLHealth(BaseModel):
    status: str = "ok"
    model_loaded: bool = False


class MLPredictionResponse(BaseModel):
    """Single-pair prediction exactly as the ML service returns it."""

    target1: str
    target2: str
    prediction_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = 0  # ML returns 0 for single /predict; real ranks come from /rank
    features: dict[str, float] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    model: str = "unknown"

    model_config = {"extra": "allow"}

    def sorted_importance(self) -> list[dict[str, Any]]:
        """Importance as a sorted list for frontend convenience."""
        items = sorted(self.feature_importance.items(), key=lambda kv: kv[1], reverse=True)
        return [{"feature": k, "importance": v, "rank": i} for i, (k, v) in enumerate(items, start=1)]


class MLRankResponse(BaseModel):
    results: list[MLPredictionResponse] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class MLServiceError(Exception):
    """Controlled ML-service error (no raw provider tracebacks)."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _require_finite_features(features: dict, op: str) -> None:
    """Reject missing/non-finite feature data before any ML HTTP call.

    Never zero-fills, fabricates, or falls back to mock predictions: callers
    get a structured 503 so ranking/analysis visibly waits for valid data.
    """
    import math

    bad = [k for k in FEATURE_ORDER if not isinstance(features.get(k), (int, float)) or not math.isfinite(features[k])]
    if bad:
        raise MLServiceError(
            f"Valid feature data unavailable ({op}): non-finite or missing values for "
            f"{bad}. Ranking/analysis cannot proceed until finite 11-feature data are "
            f"available (feature store or caller-supplied finite features).",
            503,
        )


class MLService:
    """HTTP client for the existing ML service."""

    def __init__(self, base_url: str, timeout: float = 30.0):
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

    def _handle_response(self, response: httpx.Response, op: str) -> dict:
        if response.status_code == 503:
            raise MLServiceError(f"ML model unavailable ({op}): train first / MODEL_PATH missing", 503)
        if response.status_code >= 400:
            detail = response.text[:500]
            try:
                detail = str(response.json())[:500]
            except Exception:
                pass
            raise MLServiceError(f"ML service error ({op}): {detail}", 502)
        return response.json()

    async def predict(
        self,
        target1: str,
        target2: str,
        features: dict[str, float],
    ) -> MLPredictionResponse:
        """POST /predict — features must contain the 11 FEATURE_ORDER keys."""
        _require_finite_features(features, "predict")
        client = await self._get_client()
        try:
            response = await client.post(
                "/predict",
                json={"target1": target1.upper(), "target2": target2.upper(), "features": features},
            )
        except httpx.ConnectError as e:
            raise MLServiceError(f"ML service unreachable at {self.base_url}: {e}", 503) from e
        except httpx.TimeoutException as e:
            raise MLServiceError(f"ML service timeout after {self.timeout}s", 504) from e
        data = self._handle_response(response, "predict")
        return MLPredictionResponse(**data)

    async def rank_candidates(
        self,
        candidates: list[dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> MLRankResponse:
        """POST /rank — each candidate: {target1, target2, features}."""
        for c in candidates:
            _require_finite_features(c.get("features", {}), f"rank {c.get('target1')}-{c.get('target2')}")
        client = await self._get_client()
        payload: dict[str, Any] = {"candidates": candidates}
        if top_k is not None:
            payload["top_k"] = top_k
        try:
            response = await client.post("/rank", json=payload)
        except httpx.ConnectError as e:
            raise MLServiceError(f"ML service unreachable at {self.base_url}: {e}", 503) from e
        except httpx.TimeoutException as e:
            raise MLServiceError(f"ML service timeout after {self.timeout}s", 504) from e
        data = self._handle_response(response, "rank")
        return MLRankResponse(**data)

    async def health_check(self) -> MLHealth:
        """GET /health — never raises; reports reachable + model_loaded."""
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            if response.status_code != 200:
                return MLHealth(status="unreachable", model_loaded=False)
            return MLHealth(**response.json())
        except Exception as e:
            logger.warning("ML health check failed: %s", e)
            return MLHealth(status="unreachable", model_loaded=False)


class MockMLService(MLService):
    """Deterministic mock using real FEATURE_ORDER names. Dev/test only.

    Scores for a few paper pairs are fixed; all other pairs get 0.45.
    Feature values are synthetic and flagged via model='mock-*' — never
    present these as real biological measurements.
    """

    _KNOWN_SCORES = {
        frozenset(["CD274", "CTLA4"]): 0.93,
        frozenset(["PDCD1", "CTLA4"]): 0.79,
        frozenset(["EGFR", "ERBB2"]): 0.57,
        frozenset(["CD47", "ERBB2"]): 0.69,
    }

    def __init__(self):
        self.base_url = "mock"
        self.timeout = 0

    @staticmethod
    def _synthetic_features() -> dict[str, float]:
        return {
            "target_activity_1": 1.2,
            "target_activity_2": 0.9,
            "single_ratio_max": 0.45,
            "single_ratio_min": 0.24,
            "double_positive": 0.0071,
            "safety_score": 0.0012,
            "pathway_similarity": 0.35,
            "pathway_shared_count": 3.0,
            "gene_embedding_similarity": 0.55,
            "gene_embedding_distance": 3.85,
            "pearson_correlation": 0.012,
        }

    @staticmethod
    def _synthetic_importance() -> dict[str, float]:
        return {
            "gene_embedding_similarity": 0.22,
            "safety_score": 0.16,
            "pearson_correlation": 0.13,
            "single_ratio_max": 0.11,
            "double_positive": 0.10,
            "pathway_similarity": 0.08,
            "target_activity_1": 0.06,
            "target_activity_2": 0.05,
            "single_ratio_min": 0.04,
            "pathway_shared_count": 0.03,
            "gene_embedding_distance": 0.02,
        }

    async def predict(self, target1: str, target2: str, features: dict[str, float]) -> MLPredictionResponse:
        score = self._KNOWN_SCORES.get(frozenset([target1.upper(), target2.upper()]), 0.45)
        echo = {k: float(features.get(k, float("nan"))) for k in FEATURE_ORDER}
        return MLPredictionResponse(
            target1=target1.upper(),
            target2=target2.upper(),
            prediction_score=score,
            rank=0,
            features=echo,
            feature_importance=self._synthetic_importance(),
            model="mock-xgboost_pairwise",
        )

    async def rank_candidates(self, candidates: list[dict[str, Any]], top_k: Optional[int] = None) -> MLRankResponse:
        scored = []
        for c in candidates:
            r = await self.predict(c["target1"], c["target2"], c.get("features", {}))
            scored.append(r)
        scored.sort(key=lambda r: r.prediction_score, reverse=True)
        for i, r in enumerate(scored, start=1):
            r.rank = i
        return MLRankResponse(results=scored[:top_k] if top_k else scored)

    async def health_check(self) -> MLHealth:
        return MLHealth(status="ok", model_loaded=True)

    async def close(self) -> None:
        pass


_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Get or create ML service instance (mock when ENABLE_MOCK_ML=true)."""
    global _ml_service
    if _ml_service is None:
        settings = get_settings()
        if settings.ENABLE_MOCK_ML:
            logger.info("Using mock ML service")
            _ml_service = MockMLService()
        else:
            _ml_service = MLService(settings.ML_SERVICE_URL, settings.ML_SERVICE_TIMEOUT)
    return _ml_service


async def close_ml_service() -> None:
    global _ml_service
    if _ml_service:
        await _ml_service.close()
        _ml_service = None
