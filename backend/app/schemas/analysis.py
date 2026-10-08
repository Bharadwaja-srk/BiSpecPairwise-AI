"""
Pydantic schemas for analysis API requests and responses.

Features use the canonical 11-name FEATURE_ORDER contract
(see feature_engineering/feature_schema.py and docs/api/ml-interface.md).
The backend never invents feature names — it echoes the ML response.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AnalysisRequest(BaseModel):
    """Request to analyze a target pair."""

    target1: str = Field(..., min_length=1, max_length=50, description="First target gene symbol (e.g., CD274)")
    target2: str = Field(..., min_length=1, max_length=50, description="Second target gene symbol (e.g., CTLA4)")
    features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional caller-supplied 11-key feature dict. When omitted, the backend resolves via target_store (precomputed or NaN-fill with missing_features reported).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"target1": "CD274", "target2": "CTLA4"}
        }
    )

    def normalized_pair(self) -> tuple[str, str]:
        t1, t2 = self.target1.strip().upper(), self.target2.strip().upper()
        if not t1 or not t2:
            raise ValueError("target symbols must be non-empty")
        if t1 == t2:
            raise ValueError("target1 and target2 must differ")
        return t1, t2


class FeatureImportanceItem(BaseModel):
    """One ranked importance entry (backend-sorted from ML dict)."""

    feature: str
    importance: float
    rank: int


class PredictionResponse(BaseModel):
    """ML prediction results (echoes ML contract + backend conveniences)."""

    prediction_score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(default=0, description="0 for single /predict; real rank from /rank")
    features: Dict[str, float] = Field(default_factory=dict)
    discretized_features: Dict[str, str] = Field(default_factory=dict)
    missing_features: List[str] = Field(default_factory=list)
    feature_status: str = Field(default="complete", description="complete | incomplete (NaN-filled, not biologically valid)")
    feature_importance: List[FeatureImportanceItem] = Field(default_factory=list)
    model: Optional[str] = Field(default=None, description="ML model kind, e.g. xgboost_pairwise")

    model_config = ConfigDict(extra="allow")


class AnalysisStatusResponse(BaseModel):
    """Status of an analysis job."""

    analysis_id: UUID
    status: str  # pending, processing, completed, failed
    target1: str
    target2: str
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class AnalysisResultResponse(BaseModel):
    """Complete analysis result including prediction and report."""

    analysis_id: Optional[UUID] = None
    target1: str
    target2: str
    status: str = "completed"
    prediction: Optional[PredictionResponse] = None
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    report: Optional[Dict[str, Any]] = None
    rag_error: Optional[str] = None
    persistence: str = Field(default="skipped", description="persisted | skipped")
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra="allow")
