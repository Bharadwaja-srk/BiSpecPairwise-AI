"""
Pydantic schemas for report API requests and responses.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EvidenceItem(BaseModel):
    """Retrieved biomedical evidence item."""

    source_title: str
    source_type: str  # paper, clinical_trial, database, etc.
    source_url: Optional[str] = None
    source_id: Optional[str] = None  # PMID, DOI, NCT ID, etc.
    publication_date: Optional[datetime] = None
    snippet: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ReportResponse(BaseModel):
    """Complete structured report response."""

    report_id: UUID
    analysis_id: UUID
    target1: str
    target2: str
    summary: str
    prediction: dict  # {score, rank}
    biological_interpretation: str
    mechanistic_rationale: str
    safety_considerations: str
    clinical_evidence: str
    retrieved_evidence: List[EvidenceItem]
    limitations: List[str]
    confidence: str  # high, medium, low
    model_name: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "report_id": "550e8400-e29b-41d4-a716-446655440000",
                "analysis_id": "550e8400-e29b-41d4-a716-446655440001",
                "target1": "CD274",
                "target2": "CTLA4",
                "summary": "Analysis of CD274 (PD-L1) and CTLA4 as dual targets...",
                "prediction": {"score": 0.93, "rank": 4},
                "biological_interpretation": "CD274 and CTLA4 are co-expressed...",
                "mechanistic_rationale": "Dual blockade of PD-1/PD-L1 and CTLA4 pathways...",
                "safety_considerations": "Low safety score indicates...",
                "clinical_evidence": "KN046 (CD274+CTLA4) in Phase III...",
                "retrieved_evidence": [],
                "limitations": ["Limited clinical data for this specific combination"],
                "confidence": "high",
                "model_name": "gpt-4",
                "created_at": "2024-01-15T10:30:00Z",
            }
        },
    )


class ReportListResponse(BaseModel):
    """Paginated list of reports."""

    items: List[ReportResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ReportSummary(BaseModel):
    """Brief report summary for listings."""

    report_id: UUID
    analysis_id: UUID
    target1: str
    target2: str
    prediction_score: float
    rank: Optional[int] = None
    confidence: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)