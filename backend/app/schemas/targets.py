"""
Pydantic schemas for target API requests and responses.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TargetBase(BaseModel):
    """Base target fields."""

    gene_symbol: str = Field(..., min_length=1, max_length=50)
    gene_name: Optional[str] = Field(None, max_length=200)
    uniprot_id: Optional[str] = Field(None, max_length=50)
    ensembl_id: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    function: Optional[str] = None
    pathway: Optional[str] = None
    subcellular_location: Optional[str] = Field(None, max_length=200)
    tissue_specificity: Optional[str] = None


class TargetCreate(TargetBase):
    """Schema for creating a target."""

    pass


class TargetUpdate(BaseModel):
    """Schema for updating a target."""

    gene_name: Optional[str] = Field(None, max_length=200)
    uniprot_id: Optional[str] = Field(None, max_length=50)
    ensembl_id: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    function: Optional[str] = None
    pathway: Optional[str] = None
    subcellular_location: Optional[str] = Field(None, max_length=200)
    tissue_specificity: Optional[str] = None


class TargetResponse(TargetBase):
    """Target response with metadata."""

    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TargetListResponse(BaseModel):
    """Paginated list of targets."""

    items: List[TargetResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class TargetPairBase(BaseModel):
    """Base target pair fields."""

    target1_symbol: str = Field(..., min_length=1, max_length=50)
    target2_symbol: str = Field(..., min_length=1, max_length=50)
    clinical_stage: Optional[int] = Field(None, ge=1, le=5)
    bsab_name: Optional[str] = Field(None, max_length=200)
    indication: Optional[str] = Field(None, max_length=200)


class TargetPairCreate(TargetPairBase):
    """Schema for creating a target pair."""

    pass


class TargetPairResponse(TargetPairBase):
    """Target pair response with metadata."""

    id: UUID
    target1: TargetResponse
    target2: TargetResponse
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TargetPairListResponse(BaseModel):
    """Paginated list of target pairs."""

    items: List[TargetPairResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class RankingItem(BaseModel):
    """Single item in the ranking list."""

    rank: int = Field(..., ge=1)
    target1_symbol: str
    target2_symbol: str
    target1_name: Optional[str] = None
    target2_name: Optional[str] = None
    prediction_score: float = Field(..., ge=0.0, le=1.0)
    clinical_stage: Optional[int] = None
    bsab_name: Optional[str] = None
    indication: Optional[str] = None


class RankingResponse(BaseModel):
    """Ranking of target pairs by prediction score."""

    items: List[RankingItem]
    total: int
    model_version: Optional[str] = None