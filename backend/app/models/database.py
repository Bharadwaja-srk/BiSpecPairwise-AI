"""
Database models for BSPAI Backend.

Defines SQLAlchemy models for Target, TargetPair, Analysis, Prediction, Report, and Evidence.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class AnalysisStatus(str, PyEnum):
    """Status of an analysis job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Target(Base):
    """Target (gene/protein) information."""

    __tablename__ = "targets"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    gene_symbol: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    gene_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    uniprot_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    ensembl_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    function: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pathway: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    subcellular_location: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    tissue_specificity: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    pairs_as_target1: Mapped[list["TargetPair"]] = relationship(
        "TargetPair", foreign_keys="TargetPair.target1_id", back_populates="target1"
    )
    pairs_as_target2: Mapped[list["TargetPair"]] = relationship(
        "TargetPair", foreign_keys="TargetPair.target2_id", back_populates="target2"
    )


class TargetPair(Base):
    """A pair of targets for bispecific antibody analysis."""

    __tablename__ = "target_pairs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    target1_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("targets.id"), nullable=False, index=True
    )
    target2_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("targets.id"), nullable=False, index=True
    )
    # Clinical stage info (from paper: 1=Approved, 2=Phase 3, 3=Phase 1/2, 4=Preclinical, 5=Discontinued)
    clinical_stage: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bsab_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # e.g., "KN046"
    indication: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    target1: Mapped["Target"] = relationship("Target", foreign_keys=[target1_id], back_populates="pairs_as_target1")
    target2: Mapped["Target"] = relationship("Target", foreign_keys=[target2_id], back_populates="pairs_as_target2")
    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="target_pair", cascade="all, delete-orphan")

    # Ensure unique pairs (order-independent)
    __table_args__ = (
        UniqueConstraint("target1_id", "target2_id", name="uq_target_pair"),
        Index("ix_target_pair_targets", "target1_id", "target2_id"),
    )


class Analysis(Base):
    """An analysis request for a target pair."""

    __tablename__ = "analyses"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    target_pair_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("target_pairs.id"), nullable=False, index=True
    )
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False, index=True
    )
    request_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    target_pair: Mapped["TargetPair"] = relationship("TargetPair", back_populates="analyses")
    prediction: Mapped[Optional["Prediction"]] = relationship(
        "Prediction", back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )
    report: Mapped[Optional["Report"]] = relationship(
        "Report", back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )


class Prediction(Base):
    """ML model prediction results for an analysis."""

    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False, unique=True, index=True
    )
    prediction_score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_candidates: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    features: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    feature_importance: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="prediction")


class Report(Base):
    """Generated interpretive report for an analysis."""

    __tablename__ = "reports"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("analyses.id"), nullable=False, unique=True, index=True
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    biological_interpretation: Mapped[str] = mapped_column(Text, nullable=False)
    mechanistic_rationale: Mapped[str] = mapped_column(Text, nullable=False)
    safety_considerations: Mapped[str] = mapped_column(Text, nullable=False)
    clinical_evidence: Mapped[str] = mapped_column(Text, nullable=False)
    limitations: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    confidence: Mapped[str] = mapped_column(String(50), nullable=False)  # high, medium, low
    retrieved_evidence: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    analysis: Mapped["Analysis"] = relationship("Analysis", back_populates="report")
    evidence_items: Mapped[list["Evidence"]] = relationship(
        "Evidence", back_populates="report", cascade="all, delete-orphan"
    )


class Evidence(Base):
    """Retrieved biomedical evidence linked to a report."""

    __tablename__ = "evidence"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    report_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("reports.id"), nullable=False, index=True
    )
    source_title: Mapped[str] = mapped_column(String(500), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # paper, clinical_trial, database, etc.
    source_url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    source_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)  # PMID, DOI, NCT ID, etc.
    publication_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    # NOTE: attribute named extra_metadata (column "metadata") because
    # `metadata` is reserved on SQLAlchemy Declarative Base.
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    report: Mapped["Report"] = relationship("Report", back_populates="evidence_items")