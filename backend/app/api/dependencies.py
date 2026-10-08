"""
FastAPI dependency injection for BSPAI Backend.

Provides dependencies for database, ML service, RAG service, and common parameters.
"""

from typing import Annotated, Optional
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.database.connection import db_manager, get_db
from app.services.ml_service import MLService, get_ml_service

try:  # RAG service may not exist in minimal installs; don't break imports
    from app.services.rag_service import RAGService, get_rag_service
except ImportError:  # pragma: no cover
    RAGService = None  # type: ignore

    def get_rag_service():  # type: ignore
        raise RuntimeError("RAG service not available")


# Database dependency
AsyncSessionDep = Annotated[AsyncSession, Depends(get_db)]

# Settings dependency
SettingsDep = Annotated[Settings, Depends(get_settings)]

# Service dependencies
MLServiceDep = Annotated[MLService, Depends(get_ml_service)]
try:
    RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]  # type: ignore
except Exception:  # pragma: no cover
    RAGServiceDep = None  # type: ignore


# Common query parameters
class PaginationParams:
    """Pagination parameters for list endpoints."""

    def __init__(
        self,
        page: Annotated[int, Query(ge=1, description="Page number")] = 1,
        page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    ):
        self.page = page
        self.page_size = page_size
        self.offset = (page - 1) * page_size
        self.limit = page_size


PaginationDep = Annotated[PaginationParams, Depends()]


class TargetPairParams:
    """Parameters for target pair validation."""

    def __init__(
        self,
        target1: Annotated[str, Query(min_length=1, max_length=50, description="First target gene symbol")],
        target2: Annotated[str, Query(min_length=1, max_length=50, description="Second target gene symbol")],
    ):
        self.target1 = target1.upper()
        self.target2 = target2.upper()

    def __eq__(self, other):
        if not isinstance(other, TargetPairParams):
            return False
        # Order-independent comparison
        return {self.target1, self.target2} == {other.target1, other.target2}

    def __hash__(self):
        return hash(frozenset([self.target1, self.target2]))


TargetPairQueryDep = Annotated[TargetPairParams, Depends()]


# Optional authentication (for future use)
async def get_current_user_id(
    authorization: Annotated[Optional[str], Header()] = None,
) -> Optional[str]:
    """Extract user ID from authorization header (placeholder for future auth)."""
    if not authorization:
        return None
    # TODO: Implement JWT validation
    return "anonymous"


CurrentUserDep = Annotated[Optional[str], Depends(get_current_user_id)]


# Validation helpers
async def validate_target_exists(
    target_symbol: str,
    session: AsyncSessionDep,
) -> None:
    """Validate that a target exists in the database."""
    from sqlalchemy import select
    from app.models.database import Target

    result = await session.execute(select(Target).where(Target.gene_symbol == target_symbol.upper()))
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target '{target_symbol}' not found in database",
        )


async def validate_target_pair(
    target1: str,
    target2: str,
    session: AsyncSessionDep,
) -> UUID:
    """Validate target pair exists and return its ID (order-independent)."""
    from sqlalchemy import select
    from app.models.database import Target, TargetPair

    t1, t2 = target1.upper(), target2.upper()
    r1 = await session.execute(select(Target).where(Target.gene_symbol == t1))
    r2 = await session.execute(select(Target).where(Target.gene_symbol == t2))
    o1, o2 = r1.scalar_one_or_none(), r2.scalar_one_or_none()
    if not o1 or not o2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target pair '{t1}-{t2}' not found in database",
        )
    result = await session.execute(
        select(TargetPair).where(
            ((TargetPair.target1_id == o1.id) & (TargetPair.target2_id == o2.id))
            | ((TargetPair.target1_id == o2.id) & (TargetPair.target2_id == o1.id))
        )
    )
    pair = result.scalar_one_or_none()
    if not pair:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target pair '{t1}-{t2}' not found in database",
        )
    return pair.id


def normalize_pair(t1: str, t2: str) -> tuple[str, str]:
    """Uppercase + validate two gene symbols (order preserved)."""
    a, b = t1.strip().upper(), t2.strip().upper()
    if not a or not b:
        raise HTTPException(status_code=422, detail="target symbols must be non-empty")
    if a == b:
        raise HTTPException(status_code=422, detail="target1 and target2 must differ")
    return a, b