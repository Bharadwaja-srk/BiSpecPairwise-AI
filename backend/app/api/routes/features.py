"""Feature-schema route: GET /api/features (contract discovery for frontend)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/features")
async def get_feature_schema():
    try:
        from feature_engineering.feature_schema import FEATURE_SCHEMA

        return {"features": FEATURE_SCHEMA, "source": "feature_engineering/feature_schema.py"}
    except ImportError:
        from app.services.target_store import FEATURE_ORDER

        return {
            "features": [{"name": n} for n in FEATURE_ORDER],
            "source": "backend fallback (feature_engineering not importable)",
        }
