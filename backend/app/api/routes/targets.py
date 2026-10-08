"""Target routes: GET /api/targets, GET /api/targets/{symbol}."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/targets")
async def list_targets():
    """List targets: DB when reachable, else clearly-marked dev catalogue."""
    try:
        from sqlalchemy import select

        from app.database.connection import db_manager
        from app.models.database import Target

        async with db_manager.session() as session:
            rows = (await session.execute(select(Target).limit(200))).scalars().all()
            if rows:
                return {
                    "items": [{"gene_symbol": r.gene_symbol, "gene_name": r.gene_name} for r in rows],
                    "total": len(rows),
                    "source": "database",
                }
    except Exception:
        pass
    from app.services.target_store import DEV_TARGETS

    return {"items": DEV_TARGETS, "total": len(DEV_TARGETS), "source": "dev-catalogue (NOT real biology)"}


@router.get("/targets/{symbol}")
async def get_target(symbol: str):
    sym = symbol.strip().upper()
    try:
        from sqlalchemy import select

        from app.database.connection import db_manager
        from app.models.database import Target

        async with db_manager.session() as session:
            r = await session.execute(select(Target).where(Target.gene_symbol == sym))
            obj = r.scalar_one_or_none()
            if obj:
                return {"gene_symbol": obj.gene_symbol, "gene_name": obj.gene_name, "source": "database"}
    except Exception:
        pass
    from fastapi import HTTPException

    from app.services.target_store import DEV_TARGETS

    for t in DEV_TARGETS:
        if t["gene_symbol"] == sym:
            return {**t, "source": "dev-catalogue (NOT real biology)"}
    raise HTTPException(404, f"Target '{sym}' not found")
