"""Seed clearly-marked DEV targets/pairs. NOT real biological data.

Usage: python -m app.database.seed_dev  (with DATABASE_URL reachable)
Idempotent: inserts only missing gene_symbols.
"""

import asyncio
import sys

DEV_TARGETS = [
    ("CD274", "PD-L1 (DEV placeholder)"),
    ("CTLA4", "CTLA-4 (DEV placeholder)"),
    ("PDCD1", "PD-1 (DEV placeholder)"),
    ("EGFR", "EGFR (DEV placeholder)"),
    ("ERBB2", "HER2 (DEV placeholder)"),
    ("CD47", "CD47 (DEV placeholder)"),
    ("LAG3", "LAG-3 (DEV placeholder)"),
    ("TNFRSF9", "4-1BB (DEV placeholder)"),
    ("ICOS", "ICOS (DEV placeholder)"),
    ("MUC1", "MUC1 (DEV placeholder)"),
]

DEV_PAIRS = [("CD274", "CTLA4"), ("PDCD1", "CTLA4"), ("EGFR", "ERBB2")]


async def main() -> None:
    from sqlalchemy import select

    from app.database.connection import db_manager
    from app.models.database import Target, TargetPair

    db_manager.initialize()
    async with db_manager.session() as session:
        objs: dict[str, Target] = {}
        for sym, name in DEV_TARGETS:
            r = await session.execute(select(Target).where(Target.gene_symbol == sym))
            obj = r.scalar_one_or_none()
            if obj is None:
                obj = Target(
                    gene_symbol=sym,
                    gene_name=name,
                    description="DEV-ONLY placeholder; not real biological data.",
                )
                session.add(obj)
                await session.flush()
            objs[sym] = obj
        for a, b in DEV_PAIRS:
            r = await session.execute(
                select(TargetPair).where(
                    ((TargetPair.target1_id == objs[a].id) & (TargetPair.target2_id == objs[b].id))
                    | ((TargetPair.target1_id == objs[b].id) & (TargetPair.target2_id == objs[a].id))
                )
            )
            if r.scalar_one_or_none() is None:
                session.add(TargetPair(target1_id=objs[a].id, target2_id=objs[b].id))
        print(f"seeded {len(DEV_TARGETS)} dev targets, {len(DEV_PAIRS)} dev pairs")
    await db_manager.close()


if __name__ == "__main__":
    sys.path.insert(0, ".")
    asyncio.run(main())
