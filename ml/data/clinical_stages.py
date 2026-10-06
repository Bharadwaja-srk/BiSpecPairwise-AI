"""Clinical stage processing (Zhang et al. 2024, Table 2).

791 tumor-indication BsAbs across 5 ordinal levels. Lower number = more
advanced. No fabrication: unknown labels raise ValueError.
"""
from __future__ import annotations

import pandas as pd

# rank: smaller = more advanced (approved strongest)
STAGE_RANK: dict[str, int] = {
    "approved": 0, "nda": 0, "bla": 0,
    "phase 3": 1, "phase 2/3": 1, "phase2/3": 1,
    "phase 1": 2, "phase 2": 2, "phase1": 2, "phase2": 2,
    "preclinical": 3, "ind": 3, "ind application": 3,
    "discontinued": 4, "pending": 4,
}


def normalize_stage(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("/", "/").split())


def stage_to_rank(label: str) -> int:
    key = normalize_stage(label)
    # also try compact variants
    candidates = [key, key.replace(" ", "")]
    for c in candidates:
        if c in STAGE_RANK:
            return STAGE_RANK[c]
    # substring fallback for labels like "Approved, NDA/BLA"
    for name, rank in STAGE_RANK.items():
        if name in key:
            return rank
    raise ValueError(f"unknown clinical stage: {label!r}")


def process_clinical_df(
    df: pd.DataFrame, target1_col: str = "target1",
    target2_col: str = "target2", stage_col: str = "stage",
) -> pd.DataFrame:
    out = df.copy()
    out["stage_rank"] = out[stage_col].map(stage_to_rank)
    out["pair_id"] = (
        out[target1_col].astype(str) + "__" + out[target2_col].astype(str)
    )
    return out


def load_clinical_csv(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path)
    # accept flexible column names
    lower = {c.lower(): c for c in df.columns}
    t1 = lower.get("target1", lower.get("gene1", "target1"))
    t2 = lower.get("target2", lower.get("gene2", "target2"))
    st = lower.get("stage", lower.get("clinical_stage",
                   lower.get("clinical progress", "stage")))
    return process_clinical_df(df, t1, t2, st)
