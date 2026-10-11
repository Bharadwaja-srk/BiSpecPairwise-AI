"""Offline feature-store builder (Day 4).

Reads one processed tumor AnnData + a pairs CSV (target1,target2 columns),
computes the 11 BSPAI features per pair with grouped top-2 cell populations,
and writes a versioned JSON store. No fabrication: unavailable inputs yield
documented NaN/0.0 per feature_schema.py; every row carries provenance.

Usage (needs anndata; scanpy NOT required):
  python -m feature_engineering.build_feature_store \
    --h5ad data/processed/hnscc_raw.h5ad --tumor HNSCC \
    --dataset-id GSE164690 --pairs docs/target_universe/verified_pairs_inventory.csv \
    --embeddings data/embeddings/gene2vec_dim_200_iter_9.txt \
    --celltype-obs sample --out data/precomputed_features.hnscc.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

GENERATOR_VERSION = "build_feature_store/1.0"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--tumor", required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--pathways", default=None)
    ap.add_argument("--normal-obs", default=None,
                    help="obs column + value marking normal cells, e.g. origin:nLung")
    ap.add_argument("--celltype-obs", default="sample")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import anndata as ad
    import numpy as np
    import pandas as pd

    from feature_engineering.biological_features.gene_embedding import load_gene2vec_txt
    from feature_engineering.biological_features.target_activity import (
        top_expressing_populations,
    )
    from feature_engineering.feature_builder import build_features
    from feature_engineering.feature_schema import FEATURE_ORDER

    adata = ad.read_h5ad(args.h5ad)
    log(f"loaded {args.h5ad}: {adata.shape}")
    pairs = pd.read_csv(args.pairs)
    log(f"pairs: {len(pairs)}")

    embeddings = load_gene2vec_txt(args.embeddings) if args.embeddings else None
    log(f"embeddings: {len(embeddings) if embeddings else 0} genes")
    pathways = None
    if args.pathways:
        from feature_engineering.biological_features.pathway import load_gmt

        pathways = load_gmt(args.pathways)
        log(f"pathways: {len(pathways)} sets")
    else:
        log("pathways: none -> 0.0 per schema (documented gap: no KEGG/GO GMT staged)")

    alias_map = {"CD20": "MS4A1", "VSIR": "C10ORF54", "NECTIN4": "PVRL4"}
    obs_col = args.celltype_obs
    if obs_col not in adata.obs:
        print(f"ERROR: obs column {obs_col!r} missing (have {list(adata.obs.columns)[:8]})")
        return 2
    labels = np.asarray(adata.obs[obs_col].astype(str))
    norm_col, norm_val = (args.normal_obs.split(":", 1) + [None])[:2] if args.normal_obs else (None, None)

    var = np.asarray(adata.var_names)
    gene_index = {g.strip().upper(): i for i, g in enumerate(var)}

    def resolve(symbol: str) -> str | None:
        s = symbol.strip().upper()
        if s in gene_index:
            return s
        a = alias_map.get(s)
        return a if a in gene_index else None

    def col_vec(g: str) -> np.ndarray:
        m = adata.X[:, gene_index[g]]
        return np.asarray(m.todense()).ravel() if hasattr(m, "todense") else np.asarray(m).ravel()

    def pop_mean(g: str, mask: np.ndarray) -> float:
        v = col_vec(g)[mask]
        v = v[~np.isnan(v)]
        return float(np.mean(v)) if v.size else float("nan")

    rows = []
    for _, pr in pairs.iterrows():
        t1, t2 = str(pr["target1"]).strip().upper(), str(pr["target2"]).strip().upper()
        g1, g2 = resolve(t1), resolve(t2)
        missing_pre = [g for g, r in ((t1, g1), (t2, g2)) if r is None]
        if g1 is None or g2 is None:
            feats = {k: float("nan") for k in FEATURE_ORDER}
            feats["pathway_similarity"] = 0.0
            feats["pathway_shared_count"] = 0.0
            rows.append({"target1": t1, "target2": t2, "tumor": args.tumor,
                         "features": feats, "missing_features": missing_pre,
                         "feature_status": "incomplete",
                         "provenance": {"reason": "target absent from matrix"}})
            continue
        e1, e2 = col_vec(g1), col_vec(g2)
        pops = sorted(set(top_expressing_populations(e1, labels, 2)) |
                      set(top_expressing_populations(e2, labels, 2)))
        mask = np.array([l in pops for l in labels])
        se1, se2 = e1[mask], e2[mask]
        tm1 = tm2 = nm1 = nm2 = float("nan")
        normal_flag = "unavailable"
        if norm_col and norm_col in adata.obs:
            nmask = np.asarray(adata.obs[norm_col].astype(str)) == str(norm_val)
            tmask = ~nmask
            tm1, nm1 = pop_mean(g1, tmask), pop_mean(g1, nmask)
            tm2, nm2 = pop_mean(g2, tmask), pop_mean(g2, nmask)
            normal_flag = "computed" if nmask.any() else "no-normal-cells"
        out = build_features(t1, t2, se1, se2, tm1, nm1, tm2, nm2,
                             pathways=pathways,
                             embeddings=embeddings,
                             threshold=args.threshold)
        feats = out["features"]
        missing = [k for k, v in feats.items() if isinstance(v, float) and np.isnan(v)]
        rows.append({"target1": t1, "target2": t2, "tumor": args.tumor,
                     "features": feats,
                     "missing_features": missing,
                     "feature_status": "complete" if not missing else "incomplete",
                     "provenance": {
                         "dataset": args.dataset_id,
                         "matrix": args.h5ad,
                         "generator": GENERATOR_VERSION,
                         "populations": pops,
                         "population_column": obs_col,
                         "cells_used": int(mask.sum()),
                         "threshold": args.threshold,
                         "normal_split": normal_flag,
                         "embeddings": args.embeddings,
                         "pathways": args.pathways,
                     }})
    store = {"version": GENERATOR_VERSION,
             "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "tumor": args.tumor, "dataset": args.dataset_id,
             "feature_order": FEATURE_ORDER, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(store, f)
    n_ok = sum(1 for r in rows if r["feature_status"] == "complete")
    log(f"wrote {args.out}: {len(rows)} rows, {n_ok} complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
