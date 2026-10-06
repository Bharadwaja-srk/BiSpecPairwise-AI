"""Scanpy-based scRNA-seq preprocessing (Zhang et al. 2024, Methods).

Pipeline: standardize -> QC -> filter -> normalize -> HVG -> PCA ->
clustering. Each step is a standalone function; `run_pipeline` supports
skip-if-cached via `adata.uns['bspai_<step>']` markers.

Scanpy is imported lazily so unit tests / lightweight envs work without it.
Real runs require scanpy+anndata (see requirements-ml.txt, Docker py3.11).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreprocessConfig:
    min_genes: int = 200
    min_cells: int = 3
    mt_pct_max: float = 20.0
    n_top_genes: int = 2000
    n_pcs: int = 50
    resolution: float = 0.5
    target_sum: float = 1e4


def _require_scanpy() -> Any:
    try:
        import scanpy as sc  # type: ignore
    except ImportError as e:
        raise ImportError(
            "scanpy is required for preprocessing. "
            "pip install -r requirements-ml.txt (Python 3.11+, Linux container)."
        ) from e
    return sc


def load_data(path: str, **kwargs: Any) -> Any:
    """Load h5ad / csv / txt into AnnData."""
    import os

    import anndata as ad
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext == ".h5ad":
        return ad.read_h5ad(path, **kwargs)
    df = pd.read_csv(path, index_col=0)
    return ad.AnnData(df.values, obs=pd.DataFrame(index=df.index),
                      var=pd.DataFrame(index=df.columns))


def quality_control(adata: Any, cfg: PreprocessConfig = PreprocessConfig()) -> Any:
    """Flag MT %, store QC metrics; does not filter yet.

    percent_top=None skips the per-cell top-gene ranking temporaries (unused
    downstream); all QC sums stay sparse.
    """
    sc = _require_scanpy()
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    adata.uns["bspai_qc"] = True
    gc.collect()
    return adata


def filter_cells_genes(adata: Any, cfg: PreprocessConfig = PreprocessConfig()) -> Any:
    sc = _require_scanpy()
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells)
    if "pct_counts_mt" in adata.obs:
        adata = adata[adata.obs["pct_counts_mt"] < cfg.mt_pct_max].copy()
    adata.uns["bspai_filtered"] = True
    gc.collect()
    return adata


def normalize(adata: Any, cfg: PreprocessConfig = PreprocessConfig()) -> Any:
    """Total-count normalize + log1p, both sparse-inplace; no dense copies."""
    sc = _require_scanpy()
    sc.pp.normalize_total(adata, target_sum=cfg.target_sum)
    sc.pp.log1p(adata)
    adata.uns["bspai_normalized"] = True
    gc.collect()
    return adata


def highly_variable_genes(
    adata: Any, cfg: PreprocessConfig = PreprocessConfig()
) -> Any:
    sc = _require_scanpy()
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_top_genes, subset=True)
    adata.uns["bspai_hvg"] = True
    gc.collect()  # release the pre-subset full-width matrix
    return adata


def run_pca(adata: Any, cfg: PreprocessConfig = PreprocessConfig()) -> Any:
    sc = _require_scanpy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(cfg.n_pcs, adata.n_vars - 1, adata.n_obs - 1))
    adata.uns["bspai_pca"] = True
    return adata


def cluster(
    adata: Any, cfg: PreprocessConfig = PreprocessConfig(), key: str = "leiden"
) -> Any:
    sc = _require_scanpy()
    sc.pp.neighbors(adata)
    try:
        sc.tl.leiden(adata, resolution=cfg.resolution, key_added=key)
    except Exception:
        sc.tl.louvain(adata, resolution=cfg.resolution, key_added=key)
    adata.uns["bspai_clustered"] = True
    return adata


def run_pipeline(adata: Any, cfg: PreprocessConfig = PreprocessConfig()) -> Any:
    """Run steps in order; skip steps already marked in adata.uns."""
    steps = [
        ("bspai_qc", quality_control),
        ("bspai_filtered", filter_cells_genes),
        ("bspai_normalized", normalize),
        ("bspai_hvg", highly_variable_genes),
        ("bspai_pca", run_pca),
        ("bspai_clustered", cluster),
    ]
    for marker, fn in steps:
        if not adata.uns.get(marker):
            adata = fn(adata, cfg)
    return adata


__all__ = [
    "PreprocessConfig", "load_data", "quality_control", "filter_cells_genes",
    "normalize", "highly_variable_genes", "run_pca", "cluster", "run_pipeline",
]
