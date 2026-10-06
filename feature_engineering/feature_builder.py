"""Combine all biological features into a stable ordered vector (Zhang et al. 2024).

Inputs are precomputed per-target/per-pair quantities (arrays or scalars) so
this builder stays decoupled from AnnData I/O. Missing values follow
feature_schema.py policies (nan, except pathway 0.0).
"""
from __future__ import annotations

import numpy as np

from .biological_features.correlation import pearson_correlation
from .biological_features.double_positive import (
    double_positive_proportion,
    single_proportions,
)
from .biological_features.gene_embedding import pair_similarity
from .biological_features.pathway import pathway_features
from .biological_features.safety import safety_score
from .biological_features.target_activity import mean_expression
from .feature_schema import FEATURE_ORDER, validate_feature_dict


def build_features(
    target1: str,
    target2: str,
    expr1: np.ndarray,
    expr2: np.ndarray,
    expr_tumor_1: float = float("nan"),
    expr_normal_1: float = float("nan"),
    expr_tumor_2: float = float("nan"),
    expr_normal_2: float = float("nan"),
    pathways: dict | None = None,
    embeddings: dict | None = None,
    threshold: float = 0.0,
) -> dict:
    e1 = np.asarray(expr1, dtype=float).ravel()
    e2 = np.asarray(expr2, dtype=float).ravel()

    singles = single_proportions(e1, e2, threshold)
    dp = double_positive_proportion(e1, e2, threshold)

    try:
        safe = safety_score(expr_tumor_1, expr_normal_1,
                            expr_tumor_2, expr_normal_2)
    except (TypeError, ValueError):
        safe = float("nan")

    if pathways is None:
        pw = {"pathway_shared_count": 0.0, "pathway_jaccard": 0.0,
              "pathway_similarity": 0.0}
    else:
        pw = pathway_features(target1, target2, pathways)

    if embeddings is None:
        emb = {"gene_embedding_distance": float("nan"),
               "gene_embedding_similarity": float("nan")}
    else:
        try:
            emb = pair_similarity(target1, target2, embeddings)
        except ValueError:
            emb = {"gene_embedding_distance": float("nan"),
                   "gene_embedding_similarity": float("nan")}

    raw = {
        "target_activity_1": mean_expression(e1),
        "target_activity_2": mean_expression(e2),
        "single_ratio_max": singles["single_ratio_max"],
        "single_ratio_min": singles["single_ratio_min"],
        "double_positive": dp,
        "safety_score": safe,
        "pathway_similarity": pw["pathway_similarity"],
        "pathway_shared_count": pw["pathway_shared_count"],
        "gene_embedding_similarity": emb["gene_embedding_similarity"],
        "gene_embedding_distance": emb["gene_embedding_distance"],
        "pearson_correlation": pearson_correlation(e1, e2),
    }
    feats = validate_feature_dict(raw)
    return {
        "target1": target1,
        "target2": target2,
        "features": feats,
        "vector": [feats[k] for k in FEATURE_ORDER],
    }
