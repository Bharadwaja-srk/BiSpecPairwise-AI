"""Machine-readable feature schema (consumed by ml/ pipeline).

Each entry: name / type / description / source / range / missing strategy.
"""
from __future__ import annotations

FEATURE_SCHEMA: list[dict] = [
    {"name": "target_activity_1", "type": "float",
     "description": "Mean expression of target1 in top expressing cell pops",
     "source": "scRNA-seq (target_activity.py)", "range": "[0, inf)",
     "missing": "nan"},
    {"name": "target_activity_2", "type": "float",
     "description": "Mean expression of target2 in top expressing cell pops",
     "source": "scRNA-seq (target_activity.py)", "range": "[0, inf)",
     "missing": "nan"},
    {"name": "single_ratio_max", "type": "float",
     "description": "Max of the two single-positive proportions",
     "source": "scRNA-seq (double_positive.py)", "range": "[0, 1]",
     "missing": "nan"},
    {"name": "single_ratio_min", "type": "float",
     "description": "Min of the two single-positive proportions",
     "source": "scRNA-seq (double_positive.py)", "range": "[0, 1]",
     "missing": "nan"},
    {"name": "double_positive", "type": "float",
     "description": "P_g1,g2: fraction of cells expressing both targets > T",
     "source": "scRNA-seq (double_positive.py)", "range": "[0, 1]",
     "missing": "nan"},
    {"name": "safety_score", "type": "float",
     "description": "Harmonic-mean tumor-vs-normal score dx*dy/(dx+dy)",
     "source": "tumor+normal expression (safety.py)", "range": "(-inf, inf)",
     "missing": "0.0 when dx+dy==0; else nan"},
    {"name": "pathway_similarity", "type": "float",
     "description": "Jaccard of KEGG+GO pathway memberships",
     "source": "KEGG/GO adapters (pathway.py)", "range": "[0, 1]",
     "missing": "0.0 (unknown genes)"},
    {"name": "pathway_shared_count", "type": "float",
     "description": "Count of shared pathways",
     "source": "KEGG/GO adapters (pathway.py)", "range": "[0, inf)",
     "missing": "0.0"},
    {"name": "gene_embedding_similarity", "type": "float",
     "description": "1/(1+Euclidean) Gene2Vec-style similarity",
     "source": "gene embeddings (gene_embedding.py)", "range": "(0, 1]",
     "missing": "nan (missing gene)"},
    {"name": "gene_embedding_distance", "type": "float",
     "description": "Euclidean distance between gene vectors",
     "source": "gene embeddings (gene_embedding.py)", "range": "[0, inf)",
     "missing": "nan"},
    {"name": "pearson_correlation", "type": "float",
     "description": "Pearson r of target expressions across single cells",
     "source": "scRNA-seq (correlation.py)", "range": "[-1, 1]",
     "missing": "nan (<3 cells); 0.0 if zero variance"},
]

FEATURE_ORDER: list[str] = [f["name"] for f in FEATURE_SCHEMA]


def get_feature_names() -> list[str]:
    return list(FEATURE_ORDER)


def validate_feature_dict(d: dict) -> dict:
    """Fill missing keys with nan; return ordered float dict."""
    import math

    out: dict = {}
    for name in FEATURE_ORDER:
        v = d.get(name, float("nan"))
        try:
            out[name] = float(v)
        except (TypeError, ValueError):
            out[name] = float("nan")
    return out
