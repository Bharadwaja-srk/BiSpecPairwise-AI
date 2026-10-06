"""Pathway / mechanism features (Zhang et al. 2024).

Score = co-occurrence frequency of the gene pair within shared KEGG + GO
pathways. Memberships are NEVER fabricated: load from GMT/CSV adapters.
Empty / unknown genes -> 0 with no error.
"""
from __future__ import annotations

import csv
from typing import Dict, Set


def load_gmt(path: str) -> Dict[str, Set[str]]:
    """Load GMT file -> {pathway_id: {genes}}."""
    pathways: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathways[parts[0]] = set(parts[2:])
    return pathways


def load_csv_membership(path: str, pathway_col: str = "pathway",
                        gene_col: str = "gene") -> Dict[str, Set[str]]:
    pathways: Dict[str, Set[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pathways.setdefault(row[pathway_col], set()).add(row[gene_col])
    return pathways


def pathway_features(
    gene1: str, gene2: str, pathways: Dict[str, Set[str]]
) -> dict:
    shared = sum(1 for genes in pathways.values()
                 if gene1 in genes and gene2 in genes)
    union = sum(1 for genes in pathways.values()
                if gene1 in genes or gene2 in genes)
    jaccard = (shared / union) if union else 0.0
    return {
        "pathway_shared_count": float(shared),
        "pathway_jaccard": float(jaccard),
        "pathway_similarity": float(jaccard),  # alias used by feature_builder
    }
