"""Gene embedding similarity, Gene2Vec-style (Zhang et al. 2024).

Euclidean d(x,y) = sqrt(sum((xi-yi)^2)); similarity = 1/(1+d).
Embeddings loaded from npy (dict) / csv (gene,dim...) adapters.
Missing genes -> ValueError listing them (caller maps to NaN per schema).
Synthetic embeddings allowed ONLY in tests — never presented as real.
"""
from __future__ import annotations

import numpy as np


def load_embeddings_csv(path: str) -> dict:
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    return {g: df.loc[g].to_numpy(dtype=float) for g in df.index}


def load_gene2vec_txt(path: str, normalize: bool = True) -> dict:
    """Load Gene2Vec word2vec-format txt (Du et al. 2019, dim_200_iter_9).

    Format: one gene per line, ``SYMBOL v1 v2 ... v200`` space-separated,
    no header. Symbols are uppercased/stripped when ``normalize=True``.
    Duplicate symbols keep the first occurrence (file has IGK/NDUFAF4 x2).
    Raises ValueError on dim mismatch or empty file. Never invents vectors.
    """
    import numpy as np

    embeddings: dict = {}
    expected_dim: int | None = None
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            symbol = parts[0].strip().upper() if normalize else parts[0]
            if symbol in embeddings:
                continue  # keep first; duplicates documented in manifest
            try:
                vec = np.asarray([float(x) for x in parts[1:]], dtype=float)
            except ValueError as e:
                raise ValueError(f"non-numeric vector at line {lineno} ({symbol})") from e
            if expected_dim is None:
                expected_dim = vec.size
                if expected_dim == 0:
                    raise ValueError(f"empty vector at line {lineno} ({symbol})")
            elif vec.size != expected_dim:
                raise ValueError(
                    f"dim mismatch at line {lineno} ({symbol}): "
                    f"{vec.size} vs {expected_dim}"
                )
            embeddings[symbol] = vec
    if not embeddings:
        raise ValueError(f"no embeddings loaded from {path}")
    return embeddings


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    a = np.asarray(vec1, dtype=float).ravel()
    b = np.asarray(vec2, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"embedding dim mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.sum((a - b) ** 2)))


def embedding_similarity(vec1: np.ndarray, vec2: np.ndarray) -> dict:
    d = euclidean_distance(vec1, vec2)
    return {
        "gene_embedding_distance": d,
        "gene_embedding_similarity": float(1.0 / (1.0 + d)),
    }


def pair_similarity(gene1: str, gene2: str, embeddings: dict) -> dict:
    missing = [g for g in (gene1, gene2) if g not in embeddings]
    if missing:
        raise ValueError(f"missing embeddings for genes: {missing}")
    return embedding_similarity(embeddings[gene1], embeddings[gene2])

