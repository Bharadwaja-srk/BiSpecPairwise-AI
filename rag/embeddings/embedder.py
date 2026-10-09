"""Provider-independent embeddings: EmbeddingModel.embed(text) -> vector."""

from __future__ import annotations

import hashlib
import math
import os


class EmbeddingModel:
    """Abstract embedding interface — subclass for OpenAI/HF providers."""

    dim: int = 128

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class HashingEmbedder(EmbeddingModel):
    """Deterministic local-dev embedder (hashing trick + L2 norm). No downloads, no keys."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def get_embedder() -> EmbeddingModel:
    provider = os.getenv("EMBEDDING_PROVIDER", "hashing").lower()
    dim = int(os.getenv("EMBEDDING_DIM", "128"))
    # Future providers (openai/hf) plug in here without touching the pipeline.
    if provider in ("hashing", "local", "mock"):
        return HashingEmbedder(dim=dim)
    return HashingEmbedder(dim=dim)
