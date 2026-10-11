"""Provider-independent embeddings: EmbeddingModel.embed(text) -> vector."""

from __future__ import annotations

import hashlib
import json
import math
import os

DEFAULT_ST_MODEL = "NeuML/pubmedbert-base-embeddings"
DEFAULT_ST_REVISION = "b79526d6ef3645e0df4530322e266f24c829f5ef"
DEFAULT_ST_DIM = 768


class EmbeddingConfigError(Exception):
    """Raised when the requested embedding backend cannot be constructed.

    Never silently substituted: callers must fail loudly instead of
    falling back to hashing embeddings for production retrieval.
    """


class EmbeddingModel:
    """Abstract embedding interface — subclass for OpenAI/HF providers."""

    dim: int = 128
    provider: str = "base"

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def provenance(self) -> dict:
        return {"provider": self.provider, "dim": self.dim}

    def config_fingerprint(self) -> str:
        raw = json.dumps(self.provenance(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class HashingEmbedder(EmbeddingModel):
    """Deterministic local-dev embedder (hashing trick + L2 norm). No downloads, no keys.

    DEV/TEST ONLY: must never serve production biomedical retrieval. Any store
    built with it carries fingerprint provider=hashing, detectable via
    VectorStore.fingerprints().
    """

    provider = "hashing"

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class SentenceTransformerEmbedder(EmbeddingModel):
    """Local sentence-transformers backend (default: PubMedBERT biomedical).

    Lazy import so lightweight envs fail with a clear error, never a silent
    hashing fallback. Model ID + revision pinned via env; dim validated
    against the loaded model. Changing model/revision/dim changes the
    fingerprint -> indexed documents must be re-embedded (see VectorStore).
    """

    provider = "sentence-transformers"

    def __init__(self, model_id: str = DEFAULT_ST_MODEL,
                 revision: str = DEFAULT_ST_REVISION,
                 expected_dim: int = DEFAULT_ST_DIM):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingConfigError(
                "sentence-transformers (+torch) is not installed. "
                "Install in the RAG container (see rag/requirements.txt); "
                "refusing to fall back to hashing embeddings."
            ) from e
        try:
            self._model = SentenceTransformer(model_id, revision=revision)
        except Exception as e:
            raise EmbeddingConfigError(
                f"failed to load embedding model {model_id}@{revision}: "
                f"{type(e).__name__}: {e}"
            ) from e
        self.model_id = model_id
        self.revision = revision
        dim = int(self._model.get_sentence_embedding_dimension())
        if dim != int(expected_dim):
            raise EmbeddingConfigError(
                f"embedding dim mismatch for {model_id}: model={dim} "
                f"expected={expected_dim} (EMBEDDING_DIM). Refusing to index "
                "with an unexpected dimension."
            )
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return [float(v) for v in self._model.encode(text, normalize_embeddings=True)]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[float(v) for v in row]
                for row in self._model.encode(texts, normalize_embeddings=True)]

    def provenance(self) -> dict:
        return {"provider": self.provider, "model_id": self.model_id,
                "revision": self.revision, "dim": self.dim}


def get_embedder() -> EmbeddingModel:
    provider = os.getenv("EMBEDDING_PROVIDER", "hashing").lower()
    if provider in ("hashing", "local", "mock"):
        return HashingEmbedder(dim=int(os.getenv("EMBEDDING_DIM", "128")))
    if provider in ("sentence-transformers", "st", "hf", "pubmedbert"):
        return SentenceTransformerEmbedder(
            model_id=os.getenv("EMBEDDING_MODEL", DEFAULT_ST_MODEL),
            revision=os.getenv("EMBEDDING_REVISION", DEFAULT_ST_REVISION),
            expected_dim=int(os.getenv("EMBEDDING_DIM", str(DEFAULT_ST_DIM))),
        )
    raise EmbeddingConfigError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. Valid: hashing (dev only), "
        "sentence-transformers. Refusing to guess a backend."
    )
