"""Biomedical retriever: query over targets/context -> ranked evidence (never fabricated)."""

from __future__ import annotations

from typing import Any

from rag.embeddings.embedder import EmbeddingModel, get_embedder
from rag.retrieval.vector_store import VectorStore

_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def build_query(target1: str, target2: str, context: str = "") -> str:
    parts = [
        f"bispecific antibody {target1} {target2}",
        f"target biology {target1} {target2} mechanism",
        f"clinical evidence {target1} {target2} cancer indication safety expression",
    ]
    if context:
        parts.append(context)
    return " ".join(parts)


def retrieve(
    target1: str,
    target2: str,
    context: str = "",
    top_k: int = 5,
    embedder: EmbeddingModel | None = None,
    store: VectorStore | None = None,
) -> list[dict[str, Any]]:
    """Return [{text,title,source,url,metadata,score}] or [] when nothing matches."""
    store = store or get_store()
    if store.count() == 0:
        return []
    embedder = embedder or get_embedder()
    qv = embedder.embed(build_query(target1, target2, context))
    hits = store.search(qv, top_k=top_k)
    out = []
    for h in hits:
        md = h.get("metadata", {}) or {}
        out.append(
            {
                "text": h.get("text", ""),
                "title": h.get("title", md.get("title", "untitled")),
                "source": h.get("source", md.get("source", "unknown")),
                "url": h.get("url", md.get("url")),
                "source_id": md.get("source_id"),
                "metadata": md,
                "score": round(float(h.get("score", 0.0)), 4),
            }
        )
    return out
