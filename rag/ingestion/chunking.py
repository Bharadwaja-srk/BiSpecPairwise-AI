"""Biomedical-aware chunking: every chunk preserves source metadata."""

from __future__ import annotations

from typing import Any


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def chunk_document(doc: dict[str, Any], chunk_size: int = 800, overlap: int = 100) -> list[dict[str, Any]]:
    out = []
    for idx, c in enumerate(chunk_text(doc.get("text", ""), chunk_size, overlap)):
        out.append(
            {
                "text": c,
                "title": doc.get("title", "untitled"),
                "source": doc.get("source", "unknown"),
                "url": doc.get("url"),
                "source_id": doc.get("source_id"),
                "metadata": {**(doc.get("metadata", {}) or {}),
                             "source_id": doc.get("source_id"),
                             "chunk_index": idx},
            }
        )
    return out


def chunk_documents(docs: list[dict[str, Any]], chunk_size: int = 800, overlap: int = 100) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in docs:
        out.extend(chunk_document(d, chunk_size, overlap))
    return out
