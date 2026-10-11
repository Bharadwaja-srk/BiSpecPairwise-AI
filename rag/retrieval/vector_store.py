"""Replaceable local vector store: add_documents / search / delete / count."""

from __future__ import annotations

import json
import math
import os
import uuid


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorStore:
    """In-memory store with optional JSON persistence. Swap for pgvector/Qdrant later."""

    def __init__(self, persist_path: str | None = None):
        self.persist_path = persist_path or os.getenv("VECTOR_STORE_PATH", "")
        self._docs: dict[str, dict] = {}
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path) as f:
                    for d in json.load(f):
                        self._docs[d["id"]] = d
            except Exception:
                pass

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(list(self._docs.values()), f)
        except Exception:
            pass

    def add_documents(self, docs: list[dict]) -> list[str]:
        ids = []
        for d in docs:
            doc_id = d.get("id") or str(uuid.uuid4())
            self._docs[doc_id] = {**d, "id": doc_id}
            ids.append(doc_id)
        self._save()
        return ids

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        scored = [
            {**d, "score": _cosine(query_vector, d.get("vector", []))}
            for d in self._docs.values()
            if d.get("vector")
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:top_k]

    def delete(self, doc_id: str) -> bool:
        removed = self._docs.pop(doc_id, None) is not None
        if removed:
            self._save()
        return removed

    def count(self) -> int:
        return len(self._docs)

    def fingerprints(self) -> list[str]:
        """Distinct embedding-config fingerprints present in the store.

        More than one fingerprint (or a fingerprint that differs from the
        active embedder's) means the index mixes embedding spaces: clear and
        re-ingest. Never compare vectors across fingerprints.
        """
        return sorted({str(d.get("embedding_config", "unknown"))
                       for d in self._docs.values()})

    def fingerprint_ok(self, fingerprint: str) -> bool:
        present = self.fingerprints()
        return not present or present == [fingerprint]

    def clear(self) -> None:
        self._docs.clear()
        self._save()
