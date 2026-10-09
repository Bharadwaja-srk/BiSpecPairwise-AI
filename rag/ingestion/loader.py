"""Document ingestion: load biomedical docs, preserve provenance metadata."""

from __future__ import annotations

import os
from typing import Any


def load_text_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return {
        "text": text,
        "title": os.path.basename(path),
        "source": "local-file",
        "url": None,
        "source_id": None,
        "metadata": {"path": path},
    }


def load_documents(paths: list[str]) -> list[dict[str, Any]]:
    docs = []
    for p in paths:
        if os.path.isfile(p):
            docs.append(load_text_file(p))
    return docs


def paper_to_document(title: str, abstract: str, source_id: str | None = None, url: str | None = None) -> dict[str, Any]:
    return {
        "text": f"{title}\n\n{abstract}",
        "title": title,
        "source": "paper",
        "url": url,
        "source_id": source_id,
        "metadata": {"source_id": source_id, "url": url},
    }
