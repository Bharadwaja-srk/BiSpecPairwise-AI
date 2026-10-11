"""RAG HTTP service (FastAPI) on :8002. Backend calls http://rag:8002 in compose."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="BSPAI RAG service")


class GenerateRequest(BaseModel):
    target1: str
    target2: str
    prediction: dict = {}
    features: dict = {}
    feature_importance: dict = {}
    discretized_features: dict = {}


class RetrieveRequest(BaseModel):
    target1: str
    target2: str
    context: str = ""
    top_k: int = 5


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": _count(), "fingerprints": _fingerprints()}


def _fingerprints() -> list[str]:
    try:
        from rag.retrieval.retriever import get_store

        return get_store().fingerprints()
    except Exception:
        return []


def _count() -> int:
    try:
        from rag.retrieval.retriever import get_store

        return get_store().count()
    except Exception:
        return 0


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    from rag.retrieval.retriever import retrieve as _retrieve

    return {"results": _retrieve(req.target1.upper(), req.target2.upper(), req.context, top_k=req.top_k)}


@app.post("/ingest")
def ingest(payload: dict) -> dict:
    """Ingest documents: {documents:[{text,title,source,url,source_id,metadata}], chunk_size}."""
    from rag.embeddings.embedder import get_embedder
    from rag.ingestion.chunking import chunk_documents
    from rag.retrieval.retriever import get_store

    docs = payload.get("documents", [])
    chunk_size = int(payload.get("chunk_size", 800))
    chunks = chunk_documents(docs, chunk_size=chunk_size)
    embedder = get_embedder()
    fingerprint = embedder.config_fingerprint()
    store = get_store()
    if not store.fingerprint_ok(fingerprint):
        raise HTTPException(
            409, f"embedding config changed (active={fingerprint}, "
                 f"stored={store.fingerprints()}). Clear and re-ingest; "
                 "vectors across configs are not comparable.")
    for c in chunks:
        c["vector"] = embedder.embed(c.get("text", ""))
        c["embedding_config"] = fingerprint
        c["embedding_provenance"] = embedder.provenance()
    ids = store.add_documents(chunks)
    return {"chunks": len(ids), "documents": len(docs),
            "embedding_config": fingerprint,
            "embedding_provenance": embedder.provenance()}


@app.post("/generate-report")
def generate_report(req: GenerateRequest) -> dict:
    from rag.discretization.feature_discretizer import discretize_all
    from rag.llm.llm_client import LLMError, get_llm_client
    from rag.prompts.prompt_builder import build_prompt
    from rag.reports.report_generator import build_report
    from rag.retrieval.retriever import retrieve as _retrieve

    t1, t2 = req.target1.upper(), req.target2.upper()
    features = {k: float(v) for k, v in (req.features or {}).items()}
    importance = {k: float(v) for k, v in (req.feature_importance or {}).items()}
    discretized = dict(req.discretized_features or {}) or discretize_all(features)
    evidence = _retrieve(t1, t2, top_k=5)
    prompt = build_prompt(t1, t2, req.prediction, features, importance, evidence, discretized)
    try:
        llm = get_llm_client()
        out = llm.generate(prompt)
    except LLMError as e:
        raise HTTPException(502, f"{e.code}: {e}") from e
    except Exception as e:
        raise HTTPException(502, f"llm_error: {type(e).__name__}") from e
    try:
        return build_report(t1, t2, req.prediction, out["text"], evidence, out.get("model", "unknown"))
    except ValueError as e:
        raise HTTPException(502, f"report_validation: {e}") from e
