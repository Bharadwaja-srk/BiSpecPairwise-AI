"""RAG service tests: discretization, retrieval, prompt, report, mock LLM (no network/keys)."""

import pytest


def test_discretize_table5_spot_checks():
    from rag.discretization.feature_discretizer import discretize

    assert discretize("safety_score", 0.0012) == "safe"
    assert discretize("safety_score", -0.5) == "unsafe"
    assert discretize("pearson_correlation", 0.5) == "strongly positive correlation"
    assert discretize("double_positive", 0.0) == "low"
    assert discretize("unknown_feature_xyz", 5) == "5"  # never arbitrary labels


def test_chunking_preserves_metadata():
    from rag.ingestion.chunking import chunk_document
    from rag.ingestion.loader import paper_to_document

    doc = paper_to_document("T", "w1 w2 w3 w4 w5 w6 w7 w8", source_id="PMID:1", url="http://x")
    chunks = chunk_document(doc, chunk_size=4, overlap=1)
    assert len(chunks) >= 2
    assert all(c["title"] == "T" and c["metadata"]["source_id"] == "PMID:1" for c in chunks)


def test_vector_store_and_retriever():
    from rag.embeddings.embedder import HashingEmbedder
    from rag.retrieval import retriever
    from rag.retrieval.vector_store import VectorStore

    store = VectorStore()
    assert retriever.retrieve("A", "B", store=store) == []  # empty -> [] never fabricated
    emb = HashingEmbedder()
    docs = [
        {"text": "CD274 PD-L1 bispecific antibody clinical trial", "title": "P1", "source": "paper",
         "metadata": {}, "vector": emb.embed("CD274 PD-L1 bispecific antibody clinical trial")},
        {"text": "unrelated cooking recipe soup", "title": "P2", "source": "blog",
         "metadata": {}, "vector": emb.embed("unrelated cooking recipe soup")},
    ]
    store.add_documents(docs)
    assert store.count() == 2
    hits = retriever.retrieve("CD274", "PD-L1", top_k=1, embedder=emb, store=store)
    assert hits[0]["title"] == "P1" and "score" in hits[0]


def test_prompt_separates_concerns():
    from rag.prompts.prompt_builder import build_prompt

    p = build_prompt("CD274", "CTLA4", {"prediction_score": 0.9, "rank": 4, "model": "m"},
                     {"safety_score": 0.0012}, {"safety_score": 1.0}, [])
    for token in ["CD274", "CTLA4", "ML PREDICTION", "BIOLOGICAL FEATURES", "IMPORTANT FEATURES",
                  "RETRIEVED BIOMEDICAL EVIDENCE", "Uncertainty", "no retrieved"]:
        assert token in p or token.lower() in p.lower()


def test_report_validation_and_mock_llm():
    from rag.llm.llm_client import get_llm_client
    from rag.reports.report_generator import build_report, validate_report

    llm = get_llm_client()  # mock by default (LLM_PROVIDER unset)
    out = llm.generate("TARGETS: CD274 + CTLA4\n## summary\nx")
    assert "summary" in out["text"].lower()
    rep = build_report("CD274", "CTLA4", {"prediction_score": 0.9, "rank": 4},
                       out["text"], [], out["model"])
    assert validate_report(rep)["confidence"] in ("high", "medium", "low")
    with pytest.raises(ValueError):
        validate_report({"summary": "x"})  # missing keys rejected
