"""Embedding config, vector-store fingerprints, LLM errors, RAG contracts.

No network, no keys, no model downloads. Tests that NEED the real
sentence-transformers weights are marked real_model and skipped here;
everything below runs locally with fakes and asserts the no-silent-fallback
behavior.
"""
import os

import pytest


def test_default_embedder_is_explicit_hashing(monkeypatch):
    from rag.embeddings.embedder import HashingEmbedder, get_embedder

    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    emb = get_embedder()
    assert isinstance(emb, HashingEmbedder)
    assert emb.provenance()["provider"] == "hashing"


def test_unknown_provider_raises_no_silent_fallback(monkeypatch):
    from rag.embeddings.embedder import EmbeddingConfigError, get_embedder

    monkeypatch.setenv("EMBEDDING_PROVIDER", "mystery-vectors")
    with pytest.raises(EmbeddingConfigError):
        get_embedder()


def test_st_provider_missing_package_error(monkeypatch):
    import builtins

    from rag.embeddings.embedder import EmbeddingConfigError, get_embedder

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EmbeddingConfigError, match="sentence-transformers.*not installed"):
        get_embedder()


def test_fingerprint_stable_and_config_sensitive():
    from rag.embeddings.embedder import HashingEmbedder

    a = HashingEmbedder(dim=128)
    assert a.config_fingerprint() == HashingEmbedder(dim=128).config_fingerprint()
    assert a.config_fingerprint() != HashingEmbedder(dim=256).config_fingerprint()


def test_store_fingerprints_and_reembed_gate(tmp_path):
    from rag.retrieval.vector_store import VectorStore

    store = VectorStore(persist_path=str(tmp_path / "vs.json"))
    assert store.fingerprints() == []
    assert store.fingerprint_ok("anything") is True  # empty store accepts any config
    store.add_documents([{"text": "t", "vector": [1.0, 0.0], "embedding_config": "fpA"}])
    assert store.fingerprints() == ["fpA"]
    assert store.fingerprint_ok("fpA") is True
    assert store.fingerprint_ok("fpB") is False  # config change -> re-ingest required
    # persistence roundtrip keeps fingerprints
    assert VectorStore(persist_path=str(tmp_path / "vs.json")).fingerprints() == ["fpA"]


def test_ingest_records_fingerprint_and_health_reports_it(monkeypatch):
    from fastapi.testclient import TestClient

    import rag.app as rag_app
    from rag.retrieval import retriever

    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setattr(retriever, "_store", None)
    c = TestClient(rag_app.app)
    r = c.post("/ingest", json={"documents": [
        {"text": "CD274 PD-L1 bispecific antibody trial", "title": "T",
         "source": "paper", "url": "http://x", "source_id": "PMID:1", "metadata": {}}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks"] >= 1 and body["embedding_config"]
    assert body["embedding_provenance"]["provider"] == "hashing"
    h = c.get("/health").json()
    assert h["documents"] >= 1 and body["embedding_config"] in h["fingerprints"]
    monkeypatch.setattr(retriever, "_store", None)


def test_ingest_rejects_config_change_with_409(monkeypatch):
    from fastapi.testclient import TestClient

    import rag.app as rag_app
    from rag.embeddings.embedder import HashingEmbedder
    from rag.retrieval import retriever

    monkeypatch.setattr(retriever, "_store", None)
    c = TestClient(rag_app.app)
    assert c.post("/ingest", json={"documents": [{"text": "hello world"}]}).status_code == 200

    class OtherDim(HashingEmbedder):
        def __init__(self):
            super().__init__(dim=256)

    import rag.embeddings.embedder as emb_mod

    monkeypatch.setattr(emb_mod, "get_embedder", lambda: OtherDim())
    r = c.post("/ingest", json={"documents": [{"text": "more text here"}]})
    assert r.status_code == 409
    monkeypatch.setattr(retriever, "_store", None)


def _llm_client():
    from rag.llm.llm_client import OpenAICompatibleClient

    return OpenAICompatibleClient("openai", "gpt-4o-mini", "test-key",
                                  "https://api.openai.com/v1", timeout=5.0)


def test_llm_timeout_mapped(monkeypatch):
    import httpx

    from rag.llm.llm_client import LLMError

    def boom(*a, **k):
        raise httpx.TimeoutException("too slow")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(LLMError) as e:
        _llm_client().generate("hello")
    assert e.value.code == "timeout"


def test_llm_http_codes_mapped(monkeypatch):
    import httpx

    from rag.llm.llm_client import LLMError

    class Resp:
        def __init__(self, code, payload=None):
            self.status_code = code
            self._payload = payload or {}

        def json(self):
            return self._payload

    cases = [(401, "auth_error"), (429, "rate_limit"), (500, "provider_error")]
    for code, want in cases:
        def fake_post(*a, _code=code, **k):
            return Resp(_code)

        monkeypatch.setattr(httpx, "post", fake_post)
        with pytest.raises(LLMError) as e:
            _llm_client().generate("hello")
        assert e.value.code == want, code


def test_llm_malformed_and_empty_rejected(monkeypatch):
    import httpx

    from rag.llm.llm_client import LLMError

    class Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp({"nope": 1}))
    with pytest.raises(LLMError) as e:
        _llm_client().generate("hello")
    assert e.value.code == "malformed_response"

    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **k: Resp({"choices": [{"message": {"content": "   "}}]}))
    with pytest.raises(LLMError) as e:
        _llm_client().generate("hello")
    assert e.value.code == "empty_response"

    with pytest.raises(LLMError) as e:
        _llm_client().generate("   ")
    assert e.value.code == "empty_response"


def test_mock_output_labelled_not_real():
    from rag.llm.llm_client import get_llm_client

    os.environ.pop("LLM_PROVIDER", None)
    out = get_llm_client().generate("TARGETS: A + B")
    assert out["model"] == "mock-llm"
    assert "mock" in out["text"].lower()


def test_generate_report_contract_fields():
    from rag.app import GenerateRequest

    fields = set(GenerateRequest.model_fields)
    assert {"target1", "target2", "prediction", "features",
            "feature_importance", "discretized_features"} <= fields
