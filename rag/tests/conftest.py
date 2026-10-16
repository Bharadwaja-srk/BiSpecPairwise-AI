"""Isolate RAG tests from deployment persistence (VECTOR_STORE_PATH).

Production deployments mount a persistent store; tests must never read or
write it. This fixture points persistence at a per-test tmp dir and resets
the retriever singleton before and after every test.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_vector_store(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "test_store.json"))
    from rag.retrieval import retriever

    monkeypatch.setattr(retriever, "_store", None)
    yield
    monkeypatch.setattr(retriever, "_store", None)
