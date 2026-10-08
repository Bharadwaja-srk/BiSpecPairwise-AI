"""Backend health + validation tests (no DB, no network required)."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("ENABLE_MOCK_ML", "true")
os.environ.setdefault("ENABLE_MOCK_RAG", "true")


@pytest.fixture()
def client():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_degraded_without_db(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("healthy", "degraded")
    assert "database" in body


def test_analyze_rejects_same_gene(client):
    r = client.post("/api/analyze", json={"target1": "CD274", "target2": "CD274"})
    assert r.status_code == 422


def test_analyze_rejects_empty(client):
    r = client.post("/api/analyze", json={"target1": "", "target2": "CTLA4"})
    assert r.status_code == 422
