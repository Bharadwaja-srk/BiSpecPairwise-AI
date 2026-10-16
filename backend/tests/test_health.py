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


def test_health_reports_connected_when_db_reachable(monkeypatch):
    """Health uses a valid SQLAlchemy expression: stubbed DB success -> healthy/connected."""
    from contextlib import asynccontextmanager

    import app.main as main_mod

    @asynccontextmanager
    async def _fake_session():
        class _Session:
            async def execute(self, *args, **kwargs):
                from sqlalchemy import text as _text

                assert str(args[0]) == str(_text("SELECT 1"))  # must be an executable clause
                return None

        yield _Session()

    monkeypatch.setattr(main_mod.db_manager, "session", _fake_session)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
    from app.main import create_app

    with TestClient(create_app()) as c:
        r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["database"] == "connected"


def test_analyze_returns_503_without_valid_features():
    """No store + no caller features -> structured 503 (never 500 NaN crash)."""
    import app.services.ml_service as mls

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
    from app.main import create_app

    mls._ml_service = mls.MLService("http://ml:8001", timeout=5.0)
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            r = c.post("/api/analyze", json={"target1": "CD274", "target2": "CTLA4"})
    finally:
        mls._ml_service = None
    assert r.status_code == 503
    body = r.json()
    assert body["status_code"] == 503
    assert "feature" in body["error"].lower()


def test_ranking_returns_503_without_valid_features():
    import app.services.ml_service as mls

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
    from app.main import create_app

    mls._ml_service = mls.MLService("http://ml:8001", timeout=5.0)
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            r = c.get("/api/ranking", params={"candidate": "CD274:CTLA4", "top_k": 2})
    finally:
        mls._ml_service = None
    assert r.status_code == 503
    assert r.json()["status_code"] == 503
