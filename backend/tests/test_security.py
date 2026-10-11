"""API-key boundary tests: open mode, enforcement, health exemption (no network)."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture()
def client_open():
    os.environ.pop("BSPAI_API_KEYS", None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture()
def client_locked():
    os.environ["BSPAI_API_KEYS"] = "secret-one,secret-two"
    from app.core.config import get_settings
    get_settings.cache_clear()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    os.environ.pop("BSPAI_API_KEYS", None)
    get_settings.cache_clear()


def test_open_mode_allows_analysis(client_open):
    r = client_open.post("/api/analyze", json={"target1": "CD274", "target2": "CTLA4"})
    assert r.status_code in (200, 502, 503)  # open mode: never 401


def test_locked_mode_rejects_missing_key(client_locked):
    r = client_locked.post("/api/analyze", json={"target1": "CD274", "target2": "CTLA4"})
    assert r.status_code == 401
    assert "API key" in r.json()["error"]


def test_locked_mode_rejects_wrong_key(client_locked):
    r = client_locked.get("/api/targets", headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_locked_mode_accepts_valid_key(client_locked):
    r = client_locked.get("/api/targets", headers={"X-API-Key": "secret-two"})
    assert r.status_code in (200, 503)  # auth passes; DB may be down in tests


def test_health_exempt_from_auth(client_locked):
    assert client_locked.get("/api/health").status_code == 200
