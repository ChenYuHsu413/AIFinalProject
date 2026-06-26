"""API-level tests for the FastAPI backend (app.backend.main)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_figures_served():
    """A pre-rendered figure is served as an image under /figures/{name}."""
    resp = client.get("/figures/confusion_matrix.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    assert len(resp.content) > 0


def test_figures_missing_returns_404():
    resp = client.get("/figures/does_not_exist.png")
    assert resp.status_code == 404
