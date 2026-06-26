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


def test_metrics_summary_modules():
    """Each known module returns the expected metric sub-objects."""
    expected_keys = {
        "servo": {"clf", "reg"},
        "B": {"ims_rul"},
        "Bplus": {"generalization", "lobo", "loco"},
        "C": {"paderborn"},
    }
    for module, keys in expected_keys.items():
        resp = client.get("/metrics/summary", params={"module": module})
        assert resp.status_code == 200, module
        assert set(resp.json()) == keys, module


def test_metrics_summary_unknown_module_returns_400():
    resp = client.get("/metrics/summary", params={"module": "nope"})
    assert resp.status_code == 400


def test_metrics_summary_requires_module():
    resp = client.get("/metrics/summary")
    assert resp.status_code == 422


def test_paderborn_eval():
    resp = client.get("/paderborn/eval")
    assert resp.status_code == 200
    body = resp.json()
    assert {"method", "features", "results", "summary"} <= set(body)
    assert {"baseline", "artificial_to_real"} <= set(body["results"])
