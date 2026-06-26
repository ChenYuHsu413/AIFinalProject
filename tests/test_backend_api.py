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


def test_xjtu_generalization():
    resp = client.get("/xjtu/generalization")
    assert resp.status_code == 200
    assert {"per_bearing", "aggregate"} <= set(resp.json())


def test_xjtu_lobo_loco():
    resp = client.get("/xjtu/lobo_loco")
    assert resp.status_code == 200
    body = resp.json()
    assert {"lobo", "loco"} == set(body)
    assert "pooled" in body["lobo"]
    assert "pooled" in body["loco"]


def test_xjtu_domain_adapt():
    resp = client.get("/xjtu/domain_adapt")
    assert resp.status_code == 200
    assert {"results", "summary"} <= set(resp.json())


def test_ims_metrics():
    resp = client.get("/ims/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "metrics" in body
    assert "lead_time_days" in body


def test_ims_health_curve():
    resp = client.get("/ims/health_curve")
    assert resp.status_code == 200
    points = resp.json()
    assert isinstance(points, list) and len(points) > 0
    assert set(points[0]) == {
        "timestamp", "rul_true", "rul_pred", "health", "is_degrading",
    }
    # Pre-degradation rul_pred is NaN in the CSV -> must serialize as JSON null.
    assert any(p["rul_pred"] is None for p in points)


def test_knowledge_documents():
    resp = client.get("/knowledge/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list) and len(docs) > 0
    assert {"source", "title", "preview", "chars"} <= set(docs[0])


def test_knowledge_search():
    resp = client.get("/knowledge/search", params={"q": "過熱 溫度 警報", "top_k": 2})
    assert resp.status_code == 200
    hits = resp.json()
    assert isinstance(hits, list) and 0 < len(hits) <= 2
    assert {"text", "score", "source", "title", "topic"} <= set(hits[0])


def test_knowledge_search_requires_q():
    resp = client.get("/knowledge/search")
    assert resp.status_code == 422
