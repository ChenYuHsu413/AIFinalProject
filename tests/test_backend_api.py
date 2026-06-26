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


def test_xjtu_rul_predictions():
    resp = client.get("/xjtu/rul_predictions")
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list) and len(rows) > 0
    assert {"condition", "bearing", "rul_pred", "health"} <= set(rows[0])


def test_xjtu_health_overlay():
    resp = client.get("/xjtu/health_overlay")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert len(body["curves"]) > 0
    c0 = body["curves"][0]
    assert len(c0["life_pct"]) == len(c0["hi"]) <= 200
    assert len(body["fpt_markers"]) == len(body["curves"])


def test_xjtu_replay():
    resp = client.get("/xjtu/replay/35Hz12kN/Bearing1_1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert 0 < len(body["frames"]) <= 100
    f = body["frames"][0]
    assert {"k", "minute", "health", "risk", "x", "y"} <= set(f)
    assert len(f["x"]) == len(f["y"])


def test_xjtu_replay_not_found():
    resp = client.get("/xjtu/replay/nope/nope")
    assert resp.status_code == 404


def test_maintenance_advice_healthy():
    resp = client.post("/maintenance/advice",
                       json={"health": 95.0, "rul_hours": None, "past_fpt": False})
    assert resp.status_code == 200
    body = resp.json()
    assert {"risk", "risk_label_zh", "suggested_window_hours", "rationale"} <= set(body)
    assert body["risk"] == "green"


def test_maintenance_advice_window_from_rul():
    resp = client.post("/maintenance/advice",
                       json={"health": 40.0, "rul_hours": 10.0, "past_fpt": True,
                             "safety_margin": 0.3})
    assert resp.status_code == 200
    # suggested window = rul * (1 - safety_margin) = 10 * 0.7 = 7.0
    assert resp.json()["suggested_window_hours"] == 7.0


def test_maintenance_advice_requires_fields():
    resp = client.post("/maintenance/advice", json={"health": 50.0})
    assert resp.status_code == 422


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


def test_ims_health_indicator_default():
    resp = client.get("/ims/health_indicator")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["indicator"] == "b1_rms"
    assert len(body["candidates"]) > 0
    assert isinstance(body["fpt_index"], int)
    assert len(body["points"]) > 0
    assert {"timestamp", "health"} <= set(body["points"][0])


def test_ims_health_indicator_switch():
    resp = client.get("/ims/health_indicator", params={"indicator": "b1_kurtosis"})
    assert resp.status_code == 200
    assert resp.json()["indicator"] == "b1_kurtosis"


def test_ims_health_indicator_unknown():
    resp = client.get("/ims/health_indicator", params={"indicator": "nope"})
    assert resp.status_code == 400


def test_ims_snapshot():
    resp = client.get("/ims/snapshot/0")
    assert resp.status_code == 200
    body = resp.json()
    if body["available"]:
        assert len(body["waveform"]) > 0
        assert {"freqs", "mags"} == set(body["spectrum"])
        assert len(body["spectrum"]["freqs"]) == len(body["spectrum"]["mags"])
        assert max(body["spectrum"]["freqs"]) <= 2000.0
    else:
        assert "reason" in body  # raw IMS data absent (e.g. cloud / CI)


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


def test_servo_glossary():
    resp = client.get("/servo/glossary")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list) and len(docs) > 0
    assert {"name", "zh", "desc", "meaning", "anomaly"} <= set(docs[0])


def test_servo_feature_sets():
    resp = client.get("/servo/feature_sets")
    assert resp.status_code == 200
    sets = resp.json()
    assert "engineered" in sets
    assert {"label", "desc", "columns"} <= set(sets["engineered"])


def test_servo_samples():
    resp = client.get("/servo/samples")
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list) and len(rows) > 0
    assert {"run_index", "ylabel", "DV"} <= set(rows[0])


def test_servo_reference_metrics():
    resp = client.get("/servo/reference_metrics")
    assert resp.status_code == 200
    assert set(resp.json()) == {"clf", "reg", "dl"}


def test_servo_simulate_options():
    resp = client.get("/servo/simulate/options")
    assert resp.status_code == 200
    body = resp.json()
    assert {"classifiers", "regressors", "algo_labels"} <= set(body)
    assert len(body["classifiers"]) > 0


def test_servo_simulate_classification():
    resp = client.post("/servo/simulate", json={
        "task": "clf", "feature_set": "engineered", "algo": "random_forest", "n": 400,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert {"macro_f1", "accuracy", "confusion_matrix", "explanation"} <= set(body)
    assert isinstance(body["explanation"], list)


def test_servo_simulate_regression():
    resp = client.post("/servo/simulate", json={
        "task": "reg", "feature_set": "engineered", "algo": "random_forest", "n": 400,
    })
    assert resp.status_code == 200
    assert {"r2", "mae", "rmse", "explanation"} <= set(resp.json())


def test_servo_simulate_bad_algo():
    resp = client.post("/servo/simulate", json={
        "task": "clf", "feature_set": "engineered", "algo": "nope", "n": 400,
    })
    assert resp.status_code == 400


def test_servo_simulate_bad_task():
    resp = client.post("/servo/simulate", json={
        "task": "xxx", "feature_set": "engineered", "algo": "random_forest", "n": 400,
    })
    assert resp.status_code == 422


_SAMPLE_RECORD = {
    "type": "L",
    "air_temperature_K": 298.1,
    "process_temperature_K": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_Nm": 42.8,
    "tool_wear_min": 108,
}


def test_predict_batch():
    body = [
        _SAMPLE_RECORD,
        {**_SAMPLE_RECORD, "torque_Nm": 70.0, "tool_wear_min": 240},
    ]
    resp = client.post("/predict/batch", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert {"failure_probability", "risk_level"} <= set(data["results"][0])


def test_predict_batch_empty():
    resp = client.post("/predict/batch", json=[])
    assert resp.status_code == 200
    assert resp.json() == {"count": 0, "results": []}


def test_predict_batch_validation_error():
    resp = client.post("/predict/batch", json=[{"type": "X"}])
    assert resp.status_code == 422


def test_predict_explain():
    resp = client.post("/predict/explain", json=_SAMPLE_RECORD)
    assert resp.status_code == 200
    body = resp.json()
    assert "supported" in body
    if body["supported"]:
        assert len(body["feature_names"]) == len(body["shap_values"])
        assert len(body["feature_values"]) == len(body["shap_values"])
        assert isinstance(body["base_value"], (int, float))


def test_metrics_test_predictions():
    resp = client.get("/metrics/test_predictions")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"y_true", "y_proba"}
    assert len(body["y_true"]) == len(body["y_proba"]) > 0
    assert set(body["y_true"]) <= {0, 1}
