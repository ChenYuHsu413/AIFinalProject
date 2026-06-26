"""Phase 1.5 verification gate — cross-endpoint integration flows.

Unlike test_backend_api.py (each endpoint in isolation), these exercise the
realistic page flows a frontend will chain, proving the *contract links* between
endpoints hold (output of one is directly consumable by the next) and that
endpoints reading the same artifact stay consistent.
"""
from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def _force_llm_fallback(monkeypatch):
    """Keep the LLM assistant offline/deterministic (local .env may hold keys)."""
    import src.llm.maintenance_assistant as ma

    monkeypatch.setattr(ma, "_call_llm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))


# --- Servo dashboard flow: samples -> predict -> assistant -------------------
def test_servo_dashboard_to_assistant_flow(monkeypatch):
    """A real /servo/predict output must be directly consumable by the assistant
    (this is the contract link that previously broke on the top_features 'z' key)."""
    cols = client.get("/servo/model_info").json()["feature_columns"]
    sample = client.get("/servo/samples").json()[0]
    features = {c: sample[c] for c in cols}

    pred = client.post("/servo/predict", json={"features": features})
    assert pred.status_code == 200
    prediction = pred.json()
    assert {"predicted_health_state", "degradation_score", "top_features"} <= set(prediction)

    _force_llm_fallback(monkeypatch)
    report = client.post("/servo/assistant/report", json={"prediction": prediction})
    assert report.status_code == 200 and len(report.json()["text"]) > 0

    qa = client.post("/servo/assistant/qa",
                    json={"question": "需要立刻停機嗎？", "prediction": prediction})
    assert qa.status_code == 200 and len(qa.json()["text"]) > 0


# --- Module A flow: predict_full -> explain + threshold-tuner data -----------
def test_module_a_flow():
    record = {
        "type": "L", "air_temperature_K": 298.1, "process_temperature_K": 308.6,
        "rotational_speed_rpm": 1551, "torque_Nm": 42.8, "tool_wear_min": 108,
    }
    full = client.post("/predict_full", json=record).json()
    assert {"failure_probability", "failure_type_probabilities"} <= set(full)

    explain = client.post("/predict/explain", json=record).json()
    assert "supported" in explain

    # threshold tuner needs the test-set scores
    tp = client.get("/metrics/test_predictions").json()
    assert len(tp["y_true"]) == len(tp["y_proba"]) > 0


# --- Module B consistency: metrics fpt_index == health_indicator fpt_index ---
def test_module_b_fpt_consistency():
    meta = client.get("/ims/metrics").json()
    hi = client.get("/ims/health_indicator").json()  # default indicator = b1_rms
    assert hi["indicator"] == "b1_rms"
    # both compute FPT from the same b1_rms indicator -> must agree
    assert meta["fpt_index"] == hi["fpt_index"]


# --- Module B+ flow: rul_predictions row -> maintenance/advice ---------------
def test_module_bplus_advice_flow():
    rows = client.get("/xjtu/rul_predictions").json()
    assert len(rows) > 0
    row = rows[len(rows) // 2]
    rul = row["rul_pred"]  # may be null pre-FPT
    advice = client.post("/maintenance/advice", json={
        "health": float(row["health"]),
        "rul_hours": rul,
        "past_fpt": bool(row["is_degrading"]),
    })
    assert advice.status_code == 200
    assert advice.json()["risk"] in {"green", "yellow", "red"}


# --- Module C consistency: paderborn/eval summary == metrics/summary?module=C
def test_module_c_summary_consistency():
    full = client.get("/paderborn/eval").json()["summary"]
    via_summary = client.get("/metrics/summary", params={"module": "C"}).json()["paderborn"]["summary"]
    assert full == via_summary


# --- Module B+ KPI consistency: generalization aggregate n_bearings ----------
def test_module_bplus_bearing_count_consistency():
    gen = client.get("/xjtu/generalization").json()
    n_per_bearing = len(gen["per_bearing"])
    n_aggregate = gen["aggregate"]["n_bearings"]
    assert n_per_bearing == n_aggregate
    # overlay should render one curve per bearing
    overlay = client.get("/xjtu/health_overlay").json()
    assert len(overlay["curves"]) == n_per_bearing
