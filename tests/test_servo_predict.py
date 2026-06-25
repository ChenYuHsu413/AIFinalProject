"""Module Servo — structured prediction, simulator, RAG, LLM fallback."""
import pandas as pd
import pytest

from src.data.build_servo_dataset import generate_placeholder
from src.models import servo_simulator as sim


def test_simulator_classification_runs():
    df = generate_placeholder(60)
    res = sim.run_classification(df, "engineered", "random_forest", 150)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert res["train_time_s"] >= 0
    assert len(res["confusion_matrix"]) == len(res["labels"])


def test_simulator_regression_runs():
    df = generate_placeholder(60)
    res = sim.run_regression(df, "full", "ridge", 150)
    assert res["mae"] >= 0 and res["rmse"] >= 0
    assert "r2" in res


def test_simulator_algorithms_registered():
    assert set(sim.CLASSIFIER_NAMES) >= {
        "logistic_regression", "decision_tree", "random_forest",
        "gradient_boosting", "mlp"}
    assert "ridge" in sim.REGRESSOR_NAMES


def test_predict_servo_structured_output():
    pytest.importorskip("joblib")
    try:
        from src.models.servo_predict import load_servo_models, predict_servo
        b = load_servo_models()
    except FileNotFoundError:
        pytest.skip("Servo reference model not built")
    row = {c: 0.0 for c in b.feature_columns}
    out = predict_servo(row)
    assert out["predicted_health_state"] in {"LN", "LO", "MED", "HI"}
    assert 0.0 <= out["degradation_score"] <= 1.0
    assert out["risk_level"] in {"Low", "Medium", "High"}
    assert isinstance(out["maintenance_advice"], list) and out["maintenance_advice"]
    assert len(out["top_features"]) >= 1


def test_rag_search_and_chunking():
    from src.knowledge.cleaner import chunk_text, clean_text
    from src.knowledge.maintenance_rag import search

    assert chunk_text(clean_text("a\n\nb\n\nc"), chunk_chars=2, overlap=0)
    hits = search("位置誤差 卡滯", top_k=3)
    # offline seed docs are committed, so we expect at least one hit
    assert isinstance(hits, list)


def test_llm_assistant_fallback():
    """Without ANTHROPIC_API_KEY the assistant must still return text."""
    import os

    from src.llm.maintenance_assistant import generate_report

    if os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("API key present; fallback path not exercised")
    pred = {
        "predicted_health_state": "MED", "health_state_zh": "中度退化",
        "degradation_score": 0.6, "health_score": 40.0, "risk_level": "Medium",
        "model_confidence": 0.7,
        "top_features": [{"feature": "position_error_mean", "z": 3.1,
                          "hint": "位置誤差平均偏高"}],
        "placeholder": True,
    }
    rep = generate_report(pred, None)
    assert rep["source"] == "fallback"
    assert "維修報告摘要" in rep["text"]
