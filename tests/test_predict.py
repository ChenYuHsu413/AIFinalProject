"""Tests for the prediction helpers (rules, schema, risk bands).

These do not require a trained model — they target the pure-function rules.
"""
from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import add_engineered_features
from src.models.predict import (
    REQUIRED_INPUT_COLUMNS,
    _maintenance_advice,
    _result_for_row,
    _risk_level,
)


def _row(**overrides) -> pd.Series:
    base = {
        "Type": "L",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 50.0,
    }
    base.update(overrides)
    df = add_engineered_features(pd.DataFrame([base]))
    return df.iloc[0]


def test_risk_bands_use_configured_thresholds():
    assert _risk_level(0.10) == "Low"
    assert _risk_level(0.50) == "Medium"
    assert _risk_level(0.80) == "High"


def test_required_columns_match_schema():
    assert REQUIRED_INPUT_COLUMNS == [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]


def test_advice_flags_high_torque_and_high_wear():
    row = _row(**{"Torque [Nm]": 70.0, "Tool wear [min]": 240.0})
    advice = _maintenance_advice(row, prob=0.85)
    assert any("扭矩" in a for a in advice)
    assert any("刀具磨耗" in a for a in advice)
    assert any("故障機率偏高" in a for a in advice)


def test_healthy_envelope_returns_routine_advice():
    row = _row()
    advice = _maintenance_advice(row, prob=0.05)
    assert any("健康" in a or "例行" in a for a in advice)


def test_result_envelope_is_well_formed():
    row = _row()
    res = _result_for_row(row, prob=0.92)
    assert res["predicted_class"] == 1
    assert res["risk_level"] == "High"
    assert 0.0 <= res["health_score"] <= 100.0
    assert res["health_score"] == 8.0
    assert isinstance(res["maintenance_advice"], list)
