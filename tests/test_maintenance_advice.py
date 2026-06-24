"""Tests for the maintenance-advice decision layer (E2).

All tests run on plain scalars — no dataset or model required.
"""
from __future__ import annotations

import pytest

from src.models.maintenance_advice import (
    RISK_GREEN,
    RISK_RED,
    RISK_YELLOW,
    maintenance_advice,
)


def test_green_when_healthy_and_pre_fpt():
    adv = maintenance_advice(health=90.0, rul_hours=None, past_fpt=False)
    assert adv.risk == RISK_GREEN
    assert adv.suggested_window_hours is None
    assert adv.cost_note is None


def test_yellow_when_past_fpt_above_alarm():
    adv = maintenance_advice(health=60.0, rul_hours=10.0, past_fpt=True,
                             alarm_health=30.0, safety_margin=0.3)
    assert adv.risk == RISK_YELLOW
    assert adv.suggested_window_hours == pytest.approx(7.0)


def test_red_when_health_below_alarm():
    adv = maintenance_advice(health=20.0, rul_hours=2.0, past_fpt=True,
                             alarm_health=30.0, safety_margin=0.3)
    assert adv.risk == RISK_RED
    assert adv.suggested_window_hours == pytest.approx(1.4)


def test_alarm_boundary_is_inclusive_red():
    # health exactly at the alarm threshold counts as red
    assert maintenance_advice(30.0, 5.0, True, alarm_health=30.0).risk == RISK_RED
    # just above the threshold, with FPT, is yellow
    assert maintenance_advice(30.01, 5.0, True, alarm_health=30.0).risk == RISK_YELLOW


def test_window_uses_safety_margin():
    assert maintenance_advice(50.0, 10.0, True, safety_margin=0.0
                              ).suggested_window_hours == pytest.approx(10.0)
    assert maintenance_advice(50.0, 10.0, True, safety_margin=0.5
                              ).suggested_window_hours == pytest.approx(5.0)


def test_no_window_when_rul_unavailable():
    adv = maintenance_advice(50.0, None, True)
    assert adv.suggested_window_hours is None
    # past FPT but no RUL -> rationale should flag that RUL is not reliable yet
    assert any("尚不可靠" in r for r in adv.rationale)


def test_no_window_when_rul_nonpositive():
    assert maintenance_advice(50.0, 0.0, True).suggested_window_hours is None


def test_cost_note_only_when_both_costs_and_window():
    with_cost = maintenance_advice(50.0, 10.0, True,
                                   cost_unplanned=10000, cost_planned=2000)
    assert with_cost.cost_note is not None
    assert "20%" in with_cost.cost_note  # 2000 / 10000

    # no window -> no cost note even if costs given
    no_window = maintenance_advice(50.0, None, True,
                                   cost_unplanned=10000, cost_planned=2000)
    assert no_window.cost_note is None


def test_invalid_safety_margin_raises():
    with pytest.raises(ValueError):
        maintenance_advice(50.0, 10.0, True, safety_margin=1.0)
