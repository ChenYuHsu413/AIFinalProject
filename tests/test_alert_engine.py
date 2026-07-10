"""S2 — alert engine hysteresis, event streams, and LLM-failure isolation."""
from __future__ import annotations

import json

import pytest

from src.monitor import alert_engine as ae


def _pred(risk: str, cw=None, state=None, t=0.0):
    state = state or {"Low": "LN", "Medium": "MED", "High": "HI"}[risk]
    return {
        "predicted_health_state": state, "risk_level": risk,
        "consistency_warning": cw, "degradation_score": 0.5, "health_score": 50.0,
        "model_confidence": 0.9, "health_state_proba": {"LN": .1, "HI": .9},
        "top_features": [{"feature": "torque_std", "z": 3.1, "hint": "扭矩波動"}],
        "_true": state, "_rows": 5000, "_stream_t": t,
    }


def _engine(tmp_path, monkeypatch, **over):
    cfg = {"high_consecutive": 3, "clear_consecutive": 3, "model_version": "v1",
           "alerts_dir": str(tmp_path), **over}
    # Fast, offline work-order stub so tests never touch the network.
    monkeypatch.setattr(ae, "generate_report",
                        lambda pred: {"text": "工單草稿", "source": "fallback"})
    events = []
    return ae.AlertEngine(on_event=events.append, config=cfg), events


def _feed(engine, risks):
    for i, r in enumerate(risks):
        engine.update(_pred(r, t=float(i)))


def test_triggers_only_after_N_consecutive_high(tmp_path, monkeypatch):
    engine, events = _engine(tmp_path, monkeypatch)
    _feed(engine, ["High", "High"])            # 2 < N=3
    assert not any(e["type"] == "alert_triggered" for e in events)
    engine.update(_pred("High", t=2.0))        # 3rd consecutive -> trigger
    assert engine.active
    trig = [e for e in events if e["type"] == "alert_triggered"]
    assert len(trig) == 1
    assert trig[0]["model_version"] == "v1"
    assert trig[0]["snapshot"]["degradation_score"] == 0.5


def test_high_streak_broken_resets(tmp_path, monkeypatch):
    engine, events = _engine(tmp_path, monkeypatch)
    _feed(engine, ["High", "High", "Medium", "High", "High"])  # never 3 in a row
    assert not engine.active
    assert not any(e["type"] == "alert_triggered" for e in events)


def test_no_retrigger_while_active_then_clears(tmp_path, monkeypatch):
    engine, events = _engine(tmp_path, monkeypatch)
    _feed(engine, ["High"] * 5)                 # trigger once, stay active
    assert sum(e["type"] == "alert_triggered" for e in events) == 1
    _feed(engine, ["Medium", "Low"])            # 2 < M=3, still active
    assert engine.active
    engine.update(_pred("Low", t=99))           # 3rd -> clear
    assert not engine.active
    assert sum(e["type"] == "alert_cleared" for e in events) == 1


def test_consistency_event_is_independent(tmp_path, monkeypatch):
    engine, events = _engine(tmp_path, monkeypatch)
    engine.update(_pred("Low", cw="分類器與 DV 風險不一致"))
    cw = [e for e in events if e["type"] == "consistency_warning"]
    assert len(cw) == 1 and not engine.active           # did NOT trigger main alert
    assert cw[0]["message"] == "分類器與 DV 風險不一致"


def test_trigger_writes_jsonl_and_work_order(tmp_path, monkeypatch):
    engine, _ = _engine(tmp_path, monkeypatch)
    _feed(engine, ["High"] * 3)
    engine.join_pending()
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines()]
    types = [r["type"] for r in rows]
    assert "alert_triggered" in types and "work_order_draft" in types
    wo = next(r for r in rows if r["type"] == "work_order_draft")
    assert wo["llm_source"] == "fallback" and wo["text"]
    assert wo["alert_id"] == next(r for r in rows if r["type"] == "alert_triggered")["id"]


def test_llm_failure_does_not_block_alert(tmp_path, monkeypatch):
    engine, events = _engine(tmp_path, monkeypatch)

    def _boom(pred):
        raise RuntimeError("all providers down")
    monkeypatch.setattr(ae, "generate_report", _boom)

    _feed(engine, ["High"] * 3)
    # The alert itself was emitted synchronously, before the work-order thread.
    assert any(e["type"] == "alert_triggered" for e in events)
    engine.join_pending()
    rows = [json.loads(l) for f in tmp_path.glob("*.jsonl")
            for l in f.read_text(encoding="utf-8").splitlines()]
    wo = next(r for r in rows if r["type"] == "work_order_draft")
    assert wo["llm_source"] == "error"          # degraded, but recorded — never crashed
