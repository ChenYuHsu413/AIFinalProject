"""S4 — closed loop: DRIFT triggers a non-blocking background retrain."""
from __future__ import annotations

import json
import threading
from pathlib import Path


def test_drift_triggers_background_retrain(tmp_path, monkeypatch):
    from src.models import servo_model_registry as registry
    from src.models import train_servo
    from src.monitor.closed_loop import ClosedLoop
    import src.pipeline.validation_gate as vg

    proceed = threading.Event()  # lets the test observe the "busy" state deterministically

    def fake_train(out_dir, data_config=None):
        proceed.wait(5)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "metrics.json").write_text(
            json.dumps({"version": "candidate", "macro_f1": 0.82}), encoding="utf-8")
        return out_dir

    monkeypatch.setattr(registry, "registry_dir", lambda: tmp_path)
    monkeypatch.setattr(registry, "active_version", lambda default=None: "v1")
    monkeypatch.setattr(registry, "promote_candidate", lambda c, m, note="": "v2")
    monkeypatch.setattr(train_servo, "run", fake_train)
    monkeypatch.setattr(vg, "run_gate", lambda cand, active_version=None: {"passed": True, "checks": []})

    events = []
    cl = ClosedLoop(on_event=events.append, retrain_data_config={"inject_drift": {"gain": 1.3}},
                    auto_retrain=True, alerts_dir=tmp_path)

    assert cl.on_drift({"id": "drift-0001"}) is True      # returns immediately (background)
    assert cl.on_drift({"id": "drift-0002"}) is False     # one retrain at a time
    proceed.set()
    cl.join(timeout=30)

    types = [e["type"] for e in events]
    assert "retrain_started" in types and "retrain_finished" in types
    fin = next(e for e in events if e["type"] == "retrain_finished")
    assert fin["gate_passed"] and fin["new_version"] == "v2" and fin["trigger"] == "drift-0001"
