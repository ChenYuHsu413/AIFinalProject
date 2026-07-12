"""S5 P2 — read-only MLOps status (registry history / gate report / timeline)."""
from __future__ import annotations

import json

from src.monitor.mlops_status import (
    latest_gate_report,
    mlops_timeline,
    registry_history,
)


def test_registry_history_reads_active_version():
    hist = registry_history()  # reads the committed registry (v1)
    assert hist["active_version"]
    assert isinstance(hist["versions"], list) and hist["versions"]
    active = [v for v in hist["versions"] if v["active"]]
    assert len(active) == 1 and active[0]["version"] == hist["active_version"]
    v = active[0]
    # every version row carries the deploy-decision evidence the panel renders
    assert "macro_f1" in v and "dv_r2" in v and "feature_set" in v
    assert isinstance(v["crc32"], dict) and v["crc32"]  # CRC32 integrity present


def test_latest_gate_report_picks_newest_by_created(tmp_path):
    for name, created, passed in (("candidate_a", "2026-07-10T00:00:00+00:00", False),
                                  ("candidate_b", "2026-07-11T00:00:00+00:00", True)):
        d = tmp_path / name
        d.mkdir()
        (d / "gate_report.json").write_text(
            json.dumps({"passed": passed, "created": created, "checks": []}),
            encoding="utf-8")

    rep = latest_gate_report(registry_root=tmp_path)
    assert rep is not None
    assert rep["passed"] is True and rep["_source"] == "candidate_b"  # newest wins


def test_latest_gate_report_none_when_absent(tmp_path):
    assert latest_gate_report(registry_root=tmp_path) is None


def test_mlops_timeline_causal_chain(tmp_path):
    day = tmp_path / "2026-07-11.jsonl"
    events = [
        {"id": "drift-0001", "type": "drift_detected", "rolling_recon_error": 2.6},
        {"id": "retrain-0001", "type": "retrain_started", "trigger": "drift-0001",
         "mode": "dry_run"},
        {"id": "retrain-0002", "type": "retrain_finished", "trigger": "drift-0001",
         "gate_passed": True, "new_version": None},
        # non-causal noise that must be excluded
        {"id": "alert-0001", "type": "alert_triggered"},
        {"id": "cw-0001", "type": "consistency_warning"},
    ]
    with day.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    tl = mlops_timeline(alerts_dir=tmp_path)
    types = [e["type"] for e in tl]
    assert "alert_triggered" not in types and "consistency_warning" not in types
    # newest first
    assert types == ["retrain_finished", "retrain_started", "drift_detected"]
    # the retrain events link back to the drift that triggered them
    assert all(e["trigger"] == "drift-0001" for e in tl if e["type"].startswith("retrain"))


def test_mlops_timeline_limit_and_empty(tmp_path):
    assert mlops_timeline(alerts_dir=tmp_path / "nope") == []
    day = tmp_path / "2026-07-11.jsonl"
    with day.open("w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"id": f"drift-{i}", "type": "drift_detected"}) + "\n")
    assert len(mlops_timeline(alerts_dir=tmp_path, limit=3)) == 3
