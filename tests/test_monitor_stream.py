"""S5 — backend monitor aggregation: enriched window schema + events pagination.

The SSE endpoint itself is an infinite ``StreamingResponse`` and (like
``/monitor/stream``) is NOT exercised over ``TestClient`` — that buffers streaming
bodies. Instead we drive the SAME in-process generator the endpoint uses
(``iter_enriched_windows``) and unit-test the paginated events reader directly.
"""
from __future__ import annotations

import json

import pytest

from src.utils.paths import load_config, resolve


def _manifest_present() -> bool:
    return resolve(load_config()["servo_replay"]["manifest"]).exists()


_ENRICHED_KEYS = {
    "window_index", "window_ts", "predicted_health_state", "smoothed_state",
    "recent_states", "true_label", "health_state_proba", "degradation_score",
    "model_confidence", "risk_level", "window_rows", "alert_state",
    "drift_status", "model_version", "replay_segment", "events",
}


@pytest.mark.skipif(not _manifest_present(), reason="replay material not extracted")
def test_enriched_window_schema(tmp_path):
    from src.monitor.monitor_stream import iter_enriched_windows

    events = list(iter_enriched_windows(segment="normal", max_windows=12,
                                        alerts_dir=tmp_path))
    assert len(events) >= 6

    for e in events:
        assert _ENRICHED_KEYS <= e.keys()
        # smoothed_state is a display-layer majority over the last K raw states.
        assert e["smoothed_state"] in e["recent_states"]
        assert 0.0 <= e["degradation_score"] <= 1.0
        assert 0.0 <= e["model_confidence"] <= 1.0
        assert e["risk_level"] in ("Low", "Medium", "High")
        assert set(e["alert_state"]) == {"active", "high_streak", "low_streak",
                                         "active_alert_id"}
        assert "available" in e["drift_status"]
        assert e["replay_segment"]["key"] == "normal"
        assert e["model_version"]  # stamped from the active registry version

    # DV must climb across the LN→LO→HI journey (same guarantee as the pipeline test).
    dv_first = events[0]["degradation_score"]
    dv_last = events[-1]["degradation_score"]
    assert dv_last >= dv_first


@pytest.mark.skipif(not _manifest_present(), reason="replay material not extracted")
def test_drift_segment_injects(tmp_path):
    from src.monitor.monitor_stream import iter_enriched_windows

    events = list(iter_enriched_windows(segment="drift", max_windows=12,
                                        alerts_dir=tmp_path))
    assert events and events[0]["replay_segment"]["injected"] is True
    # Drift baseline exists for the active version -> drift_status is populated.
    assert any(e["drift_status"].get("available") for e in events)


def test_unknown_segment_rejected():
    from src.monitor.monitor_stream import iter_enriched_windows

    with pytest.raises(KeyError):
        next(iter_enriched_windows(segment="nope"))


def test_events_pagination(tmp_path):
    from src.monitor.monitor_stream import read_recent_events

    day = tmp_path / "2026-07-11.jsonl"
    made = []
    with day.open("w", encoding="utf-8") as f:
        for i in range(5):
            ev = {"id": f"alert-{i:04d}", "type": "alert_triggered",
                  "ts": "2026-07-11T00:00:0%d+00:00" % i, "stream_t": float(i)}
            made.append(ev)
            f.write(json.dumps(ev) + "\n")
        # a work order + a non-feed type that must be excluded from the feed
        f.write(json.dumps({"id": "wo-0001", "alert_id": "alert-0000",
                            "type": "work_order_draft", "text": "x"}) + "\n")
        f.write(json.dumps({"id": "x", "type": "retrain_started"}) + "\n")

    page1 = read_recent_events(alerts_dir=tmp_path, limit=2, offset=0)
    assert page1["total"] == 5           # 5 feed events (wo + retrain excluded)
    assert page1["limit"] == 2 and page1["offset"] == 0
    assert [e["id"] for e in page1["events"]] == ["alert-0004", "alert-0003"]  # newest first
    assert len(page1["work_orders"]) == 1

    page2 = read_recent_events(alerts_dir=tmp_path, limit=2, offset=2)
    assert [e["id"] for e in page2["events"]] == ["alert-0002", "alert-0001"]

    page3 = read_recent_events(alerts_dir=tmp_path, limit=2, offset=4)
    assert [e["id"] for e in page3["events"]] == ["alert-0000"]  # tail partial page


def test_events_empty_dir(tmp_path):
    from src.monitor.monitor_stream import read_recent_events

    out = read_recent_events(alerts_dir=tmp_path / "nope", limit=10)
    assert out == {"events": [], "work_orders": [], "total": 0, "limit": 10, "offset": 0}
