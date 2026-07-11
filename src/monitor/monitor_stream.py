"""S5 — self-contained backend aggregation for the Servo monitor page.

The Next.js Command Center only *renders*; window aggregation, inference, alert
hysteresis and drift detection all stay here on the server (architecture rule:
"後端算、前端畫"). This module drives the SAME in-process building blocks the
Streamlit live monitor and the CLI use — no separate publisher process:

    iter_replay_rows(...)                 # read data/demo/replay/ segment CSVs
      -> iter_window_predictions_from_rows # window + reference-model inference
      -> AlertEngine.update / DriftDetector.update  # events (unchanged behaviour)

so there is a single source of truth for the pipeline. ``servo_replay_publisher``
stays as a standalone local-dev tool (it publishes raw rows over SSE for the CLI
consumer / Streamlit page); this endpoint does not depend on it.

``smoothed_state`` is a DISPLAY-layer decision — a majority vote over the last K
raw per-window predictions — computed here so the honest "raw vs smoothed" design
of the Streamlit page carries over to the web UI (the front end shows both).
"""
from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional

import numpy as np

from src.monitor.servo_replay_client import iter_window_predictions_from_rows
from src.monitor.servo_replay_stream import iter_replay_rows
from src.utils.paths import load_config, resolve

# Event types that belong in the monitor feed / polling endpoint (skips the
# internal noise; work_order_draft is kept so the UI can attach the LLM draft).
FEED_EVENT_TYPES = (
    "alert_triggered", "alert_cleared", "consistency_warning",
    "drift_detected", "drift_cleared", "work_order_draft",
)


def replay_segments() -> Dict[str, Dict[str, Any]]:
    """The replay scenarios the ``segment`` query param selects.

    ``normal`` = the real FMCRD degradation journey (LN→LO→HI). ``drift`` = the
    same journey with a SIMULATED current-sensor gain drift injected into the HI
    segment (S4 demo), so the drift detector trips on a genuine off-manifold shift
    rather than on the in-distribution degradation itself.
    """
    inj = load_config().get("servo_drift", {}).get("injection", {"gain": 1.3, "offset": 0.0})
    order = load_config()["servo_replay"].get("order", ["LN", "LO", "HI"])
    return {
        "normal": {
            "order": list(order), "inject": None,
            "label": f"{'→'.join(order)}（真實 FMCRD 一般段落）",
        },
        "drift": {
            "order": list(order),
            "inject": {"segments": ["HI"], "gain": float(inj.get("gain", 1.3)),
                       "offset": float(inj.get("offset", 0.0))},
            "label": f"{'→'.join(order)} ＋ HI 段注入感測器增益漂移（模擬真實故障）",
        },
    }


def _drift_status(drift, available: bool) -> Dict[str, Any]:
    """Per-window drift snapshot read from the detector's rolling state.

    Read AFTER ``drift.update(window)`` so ``err_buf[-1]`` is this window's instant
    reconstruction error; no recomputation. ``threshold_p95`` is the training-set
    P95 the rolling mean must exceed to trip.
    """
    if not available or drift is None or not drift.err_buf:
        return {"available": False}
    inst = float(drift.err_buf[-1])
    roll = float(np.mean(drift.err_buf))
    return {
        "available": True,
        "instant_recon_error": round(inst, 5),
        "rolling_recon_error": round(roll, 5),
        "threshold_p95": round(float(drift.p95), 5),
        "triggered": bool(drift.active),
    }


def iter_enriched_windows(segment: str = "normal",
                          max_windows: Optional[int] = None,
                          alerts_dir: Optional[Path] = None
                          ) -> Iterator[Dict[str, Any]]:
    """Yield one enriched event dict per window (synchronous, in-process).

    Server-independent: the SSE endpoint drives this off the event loop via
    ``asyncio.to_thread`` (per-window inference is blocking), and the unit tests
    drive it directly. ``alerts_dir`` overrides the JSONL sink (tests / isolation).
    """
    from src.models import servo_model_registry as registry
    from src.monitor.alert_engine import AlertEngine
    from src.monitor.drift_detector import DriftDetector

    segments = replay_segments()
    if segment not in segments:
        raise KeyError(f"未知 replay 段落 {segment!r}（可用：{list(segments)}）")
    seg = segments[segment]

    cfg = load_config()
    rep = cfg["servo_replay"]
    al = dict(cfg.get("servo_alert", {}))
    dcfg = cfg.get("servo_drift", {})
    w_s, s_s = float(rep["window"]["length_s"]), float(rep["window"]["step_s"])
    k = int(al.get("status_smoothing_windows", 3))
    if alerts_dir is not None:
        al["alerts_dir"] = str(alerts_dir)

    engine = AlertEngine(config=al)
    drift = None
    try:
        active_dir = registry.version_dir(registry.active_version())
        drift = DriftDetector(active_dir, config=dcfg, alerts_dir=alerts_dir)
    except Exception:
        drift = None  # this version has no drift baseline -> drift panel degrades

    rows = iter_replay_rows(order=seg["order"], loop=False, inject=seg["inject"])
    recent: Deque[str] = deque(maxlen=k)
    try:
        for i, pred in enumerate(iter_window_predictions_from_rows(rows, w_s, s_s), 1):
            state = pred["predicted_health_state"]
            recent.append(state)
            smoothed = Counter(recent).most_common(1)[0][0]

            events: List[Dict[str, Any]] = list(engine.update(pred))
            if drift is not None:
                events.extend(drift.update(pred))

            yield {
                "window_index": i,
                "window_ts": pred["_stream_t"],
                "predicted_health_state": state,
                "smoothed_state": smoothed,
                "recent_states": list(recent),
                "true_label": pred.get("_true"),
                "health_state_proba": pred.get("health_state_proba"),
                "degradation_score": pred.get("degradation_score"),
                "model_confidence": pred.get("model_confidence"),
                "risk_level": pred.get("risk_level"),
                "consistency_warning": pred.get("consistency_warning"),
                "window_rows": pred.get("_rows"),
                "alert_state": {
                    "active": engine.active,
                    "high_streak": engine.high_streak,
                    "low_streak": engine.low_streak,
                    "active_alert_id": engine._active_alert_id,
                },
                "drift_status": _drift_status(drift, drift is not None),
                "model_version": engine.model_version,
                "replay_segment": {"key": segment, "label": seg["label"],
                                   "injected": seg["inject"] is not None},
                "events": events,  # events emitted THIS window (live feed without polling)
            }
            if max_windows and i >= max_windows:
                break
    finally:
        engine.join_pending(timeout=5)


def _alerts_dir() -> Path:
    return resolve(load_config().get("servo_alert", {}).get("alerts_dir", "outputs/alerts"))


def read_recent_events(alerts_dir: Optional[Path] = None, limit: int = 50,
                       offset: int = 0) -> Dict[str, Any]:
    """Paginated read of the alert/consistency/DRIFT event JSONL (newest first).

    Mirrors the Streamlit ``_read_events`` glob, then filters to feed types and
    paginates. ``work_order_draft`` events are returned separately so the UI can
    attach the LLM draft to its alert without them consuming a feed slot.
    """
    adir = Path(alerts_dir) if alerts_dir is not None else _alerts_dir()
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    rows: List[Dict[str, Any]] = []
    if adir.exists():
        for f in sorted(adir.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    work_orders = [e for e in rows if e.get("type") == "work_order_draft"]
    feed = [e for e in rows if e.get("type") in FEED_EVENT_TYPES
            and e.get("type") != "work_order_draft"]
    feed.reverse()  # newest first
    page = feed[offset:offset + limit]
    return {
        "events": page,
        "work_orders": work_orders,
        "total": len(feed),
        "limit": limit,
        "offset": offset,
    }
