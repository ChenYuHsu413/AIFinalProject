"""S2 Task A — alert engine for the Servo replay live stream.

Consumes the per-window structured predictions from
``servo_replay_client.iter_window_predictions`` and turns them into alert events
with HYSTERESIS, so a health state that flickers around a decision threshold does
NOT produce a storm of on/off alerts.

Why hysteresis (S1b evidence, not textbook): S1b showed LN↔LO predictions flicker
near the threshold (24-run sample: 2 near-threshold flips, see
``outputs/metrics/replay_window_validation.json``). A raw "risk==High → alert"
would chatter; requiring N consecutive High to trigger and M consecutive
≤Medium to clear debounces it.

Two independent event streams:
  * MAIN alert — driven by ``risk_level`` with hysteresis (trigger/clear).
  * CONSISTENCY notice — emitted whenever ``consistency_warning`` is non-null
    (the classifier state and the DV risk disagree by ≥2 tiers). It is a separate
    advisory event and never drives the main alert.

On a MAIN trigger the event is written to ``outputs/alerts/<date>.jsonl``
IMMEDIATELY (snapshot + ``model_version``); the LLM work-order draft is then
generated on a BACKGROUND THREAD via the existing multi-provider assistant
(offline fallback included) and appended as a linked event — so LLM latency or
failure can never block or delay the alert itself.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.llm.maintenance_assistant import generate_report
from src.utils.paths import load_config, resolve

_RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _feature_summary(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Compact snapshot of the window's features + model output for the event."""
    return {
        "predicted_health_state": pred.get("predicted_health_state"),
        "true_label_in_window": pred.get("_true"),
        "risk_level": pred.get("risk_level"),
        "degradation_score": pred.get("degradation_score"),
        "health_score": pred.get("health_score"),
        "model_confidence": pred.get("model_confidence"),
        "health_state_proba": pred.get("health_state_proba"),
        "top_features": pred.get("top_features"),
        "window_rows": pred.get("_rows"),
        "stream_t": pred.get("_stream_t"),
    }


class AlertEngine:
    """Hysteresis state machine + event sink for the live health stream."""

    def __init__(self, on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                 config: Optional[Dict[str, Any]] = None):
        cfg = config or load_config().get("servo_alert", {})
        self.N = int(cfg.get("high_consecutive", 3))    # windows at High to TRIGGER
        self.M = int(cfg.get("clear_consecutive", 3))   # windows ≤Medium to CLEAR
        self.model_version = str(cfg.get("model_version", "v1"))
        self.alerts_dir = resolve(cfg.get("alerts_dir", "outputs/alerts"))
        self.on_event = on_event

        self.active = False
        self.high_streak = 0
        self.low_streak = 0
        self._counter = 0
        self._active_alert_id: Optional[str] = None
        self._write_lock = threading.Lock()
        self._threads: List[threading.Thread] = []

    # -- event sink --------------------------------------------------------
    def _emit(self, event: Dict[str, Any]) -> None:
        event.setdefault("model_version", self.model_version)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        path = self.alerts_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        line = json.dumps(event, ensure_ascii=False)
        with self._write_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # a UI callback must never break the engine
                pass

    def _next_id(self, kind: str) -> str:
        self._counter += 1
        return f"{kind}-{self._counter:04d}"

    # -- work-order draft (background; never blocks the alert) -------------
    def _spawn_work_order(self, alert_id: str, pred: Dict[str, Any]) -> None:
        def _worker() -> None:
            try:
                report = generate_report(pred)  # never raises; falls back offline
                text, source = report["text"], report["source"]
            except Exception as e:  # defensive: keep the thread from dying silently
                text, source = f"（工單草稿生成失敗：{type(e).__name__}: {e}）", "error"
            self._emit({
                "id": self._next_id("wo"), "alert_id": alert_id,
                "type": "work_order_draft", "ts": _now_iso(),
                "llm_source": source, "text": text,
            })
        t = threading.Thread(target=_worker, daemon=True, name=f"wo-{alert_id}")
        self._threads.append(t)
        t.start()

    # -- main step ---------------------------------------------------------
    def update(self, pred: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Feed one window prediction; return the events emitted this step."""
        events: List[Dict[str, Any]] = []

        # (1) consistency notice — independent of the main alert hysteresis.
        if pred.get("consistency_warning"):
            ev = {"id": self._next_id("cw"), "type": "consistency_warning",
                  "ts": _now_iso(), "stream_t": pred.get("_stream_t"),
                  "message": pred["consistency_warning"],
                  "snapshot": _feature_summary(pred)}
            self._emit(ev)
            events.append(ev)

        # (2) main alert hysteresis on risk_level.
        is_high = _RISK_ORDER.get(pred.get("risk_level"), 0) >= _RISK_ORDER["High"]
        if is_high:
            self.high_streak += 1
            self.low_streak = 0
        else:
            self.low_streak += 1
            self.high_streak = 0

        if not self.active and self.high_streak >= self.N:
            self.active = True
            alert_id = self._next_id("alert")
            self._active_alert_id = alert_id
            ev = {"id": alert_id, "type": "alert_triggered", "ts": _now_iso(),
                  "stream_t": pred.get("_stream_t"),
                  "trigger_rule": f"{self.N} consecutive windows at risk High",
                  "snapshot": _feature_summary(pred)}
            self._emit(ev)                 # write FIRST — never wait on the LLM
            self._spawn_work_order(alert_id, pred)  # draft off the critical path
            events.append(ev)
        elif self.active and self.low_streak >= self.M:
            self.active = False
            ev = {"id": self._next_id("clear"), "type": "alert_cleared",
                  "ts": _now_iso(), "stream_t": pred.get("_stream_t"),
                  "clear_rule": f"{self.M} consecutive windows at risk ≤ Medium",
                  "alert_id": self._active_alert_id,
                  "snapshot": _feature_summary(pred)}
            self._emit(ev)
            self._active_alert_id = None
            events.append(ev)

        return events

    def join_pending(self, timeout: float = 60.0) -> None:
        """Wait for in-flight work-order threads (for CLI/test determinism)."""
        for t in self._threads:
            t.join(timeout=timeout)


# ---------------------------------------------------------------------------
# CLI — run the engine against the live SSE feed (acceptance demo).
# ---------------------------------------------------------------------------
def _run_cli() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.monitor.servo_replay_client import iter_window_predictions

    win = load_config()["servo_replay"]["window"]
    ap = argparse.ArgumentParser(description="Servo alert engine (live SSE).")
    ap.add_argument("--url", default="http://127.0.0.1:8008/servo/stream")
    ap.add_argument("--window-s", type=float, default=float(win["length_s"]))
    ap.add_argument("--step-s", type=float, default=float(win["step_s"]))
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    icons = {"alert_triggered": "🔴 主告警", "alert_cleared": "🟢 告警解除",
             "consistency_warning": "⚠ 矛盾提示", "work_order_draft": "📝 工單草稿"}
    engine = AlertEngine(on_event=lambda e: print(
        f"  [{icons.get(e['type'], e['type'])}] {e.get('id')}"
        + (f"  source={e.get('llm_source')}" if e["type"] == "work_order_draft" else
           f"  t={e.get('stream_t')}  {e.get('trigger_rule') or e.get('clear_rule') or ''}")))

    print(f"[alert] connect {args.url}  N={engine.N} M={engine.M} "
          f"model_version={engine.model_version}\n")
    n = 0
    for pred in iter_window_predictions(args.url, args.window_s, args.step_s):
        n += 1
        print(f"win{n:>3} t={pred['_stream_t']:>7.2f} true={pred['_true']:>4} "
              f"pred={pred['predicted_health_state']:>4} risk={pred['risk_level']:>6} "
              f"DV={pred['degradation_score']:.3f}")
        engine.update(pred)
        if args.max_windows and n >= args.max_windows:
            break
    engine.join_pending()
    print(f"\n[alert] done — {n} windows. events -> {engine.alerts_dir}")


if __name__ == "__main__":
    _run_cli()
