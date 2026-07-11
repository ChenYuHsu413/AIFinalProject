"""S4 Task B — close the loop: DRIFT → retrain → gate → (optional) version switch.

On a DRIFT event this kicks the retrain pipeline on a BACKGROUND THREAD (so the
monitoring stream is never blocked) and writes the causal chain back to the same
event stream the dashboard reads:

    drift_detected → retrain_started → retrain_finished (gate PASS/FAIL, [new version])

By default the retrain runs in DRY-RUN (trains + validates, does not switch the
active version — a human confirms promotion); ``config.yaml::servo_drift.auto_retrain``
flips it to full auto-promote.

Label assumption (honest disclosure): here the retrain folds the drifted operating
condition into TRAIN using replay rows that already carry ``ylabel``. In a real
factory, post-drift data would need a LABELLING process (inspection records
back-filled, etc.) before retraining — a simplification the demo makes explicit.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.utils.paths import load_config, resolve


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ClosedLoop:
    """Coordinates the DRIFT → retrain → gate → promote reaction (non-blocking)."""

    def __init__(self, on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                 retrain_data_config: Optional[Dict[str, Any]] = None,
                 auto_retrain: Optional[bool] = None,
                 alerts_dir: Optional[Path] = None):
        cfg = load_config()
        drift_cfg = cfg.get("servo_drift", {})
        self.auto_retrain = bool(drift_cfg.get("auto_retrain", False)
                                 if auto_retrain is None else auto_retrain)
        self.retrain_data_config = retrain_data_config
        self.on_event = on_event
        acfg = cfg.get("servo_alert", {})
        self.alerts_dir = Path(alerts_dir) if alerts_dir else resolve(acfg.get("alerts_dir", "outputs/alerts"))
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._busy = False
        self._counter = 0

    def _emit(self, event: Dict[str, Any]) -> None:
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        path = self.alerts_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass

    def _next_id(self, kind: str) -> str:
        self._counter += 1
        return f"{kind}-{self._counter:04d}"

    def on_drift(self, drift_event: Dict[str, Any]) -> bool:
        """React to a DRIFT event. Returns False if a retrain is already running."""
        if self._busy:
            return False
        self._busy = True
        self._thread = threading.Thread(
            target=self._retrain, args=(drift_event,), daemon=True, name="closed-loop")
        self._thread.start()
        return True

    def _retrain(self, drift_event: Dict[str, Any]) -> None:
        from src.models import servo_model_registry as registry
        from src.models import train_servo
        from src.pipeline.validation_gate import run_gate

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        started = {"id": self._next_id("retrain"), "type": "retrain_started",
                   "ts": _now_iso(), "trigger": drift_event.get("id"),
                   "mode": "auto" if self.auto_retrain else "dry_run",
                   "reason": "DRIFT 觸發自動重訓（納入漂移工況資料）"}
        self._emit(started)
        try:
            old_active = registry.active_version(default=None)
            cand = registry.registry_dir() / f"candidate_{ts}"
            # Re-fit the DEPLOYED model family on the new data (don't re-search
            # architectures on every drift — that would need full re-validation).
            dc = dict(self.retrain_data_config or {})
            if old_active and "fixed_clf_model" not in dc:
                try:
                    dc["fixed_clf_model"] = registry.load_version(
                        old_active).feature_config.get("clf_model")
                except Exception:
                    pass
            train_servo.run(out_dir=cand, data_config=dc)
            report = run_gate(cand, active_version=old_active)
            new_version = None
            if report["passed"] and self.auto_retrain:
                metrics = json.loads((cand / "metrics.json").read_text(encoding="utf-8"))
                new_version = registry.promote_candidate(
                    cand, metrics, note=f"drift auto-retrain {ts}")
            self._emit({
                "id": self._next_id("retrain"), "type": "retrain_finished",
                "ts": _now_iso(), "trigger": drift_event.get("id"),
                "gate_passed": report["passed"], "mode": started["mode"],
                "old_active": old_active, "new_version": new_version,
                "candidate_dir": str(cand),
                "gate_report": str(cand / "gate_report.json"),
            })
        except Exception as e:  # never let the loop thread die silently
            self._emit({"id": self._next_id("retrain"), "type": "retrain_error",
                        "ts": _now_iso(), "error": f"{type(e).__name__}: {e}"})
        finally:
            self._busy = False

    def join(self, timeout: float = 300.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
