"""S5 P2 — read-only MLOps status for the ``/servo/mlops`` panel.

Assembles three things the panel renders (all READ-ONLY — no trigger surface; the
demo is driven by ``scripts/run_drift_demo.py`` to avoid an online misfire):

  1. registry version history — each version's held-out metrics, feature set,
     model-file CRC32 (integrity) and the active marker (from
     ``models/registry/registry.json`` + per-version ``metrics.json``).
  2. the most recent ``gate_report.json`` (validation gate PASS/FAIL detail).
  3. the drift → retrain → promote causal chain, read from the same
     ``outputs/alerts/*.jsonl`` event stream the closed loop writes.

Nothing here mutates the registry, trains, or promotes — it only reads what the
S3/S4 pipeline already recorded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.paths import load_config, resolve

# The drift → retrain → (gate) → promote causal chain, as event types.
CAUSAL_EVENT_TYPES = (
    "drift_detected", "drift_cleared",
    "retrain_started", "retrain_finished", "retrain_error",
)


def registry_history() -> Dict[str, Any]:
    """Version history with per-version metrics + CRC32 + active marker.

    Raises ``FileNotFoundError`` (→ 503 at the endpoint) if the registry is absent.
    """
    from src.models import servo_model_registry as registry

    reg = registry.read_registry()
    active = reg.get("active_version")
    versions: List[Dict[str, Any]] = []
    for v in registry.list_versions():
        s = reg["versions"].get(v, {})
        vdir = registry.version_dir(v)
        m: Dict[str, Any] = {}
        mp = vdir / "metrics.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                m = {}
        versions.append({
            "version": v,
            "active": v == active,
            "created": s.get("created") or m.get("created"),
            "macro_f1": s.get("macro_f1", m.get("macro_f1")),
            "dv_r2": s.get("dv_r2", m.get("dv_r2")),
            "dv_mae": m.get("dv_mae"),
            "feature_set": s.get("feature_set", m.get("feature_set")),
            "placeholder": s.get("placeholder", m.get("placeholder")),
            "note": s.get("note"),
            "eval_mode": m.get("eval_mode"),
            "clf_model": m.get("clf_model"),
            "reg_model": m.get("reg_model"),
            "crc32": m.get("model_crc32"),
            "has_gate_report": (vdir / "gate_report.json").exists(),
        })
    return {
        "active_version": active,
        "updated": reg.get("updated"),
        "versions": versions,
        "outputs_consistent": registry.outputs_models_consistent_with_active(),
    }


def latest_gate_report(registry_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The most recent ``gate_report.json`` across version + candidate dirs.

    Picks the newest by the report's own ``created`` timestamp (deterministic;
    avoids mtime). Returns None when no gate has ever run (e.g. a v1 migrated
    straight from ``outputs/models`` never went through the gate).
    """
    from src.models import servo_model_registry as registry

    root = Path(registry_root) if registry_root is not None else registry.registry_dir()
    reports: List[Dict[str, Any]] = []
    for p in sorted(root.glob("*/gate_report.json")):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rep["_source"] = p.parent.name
        reports.append(rep)
    if not reports:
        return None
    return max(reports, key=lambda r: r.get("created", ""))


def mlops_timeline(alerts_dir: Optional[Path] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Drift/retrain causal-chain events from the alert JSONL sink (newest first).

    Events carry ``trigger`` (the drift event id) so the UI can link a
    ``retrain_started`` / ``retrain_finished`` back to the ``drift_detected`` that
    caused it.
    """
    adir = (Path(alerts_dir) if alerts_dir is not None
            else resolve(load_config().get("servo_alert", {}).get("alerts_dir", "outputs/alerts")))
    limit = max(1, min(int(limit), 500))
    rows: List[Dict[str, Any]] = []
    if adir.exists():
        for f in sorted(adir.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") in CAUSAL_EVENT_TYPES:
                    rows.append(e)
    rows.reverse()  # newest first
    return rows[:limit]


def mlops_status() -> Dict[str, Any]:
    """The full read-only panel payload (registry + latest gate + timeline)."""
    return {
        "registry": registry_history(),
        "gate_report": latest_gate_report(),
        "timeline": mlops_timeline(),
    }
