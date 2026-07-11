#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 Task C — one-command drift scenario (detect → retrain → gate → switch → digest).

Repeatable, no human intervention:
  0. reset to a clean v1 (remove any v2 / candidates from a previous run).
  1. NORMAL stream (LN + HI degradation) → drift detector on v1 → assert NO drift
     (the "degradation ≠ drift" hard acceptance).
  2. INJECTED drift stream (simulated current-sensor gain) → DRIFT fires.
  3. CLOSED LOOP: DRIFT → background retrain folding the drifted condition in →
     gate → promote v2.
  4. DIGEST: replay the SAME drifted stream through v2's drift baseline → no drift
     (the drift was digested by retraining).
Records a full evidence JSON, plus a complementary signal (model confidence) and
the v1→v2 improvement on drifted data. Then resets to v1 so it can be re-run.

Run::

    python scripts/run_drift_demo.py
    python scripts/run_drift_demo.py --keep   # keep v2 (skip the reset at the end)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.models import servo_model_registry as registry  # noqa: E402
from src.monitor.closed_loop import ClosedLoop  # noqa: E402
from src.monitor.drift_detector import (DriftDetector, inject_sensor_drift_features)  # noqa: E402
from src.monitor.servo_replay_client import iter_window_predictions_from_rows  # noqa: E402
from src.monitor.servo_replay_stream import iter_replay_rows  # noqa: E402
from src.utils.paths import load_config, resolve  # noqa: E402

_GAIN = 1.3


def _reset_to_v1() -> None:
    reg = registry.read_registry()
    for v in list(reg.get("versions", {})):
        if v != "v1":
            shutil.rmtree(registry.version_dir(v), ignore_errors=True)
            reg["versions"].pop(v, None)
    for cand in registry.registry_dir().glob("candidate_*"):
        shutil.rmtree(cand, ignore_errors=True)
    reg["active_version"] = "v1"
    registry.write_registry(reg)


def _feed(detector, rows, w_s, s_s):
    """Stream rows through the window pipeline + drift detector."""
    drift_events, confs, trues = [], [], []
    for pred in iter_window_predictions_from_rows(rows, w_s, s_s):
        drift_events += detector.update(pred)
        confs.append(pred["model_confidence"])
        trues.append(pred["_true"])
    return drift_events, confs, trues


def _clf_on_drifted(version: str, te_drift, labels):
    lm = registry.load_dir(registry.version_dir(version))
    cols = lm.feature_config["feature_columns"]
    pred = lm.clf_pipeline.predict(te_drift[cols])
    return {"accuracy": float(accuracy_score(te_drift["ylabel"], pred)),
            "macro_f1": float(f1_score(te_drift["ylabel"], pred, average="macro",
                                       labels=labels, zero_division=0))}


def run(keep: bool = False) -> Path:
    cfg = load_config()
    w_s = float(cfg["servo_replay"]["window"]["length_s"])
    s_s = float(cfg["servo_replay"]["window"]["step_s"])
    dcfg = cfg.get("servo_drift", {})
    demo_alerts = resolve("outputs/alerts")

    print("[demo] 0) reset -> v1")
    _reset_to_v1()
    v1_dir = registry.version_dir("v1")

    # 1) NORMAL (LN + HI degradation): must NOT trigger drift.
    det = DriftDetector(v1_dir, config=dcfg, alerts_dir=demo_alerts)
    normal_rows = list(iter_replay_rows(order=["LN", "HI"]))
    normal_events, normal_confs, _ = _feed(det, normal_rows, w_s, s_s)
    false_trigger = any(e["type"] == "drift_detected" for e in normal_events)
    print(f"[demo] 1) 一般段落（LN+HI 退化）：drift={'FIRED (誤觸!)' if false_trigger else '無（正確）'}")

    # 2) INJECTED sensor drift: must trigger (same detector continues).
    drift_rows = list(iter_replay_rows(order=["HI"], inject={"segments": ["HI"], "gain": _GAIN}))
    drift_events, drift_confs, _ = _feed(det, drift_rows, w_s, s_s)
    drift_fired = any(e["type"] == "drift_detected" for e in drift_events)
    print(f"[demo] 2) 注入感測器漂移（gain×{_GAIN}）：drift={'FIRED（正確）' if drift_fired else '無（未觸發!)'}")

    # 3) CLOSED LOOP: retrain folding the drifted condition in -> promote v2.
    print("[demo] 3) 閉環：DRIFT → 背景重訓（納入漂移工況）→ 閘門 → 轉正 …")
    cl = ClosedLoop(retrain_data_config={"inject_drift": {"gain": _GAIN}},
                    auto_retrain=True, alerts_dir=demo_alerts)
    cl.on_drift(drift_events[0] if drift_events else {"id": "drift-demo"})
    cl.join(timeout=300)
    new_version = registry.active_version()
    print(f"[demo]    -> active = {new_version}")

    # 4) DIGEST: same drifted stream through v2's baseline -> no drift.
    digested = None
    if new_version != "v1":
        det2 = DriftDetector(registry.version_dir(new_version), config=dcfg, alerts_dir=demo_alerts)
        digest_events, _, _ = _feed(det2, drift_rows, w_s, s_s)
        digested = not any(e["type"] == "drift_detected" for e in digest_events)
        print(f"[demo] 4) v2 對同一漂移段：drift={'FIRED（未消化!)' if not digested else '無（已消化）'}")

    # improvement + complementary signal
    df = pd.read_parquet(resolve(cfg["servo"]["processed_features"]))
    te = df[df["split"] == "test"]
    labels = sorted(te["ylabel"].unique())
    te_drift = inject_sensor_drift_features(te, _GAIN)
    clf_v1 = _clf_on_drifted("v1", te_drift, labels)
    clf_v2 = _clf_on_drifted(new_version, te_drift, labels) if new_version != "v1" else None
    conf_normal, conf_drift = float(np.mean(normal_confs)), float(np.mean(drift_confs))

    evidence = {
        "scenario": "injected current-sensor gain drift (SIMULATED real-world fault)",
        "gain": _GAIN, "window": {"W": w_s, "S": s_s},
        "acceptance": {
            "hi_degradation_no_false_trigger": not false_trigger,
            "injected_drift_triggers": drift_fired,
            "v2_digests_drift": digested,
        },
        "drift_baseline_p95": {
            "v1": round(det.p95, 5),
            new_version: round(det2.p95, 5) if new_version != "v1" else None,
        },
        "complementary_signal_model_confidence": {
            "mean_normal": round(conf_normal, 4), "mean_drifted": round(conf_drift, 4),
            "dropped": bool(conf_drift < conf_normal),
            "note": ("confidence/supervised 訊號可補足重建式偵測盲區；實務應多訊號並用"
                     if conf_drift < conf_normal else "此情境信心未明顯下滑"),
        },
        "improvement_on_drifted_test_data": {"v1": clf_v1, new_version: clf_v2},
        "promoted_version": new_version,
    }
    out = resolve("outputs/metrics/drift_demo.json")
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 驗收 ===")
    a = evidence["acceptance"]
    print(f"  (a) HI 退化不誤觸：{a['hi_degradation_no_false_trigger']}")
    print(f"  (b) 注入漂移觸發：{a['injected_drift_triggers']}")
    print(f"  (c) v2 消化漂移：{a['v2_digests_drift']}")
    print(f"  信心訊號（互補）：normal {conf_normal:.3f} -> drift {conf_drift:.3f}"
          f"（{'下滑' if conf_drift < conf_normal else '未明顯下滑'}）")
    if clf_v2:
        print(f"  漂移資料上分類 macro-F1：v1 {clf_v1['macro_f1']:.3f} -> {new_version} {clf_v2['macro_f1']:.3f}")
    print(f"  -> {out}")

    if not keep:
        print("[demo] reset -> v1（可重複執行）")
        _reset_to_v1()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="保留 v2（結尾不重置回 v1）")
    args = ap.parse_args()
    run(keep=args.keep)


if __name__ == "__main__":
    main()
