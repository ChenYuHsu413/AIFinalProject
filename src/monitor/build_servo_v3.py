"""Build the Live Monitor v3 demo artefacts from the vendored generator.

Pipeline (offline, run once):
  1. Generate the 30 scenarios (1000 Hz, ~142 tags, v4.1 physics) with the
     vendored `servo_v3_generator` (seed 42) — no 476 MB raw download needed.
  2. Compute a healthy baseline (mean/std per tag) from scenario_01.
  3. For every scenario, at a down-sampled cadence (default 40 Hz) compute:
       - sliding-window features over the physical telemetry tags,
       - a transparent per-subsystem "radar" severity (deviation from baseline),
       - ground-truth labels carried from the data (stage / health / rul / alarm).
  4. Train two models on window features (physical telemetry only, no labels):
       - an early-warning classifier  -> P(warning within horizon)  [ML contribution]
       - a fault-category classifier   -> "which subsystem"          [in-distribution]
     Early-warning is evaluated on held-out whole scenarios (honest split); the
     SAME models are reused for the live SSE stream (identical schema).
  5. Emit small, git-committed replay packs (one JSON per scenario) + the models.

The feature/radar schema lives in `schema.py` and is shared with the backend
live stream, so replay and live see identical features. The model uses ONLY
physical telemetry; label-derived columns are ground truth or the target.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
import joblib

from src.monitor.schema import (
    FEATURE_TAGS,
    RAW_HZ,
    SCENARIO_ROWS,
    SEED,
    STAGE_TO_INT,
    SUBSYSTEMS,
    WARNING_LEVEL,
    healthy_baseline,
    radar_severity,
    window_features,
)
from src.monitor.servo_v3_generator import SCENARIOS, generate_servo_data
from src.utils.paths import resolve

OUT_PACKS = "data/processed/servo_v3"
OUT_MODEL = "outputs/models/monitor_v3_clf.joblib"
OUT_EVAL = "outputs/metrics/monitor_v3_eval.json"


def _process_scenario(scenario_id: int, base: pd.DataFrame, win: int, step: int, ahead: int):
    """Generate one scenario and return (frame_df, feature_df, meta).

    ``ahead`` = number of output frames to look forward for the early-warning
    target: a frame is positive if warning-or-worse occurs within that horizon.
    """
    df = generate_servo_data(rows=SCENARIO_ROWS, scenario_id=scenario_id,
                             sampling_hz=RAW_HZ, seed=SEED)
    feats_full = window_features(df, win)
    radar_full = radar_severity(df, base)

    idx = np.arange(0, len(df), step)  # down-sample to output cadence
    frames = pd.DataFrame({
        "t": df["time_s"].to_numpy()[idx].round(3),
        "stage_int": df["fault_stage"].map(STAGE_TO_INT).to_numpy()[idx],
        "health": df["health_index"].to_numpy()[idx].round(2),
        "rul": df["rul_sec"].to_numpy()[idx].round(2),
        "gt_prob": df["failure_probability"].to_numpy()[idx].round(4),
        "warning": df["warning"].to_numpy()[idx],
        "alarm": df["alarm"].to_numpy()[idx],
        "trip": df["trip"].to_numpy()[idx],
    })
    radar = radar_full.iloc[idx].reset_index(drop=True).round(3)
    frames = pd.concat([frames, radar], axis=1)

    meta = {
        "scenario_id": int(df["scenario_id"].iloc[0]),
        "scenario_name": str(df["scenario_name"].iloc[0]),
        "fault_category": str(df["fault_category"].iloc[0]),
        "alarm_code": str(df["alarm_code"].iloc[-1]),
        "root_cause": str(df["root_cause"].iloc[-1]),
        "maintenance_action": str(df["maintenance_action"].iloc[-1]),
    }

    stage_int = frames["stage_int"].to_numpy()
    warn_now = (stage_int >= WARNING_LEVEL).astype(int)
    y_warn = np.zeros(len(stage_int), dtype=int)
    for i in range(len(stage_int)):
        hi = min(len(stage_int), i + ahead + 1)
        y_warn[i] = int(warn_now[i:hi].max())

    feats = feats_full.iloc[idx].reset_index(drop=True)
    feats["_scenario_id"] = meta["scenario_id"]
    feats["_category"] = meta["fault_category"]
    feats["_stage_int"] = stage_int
    feats["_y_warn"] = y_warn
    feats["_t"] = frames["t"].to_numpy()
    return frames, feats, meta


def _first_time_at_least(times: np.ndarray, mask: np.ndarray) -> float | None:
    hits = np.where(mask)[0]
    return float(times[hits[0]]) if len(hits) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Live Monitor v3 demo artefacts")
    ap.add_argument("--out-hz", type=int, default=40, help="replay cadence (Hz)")
    ap.add_argument("--win-ms", type=int, default=300, help="feature window (ms)")
    ap.add_argument("--holdout", type=int, default=6,
                    help="every Nth scenario held out for honest eval")
    ap.add_argument("--horizon-s", type=float, default=2.0,
                    help="early-warning look-ahead horizon (seconds)")
    args = ap.parse_args()

    step = RAW_HZ // args.out_hz
    win = int(RAW_HZ * args.win_ms / 1000)
    ahead = int(round(args.horizon_s * args.out_hz))
    n_scenarios = len(SCENARIOS)

    print(f"[build] generating {n_scenarios} scenarios via vendored generator "
          f"| out={args.out_hz}Hz step={step} win={win} horizon={args.horizon_s}s")
    healthy = generate_servo_data(rows=SCENARIO_ROWS, scenario_id=1,
                                  sampling_hz=RAW_HZ, seed=SEED)
    base = healthy_baseline(healthy)

    all_frames, all_feats = {}, []
    for sid in range(1, n_scenarios + 1):
        frames, feats, meta = _process_scenario(sid, base, win, step, ahead)
        all_frames[meta["scenario_id"]] = (frames, meta)
        all_feats.append(feats)
        print(f"  scenario {meta['scenario_id']:02d} {meta['scenario_name']:<30} "
              f"frames={len(frames)} cat={meta['fault_category']}")

    feat_df = pd.concat(all_feats, ignore_index=True)
    feat_cols = [c for c in feat_df.columns if not c.startswith("_")]

    # Honest split: hold out whole scenarios (every Nth by id).
    sids = [int(s) for s in sorted(feat_df["_scenario_id"].unique())]
    test_sids = set(sids[::args.holdout])
    train_sids = [s for s in sids if s not in test_sids]
    tr = feat_df[feat_df["_scenario_id"].isin(train_sids)]
    te = feat_df[feat_df["_scenario_id"].isin(test_sids)]
    X_tr, X_te = tr[feat_cols].to_numpy(), te[feat_cols].to_numpy()

    # --- Model 1: early-warning — forecast warning within the horizon -------
    warn_clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                              max_depth=6, random_state=42)
    warn_clf.fit(X_tr, tr["_y_warn"].to_numpy())
    warn_f1 = f1_score(te["_y_warn"].to_numpy(), warn_clf.predict(X_te))

    # --- Model 2: fault category — in-distribution reference labeller -------
    # Trained on ALL scenarios (several categories are singletons, so a held-out
    # split cannot test category generalization). In-distribution reference for
    # the "suspected subsystem" text, NOT a generalization claim.
    cat_clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                             max_depth=6, random_state=42)
    cat_clf.fit(feat_df[feat_cols].to_numpy(), feat_df["_category"].to_numpy())

    print(f"[eval] early-warning F1 (held-out, horizon {args.horizon_s}s) = {warn_f1:.3f}")

    # --- Emit replay packs with per-frame predictions -----------------------
    out_dir = resolve(OUT_PACKS)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, lead_times = [], []
    for sid, (frames, meta) in all_frames.items():
        rows = feat_df[feat_df["_scenario_id"] == sid]
        X = rows[feat_cols].to_numpy()
        pred_prob = warn_clf.predict_proba(X)[:, 1]
        pred_cat = cat_clf.predict(X)
        frames = frames.copy()
        frames["pred_prob"] = pred_prob.round(4)
        frames["pred_cat"] = pred_cat

        times = frames["t"].to_numpy()
        gt_warn_t = _first_time_at_least(times, frames["warning"].to_numpy() == 1)
        gt_alarm_t = _first_time_at_least(times, frames["alarm"].to_numpy() == 1)
        model_alert_t = _first_time_at_least(times, pred_prob >= 0.5)
        lead = (round(gt_warn_t - model_alert_t, 3)
                if gt_warn_t is not None and model_alert_t is not None else None)
        lead_alarm = (round(gt_alarm_t - model_alert_t, 3)
                      if gt_alarm_t is not None and model_alert_t is not None else None)
        if lead_alarm is not None:
            lead_times.append(lead_alarm)

        final_cat = str(pd.Series(pred_cat).mode().iloc[0])
        pack = {
            "meta": {**meta,
                     "out_hz": args.out_hz,
                     "n_frames": len(frames),
                     "gt_warning_t": gt_warn_t,
                     "gt_alarm_t": gt_alarm_t,
                     "model_alert_t": model_alert_t,
                     "lead_time_s": lead,
                     "lead_to_alarm_s": lead_alarm,
                     "predicted_category": final_cat},
            "subsystems": list(SUBSYSTEMS.keys()),
            "frames": frames.to_dict(orient="records"),
        }
        (out_dir / f"scenario_{sid:02d}.json").write_text(
            json.dumps(pack, separators=(",", ":")), encoding="utf-8")
        manifest.append({
            "scenario_id": sid,
            "scenario_name": meta["scenario_name"],
            "fault_category": meta["fault_category"],
            "n_frames": len(frames),
            "lead_to_alarm_s": lead_alarm,
            "held_out": sid in test_sids,
        })

    (out_dir / "manifest.json").write_text(
        json.dumps({"scenarios": manifest,
                    "subsystems": list(SUBSYSTEMS.keys()),
                    "out_hz": args.out_hz}, indent=2), encoding="utf-8")

    joblib.dump({"warn_clf": warn_clf, "cat_clf": cat_clf,
                 "feat_cols": feat_cols}, resolve(OUT_MODEL))

    eval_json = {
        "dataset": "Servo AI Dataset v4.1 Enterprise (synthetic, AI-generated via vendored generator)",
        "n_scenarios": n_scenarios,
        "out_hz": args.out_hz,
        "win_ms": args.win_ms,
        "horizon_s": args.horizon_s,
        "n_features": len(feat_cols),
        "train_scenarios": train_sids,
        "test_scenarios": sorted(int(s) for s in test_sids),
        "early_warning_f1_heldout": round(float(warn_f1), 4),
        "median_lead_to_alarm_s": round(float(np.median(lead_times)), 3) if lead_times else None,
        "n_scenarios_with_lead": len(lead_times),
        "note": "Synthetic AI-generated demo, reproducible from the committed "
                "generator (no raw download). The early-warning model forecasts an "
                "imminent warning from physical telemetry only and is scored on "
                "held-out scenarios. The fault-category labeller is trained on all "
                "scenarios (several categories are singletons) and is an "
                "in-distribution reference, NOT a generalization claim. Demo track, "
                "separate from and not superseding the real PHM main line.",
    }
    resolve(OUT_EVAL).write_text(json.dumps(eval_json, indent=2), encoding="utf-8")
    print(f"[done] packs -> {out_dir} | model -> {resolve(OUT_MODEL)} | "
          f"median lead-to-alarm = {eval_json['median_lead_to_alarm_s']}s")


if __name__ == "__main__":
    main()
