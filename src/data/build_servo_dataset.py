"""Build the Module Servo feature table.

Two paths:
  * **real** — if ``servo.raw_dir`` contains CSVs, aggregate each ``run_index``
    segment into one feature row (``servo_features.build_feature_table``).
  * **placeholder** — otherwise synthesise a class-conditional feature table so
    the whole UI (predict / simulator / assistant) is demoable before the real
    PHM dataset is downloaded.  Clearly flagged via ``servo.placeholder``.

Outputs:
  * ``servo.processed_features``  — full aggregated feature table (parquet)
  * ``servo.feature_demo``        — small CSV used by the training simulator
  * ``servo.sample_predictions``  — a few rows for the dashboard demo

Run::

    python -m src.data.build_servo_dataset
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_servo import load_raw_servo
from src.features.servo_features import (
    HEALTH_LABELS,
    all_feature_columns,
    build_feature_table,
)
from src.utils.paths import ensure_output_dirs, load_config, resolve

# Per-class severity (0 = healthy .. 1 = most degraded). Drives the synthetic
# means so the placeholder data is *learnable* and behaves sensibly in the UI.
_SEVERITY = {"LN": 0.05, "LO": 0.35, "MED": 0.65, "HI": 0.92}


def _check_dv_range(dv: pd.Series) -> None:
    """Warn if real DV is outside 0..1 (the scale the dv_risk bands assume)."""
    lo, hi = float(dv.min()), float(dv.max())
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        print(f"    [!] 警告：真實 DV 範圍為 [{lo:.3f}, {hi:.3f}]，超出預期的 0..1。"
              "config.yaml::servo.dv_risk 風險帶是以 0..1 校準，請正規化 DV 或重校風險帶"
              "（見 docs/MODULE_SERVO_PLAN.md §10）。")


def generate_placeholder(runs_per_class: int, seed: int = 42) -> pd.DataFrame:
    """Synthesise an aggregated feature table with realistic class structure.

    Degradation raises position error, three-phase / Q-axis current, torque and
    speed variability, while healthy runs sit near nominal.  DV tracks severity.
    """
    rng = np.random.default_rng(seed)
    cols = all_feature_columns()
    rows = []
    # Noise is deliberately large so adjacent classes OVERLAP — this keeps the
    # training simulator educational (more data / better features / stronger
    # models visibly help instead of everything scoring ~1.0).
    for label in HEALTH_LABELS:
        s = _SEVERITY[label]
        for _ in range(runs_per_class):
            feat = {c: 0.0 for c in cols}
            # --- motion ---
            feat["rotor_speed_mean"] = 2000 + rng.normal(0, 40)
            feat["rotor_speed_std"] = 5 + 60 * s + rng.normal(0, 22)
            feat["rotor_speed_rms"] = feat["rotor_speed_mean"] + rng.normal(0, 8)
            feat["torque_mean"] = 0.7 + 0.5 * s + rng.normal(0, 0.18)
            feat["torque_std"] = 0.05 + 0.35 * s + rng.normal(0, 0.12)
            feat["torque_rms"] = feat["torque_mean"] + 0.5 * feat["torque_std"]
            feat["del_pos_mean"] = 0.5 + rng.normal(0, 0.04)
            feat["del_pos_std"] = 0.02 + 0.12 * s + rng.normal(0, 0.05)
            feat["del_pos_rms"] = abs(feat["del_pos_mean"]) + feat["del_pos_std"]
            # --- current ---
            base_i = 3.0 + 2.5 * s
            for ph in ("i_3p_a", "i_3p_b", "i_3p_c"):
                feat[f"{ph}_rms"] = base_i + rng.normal(0, 0.9)
            feat["current_rms"] = base_i + rng.normal(0, 0.85)
            feat["direct_rms"] = 0.4 + 0.3 * s + rng.normal(0, 0.18)
            feat["direct_std"] = 0.05 + 0.15 * s + rng.normal(0, 0.1)
            feat["quadrature_rms"] = 1.2 + 2.0 * s + rng.normal(0, 0.8)
            feat["quadrature_std"] = 0.1 + 0.5 * s + rng.normal(0, 0.25)
            # --- position tracking ---
            feat["rod_demand_pos_mean"] = 50 + rng.normal(0, 1)
            feat["rod_actual_pos_mean"] = feat["rod_demand_pos_mean"] - (0.05 + 2.0 * s)
            feat["position_error_mean"] = -(0.05 + 2.0 * s) + rng.normal(0, 0.9)
            feat["position_error_max"] = 0.2 + 5.0 * s + rng.normal(0, 2.0)
            feat["position_error_std"] = 0.05 + 1.5 * s + rng.normal(0, 0.7)
            # --- labels ---
            feat["ylabel"] = label
            feat["DV"] = float(np.clip(s + rng.normal(0, 0.12), 0.0, 1.0))
            rows.append(feat)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "run_index", np.arange(len(df)))
    return df


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    sv = cfg["servo"]

    raw = load_raw_servo()
    if len(raw):
        print(f"[Module Servo] 偵測到真實 PHM 原始資料（{len(raw)} 列），聚合中…")
        table = build_feature_table(raw, label_map=sv.get("ylabel_map"))
        _check_dv_range(table["DV"])
        is_placeholder = False
    else:
        n = int(sv.get("placeholder_runs_per_class", 200))
        print(f"[Module Servo] 找不到原始 CSV，產生 placeholder 合成特徵表"
              f"（每類 {n} 段，共 {n * len(HEALTH_LABELS)} 列）。")
        table = generate_placeholder(n, seed=int(sv.get("random_state", 42)))
        is_placeholder = True

    feat_path = resolve(sv["processed_features"])
    table.to_parquet(feat_path, index=False)
    print(f"    -> 特徵表：{feat_path}（{table.shape[0]} 列 × {table.shape[1]} 欄）")

    # Demo CSV (shuffled full table) for the in-browser training simulator.
    demo = table.sample(frac=1.0, random_state=int(sv.get("random_state", 42))).reset_index(drop=True)
    demo_path = resolve(sv["feature_demo"])
    demo.to_csv(demo_path, index=False)
    print(f"    -> 訓練模擬器 demo：{demo_path}（{len(demo)} 列）")

    # A few sample rows for the dashboard demo (first 3 per class).
    sample = table.groupby("ylabel", group_keys=False).head(3).reset_index(drop=True)
    sample_path = resolve(sv["sample_predictions"])
    sample.to_csv(sample_path, index=False)
    print(f"    -> 樣本筆：{sample_path}（{len(sample)} 列）")

    if is_placeholder:
        print("    [!] 這是 placeholder 合成資料；下載真實 PHM 資料後請重跑本腳本與 train_servo，"
              "並把 config.yaml::servo.placeholder 設為 false。")
    return feat_path


if __name__ == "__main__":
    run()
