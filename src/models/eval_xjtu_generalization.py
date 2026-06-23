"""Cross-bearing RUL/FPT generalization for XJTU-SY Condition 1 (Module B+).

Applies the SAME fixed-parameter health-indicator -> FPT -> trend-extrapolation
pipeline used for the IMS single trajectory (``src/models/rul_extrapolation.py``)
to each of the 5 Condition-1 bearings, WITHOUT per-bearing tuning.  This is the
cross-trajectory generalization evidence that IMS Set 2's single bearing cannot
provide: one method, one parameter set, validated on 5 independent run-to-failure
bearings.

Run after the feature table exists::

    python -m src.models.eval_xjtu_generalization
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.rul_extrapolation import (
    _metrics,
    build_health_indicator,
    detect_fpt,
    extrapolate_rul,
)
from src.utils.paths import ensure_output_dirs, load_config, resolve


def _evaluate_bearing(sub: pd.DataFrame, xj: dict) -> Dict:
    """Run the fixed-parameter FPT/RUL pipeline on one bearing's feature rows."""
    sub = sub.sort_values("minute").reset_index(drop=True)
    indicator = xj["health_indicator"]
    interval = xj["snapshot_interval_min"]

    hi, health, hi_base, hi_fail = build_health_indicator(
        sub[indicator], xj["hi_smooth_window"], xj["baseline_n"], xj["fail_percentile"],
    )
    fpt_idx = detect_fpt(hi, xj["baseline_n"], xj["fpt_n_sigma"], xj["fpt_consecutive"])

    minutes = sub["minute"].to_numpy()
    hours = (minutes - minutes[0]) * interval / 60.0
    rul_true = sub["rul_hours"].to_numpy()
    rul_pred = extrapolate_rul(
        hours, hi.to_numpy(), hi_base, hi_fail, fpt_idx,
        xj["min_fit_points"], xj["fit_window"], float(hours[-1]),
    )

    if np.isfinite(rul_pred).sum() == 0:
        m = {"mae_hours": float("nan"), "rmse_hours": float("nan"),
             "r2": float("nan"), "n_eval": 0}
    else:
        m = _metrics(rul_true, rul_pred)

    lead_hours = float(hours[-1] - hours[fpt_idx])
    summary = {
        "bearing": sub["bearing"].iloc[0],
        "n_snapshots": int(len(sub)),
        "life_hours": float(hours[-1]),
        "fpt_index": int(fpt_idx),
        "lead_time_hours": lead_hours,
        "lead_frac_of_life": lead_hours / float(hours[-1]) if hours[-1] > 0 else 0.0,
        "mae_hours": m["mae_hours"],
        "rmse_hours": m["rmse_hours"],
        "n_eval": m["n_eval"],
        "hi_base": hi_base,
        "hi_fail": hi_fail,
    }
    curve = pd.DataFrame({
        "bearing": sub["bearing"],
        "minute": minutes,
        "rul_true": rul_true,
        "rul_pred": rul_pred,
        "health": health.to_numpy(),
        "is_degrading": np.arange(len(sub)) >= fpt_idx,
    })
    return {"summary": summary, "curve": curve}


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    xj = cfg["xjtu"]

    print("[1/3] 載入 XJTU 特徵表...")
    features_path = resolve(xj["processed_features"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"找不到 XJTU 特徵表：{features_path}\n"
            "請先執行 python -m src.data.build_xjtu_dataset。"
        )
    df = pd.read_parquet(features_path)
    print(f"    -> 特徵表維度 = {df.shape}，{df['bearing'].nunique()} 顆軸承")

    print(f"[2/3] 對每顆軸承套用固定參數 FPT/RUL（indicator={xj['health_indicator']}）...")
    summaries: List[Dict] = []
    curves: List[pd.DataFrame] = []
    for bearing, sub in df.groupby("bearing", sort=True):
        res = _evaluate_bearing(sub, xj)
        summaries.append(res["summary"])
        curves.append(res["curve"])
        s = res["summary"]
        print(f"    -> {bearing}: FPT@{s['fpt_index']}/{s['n_snapshots']}  "
              f"提前 {s['lead_time_hours']:.2f}h ({s['lead_frac_of_life'] * 100:.0f}% 壽命)  "
              f"MAE={s['mae_hours']:.3f}h (n={s['n_eval']})")

    summary_df = pd.DataFrame(summaries)
    aggregate = {
        "n_bearings": int(len(summary_df)),
        "mean_lead_time_hours": float(summary_df["lead_time_hours"].mean()),
        "mean_mae_hours": float(summary_df["mae_hours"].mean()),
        "fixed_params": {
            k: xj[k] for k in (
                "health_indicator", "hi_smooth_window", "baseline_n", "fpt_n_sigma",
                "fpt_consecutive", "fail_percentile", "min_fit_points", "fit_window",
            )
        },
    }

    print("[3/3] 儲存彙總與曲線...")
    summary_df.to_csv(resolve(xj["gen_summary"]), index=False)
    pd.concat(curves, ignore_index=True).to_csv(resolve(xj["rul_predictions"]), index=False)
    with open(resolve(xj["gen_metrics"]), "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "fixed_param_trend_extrapolation_per_bearing",
                "per_bearing": summaries,
                "aggregate": aggregate,
            },
            f,
            indent=2,
        )
    print(f"    -> 彙總：{resolve(xj['gen_summary'])}")
    print(f"    -> 曲線：{resolve(xj['rul_predictions'])}")
    print(f"    -> 指標：{resolve(xj['gen_metrics'])}")
    print(f"    == 5 顆平均：提前 {aggregate['mean_lead_time_hours']:.2f}h，"
          f"MAE {aggregate['mean_mae_hours']:.3f}h ==")
    return resolve(xj["gen_summary"])


if __name__ == "__main__":
    run()
