"""Trend-extrapolation RUL for the IMS single run-to-failure trajectory (Module B).

A tree regressor cannot forecast RUL on one bearing: the test fold's target range
(0..49 h) lies entirely below the training range (50..164 h), and trees cannot
extrapolate a monotone target.  This module uses the standard PHM approach instead:

  1. Build a data-driven **health indicator** (HI) from a vibration feature
     (default ``b1_rms``), smoothed.
  2. Detect the **First Predicting Time (FPT)** — degradation onset — when the HI
     first exceeds ``baseline_mean + n*std`` for several consecutive snapshots.
  3. After FPT, fit a straight line to the HI trend over ``[FPT, t]`` and
     **extrapolate to the failure threshold** to estimate the failure time, hence
     RUL(t) = t_fail_est - t.

RUL is only predicted (and only evaluated) in the post-FPT degradation region,
which is the only region where forecasting is meaningful.

Run after the feature table exists::

    python -m src.models.rul_extrapolation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.paths import ensure_output_dirs, load_config, resolve


def build_health_indicator(
    raw_hi: pd.Series, smooth_window: int, baseline_n: int, fail_percentile: float
) -> Tuple[pd.Series, pd.Series, float, float]:
    """Smooth the indicator and map it to a 0..100 health score.

    Returns ``(hi_smooth, health, hi_base, hi_fail)`` where ``hi_base`` is the
    healthy baseline (high health) and ``hi_fail`` is the failure level.  The
    indicator is assumed to *increase* toward failure (true for RMS on an
    outer-race defect), so health = 100 at baseline and 0 at the failure level.
    """
    hi = raw_hi.rolling(smooth_window, min_periods=1).median()
    hi_base = float(hi.iloc[:baseline_n].mean())
    hi_fail = float(np.percentile(hi.values, fail_percentile))
    span = max(hi_fail - hi_base, 1e-9)
    health = ((hi_fail - hi) / span * 100.0).clip(0.0, 100.0)
    return hi, health, hi_base, hi_fail


def detect_fpt(
    hi: pd.Series, baseline_n: int, n_sigma: float, consecutive: int
) -> int:
    """Return the integer index of degradation onset (First Predicting Time).

    Onset = the first snapshot whose smoothed indicator stays above
    ``baseline_mean + n_sigma * baseline_std`` for ``consecutive`` points.
    Falls back to the baseline window's end if no onset is detected.
    """
    base = hi.iloc[:baseline_n]
    threshold = base.mean() + n_sigma * base.std()
    above = (hi.values > threshold).astype(int)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= consecutive:
            return i - consecutive + 1
    return baseline_n


def extrapolate_rul(
    hours: np.ndarray, hi: np.ndarray, hi_base: float, hi_fail: float,
    fpt_idx: int, min_points: int, window: int, max_life: float,
) -> np.ndarray:
    """Estimate RUL (hours) at each post-FPT snapshot by exponential extrapolation.

    Bearing degradation accelerates roughly exponentially, so a straight-line fit
    over the whole post-FPT region wildly over-shoots the failure time early on.
    Instead we model ``HI - hi_base ~ exp(k*t + b)`` and fit it on a *rolling*
    window of the most recent ``window`` points: as degradation accelerates, the
    local slope ``k`` steepens and the predicted failure time converges.

    Pre-FPT and non-degrading (k <= 0) points are left as NaN.
    """
    rul = np.full(hi.size, np.nan)
    log_excess = np.log(np.clip(hi - hi_base, 1e-6, None))
    log_fail = np.log(max(hi_fail - hi_base, 1e-6))
    for t in range(fpt_idx + min_points - 1, hi.size):
        lo = max(fpt_idx, t - window + 1)
        k, b = np.polyfit(hours[lo : t + 1], log_excess[lo : t + 1], 1)
        if k <= 0:
            continue
        t_fail = (log_fail - b) / k
        # A bearing cannot have more RUL than its design life — clip away the
        # explosive estimates that a shallow local slope would otherwise produce.
        rul[t] = float(np.clip(t_fail - hours[t], 0.0, max_life))
    return rul


def _metrics(rul_true: np.ndarray, rul_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(rul_pred)
    yt, yp = rul_true[mask], rul_pred[mask]
    return {
        "mae_hours": float(mean_absolute_error(yt, yp)),
        "rmse_hours": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "n_eval": int(mask.sum()),
    }


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    ims = cfg["ims"]

    print("[1/4] 載入 IMS 特徵表...")
    features_path = resolve(ims["processed_features"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"找不到 IMS 特徵表：{features_path}\n"
            "請先執行 python -m src.data.build_ims_dataset。"
        )
    df = pd.read_parquet(features_path).sort_index()
    print(f"    -> 特徵表維度 = {df.shape}")

    print(f"[2/4] 建立健康指標（{ims['health_indicator']}）與偵測退化起點 FPT...")
    hi, health, hi_base, hi_fail = build_health_indicator(
        df[ims["health_indicator"]], ims["hi_smooth_window"],
        ims["baseline_n"], ims["fail_percentile"],
    )
    fpt_idx = detect_fpt(hi, ims["baseline_n"], ims["fpt_n_sigma"], ims["fpt_consecutive"])
    fpt_time = df.index[fpt_idx]
    lead_days = (df.index[-1] - fpt_time).total_seconds() / 86400.0
    print(f"    -> 基線 HI={hi_base:.4f}，失效門檻 HI={hi_fail:.4f}")
    print(f"    -> FPT = 第 {fpt_idx} 個快照（{fpt_time}），距失效 {lead_days:.1f} 天")

    print("[3/4] 對退化趨勢外推 RUL（指數模型 + 滾動視窗）...")
    hours = (df.index - df.index[0]).total_seconds().to_numpy() / 3600.0
    rul_true = df["rul_hours"].to_numpy()
    rul_pred = extrapolate_rul(
        hours, hi.to_numpy(), hi_base, hi_fail, fpt_idx,
        ims["min_fit_points"], ims["fit_window"], float(hours[-1]),
    )
    m = _metrics(rul_true, rul_pred)
    near_n = ims["near_failure_n"]
    near = _metrics(rul_true[-near_n:], rul_pred[-near_n:])
    print(f"    -> 退化區評估（{m['n_eval']} 點）："
          f"MAE={m['mae_hours']:.2f}h RMSE={m['rmse_hours']:.2f}h R2={m['r2']:.3f}")
    print(f"    -> 近失效區（最後 {near_n} 筆）："
          f"MAE={near['mae_hours']:.2f}h RMSE={near['rmse_hours']:.2f}h R2={near['r2']:.3f}")

    print("[4/4] 儲存指標與健康/RUL 曲線...")
    pd.DataFrame(
        {
            "timestamp": df.index,
            "rul_true": rul_true,
            "rul_pred": rul_pred,
            "health": health.to_numpy(),
            "is_degrading": np.arange(len(df)) >= fpt_idx,
        }
    ).to_csv(resolve(ims["rul_predictions"]), index=False)

    with open(resolve(ims["rul_metrics"]), "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "trend_extrapolation",
                "indicator": ims["health_indicator"],
                "metrics": m,
                "near_failure_metrics": near,
                "fpt_index": int(fpt_idx),
                "fpt_time": str(fpt_time),
                "lead_time_days": lead_days,
                "hi_baseline": hi_base,
                "hi_failure": hi_fail,
            },
            f,
            indent=2,
        )
    print(f"    -> 指標已儲存：{resolve(ims['rul_metrics'])}")
    print(f"    -> 健康/RUL 曲線已儲存：{resolve(ims['rul_predictions'])}")
    return resolve(ims["rul_predictions"])


if __name__ == "__main__":
    run()
