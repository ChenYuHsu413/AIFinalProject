"""Leave-one-bearing-out supervised RUL on XJTU-SY Condition 1 (Module B+, step 4).

The IMS single trajectory made supervised RUL impossible: the test fold's RUL
range lay entirely outside the training range, and trees cannot extrapolate
(MAE ~120 h, R^2 ~ -76; see ``src/models/train_rul.py`` / ``MODULE_B_RESULTS.md``).

With FIVE independent XJTU trajectories we can finally do this honestly: train on
4 bearings, test on the held-out 1, rotate.  Because the other bearings span the
full RUL range (0 .. ~2.7 h), the held-out bearing's targets now lie *inside* the
training range, so supervised regression no longer has to extrapolate.

The point is not a SOTA RUL score; it is to show the earlier failure was a
*data-setting* problem (single trajectory), not a flaw of supervised learning.
The reported model uses instantaneous vibration features only (the time index is
excluded, so it maps condition -> RUL rather than memorising elapsed time).
Within-bearing rolling trend features (mean + slope) are available behind the
``xjtu.lobo_use_trend`` flag but were found NOT to improve pooled cross-bearing
R^2 on these 5 bearings, so they are off by default.  Rolling windows are
backward-looking and computed per bearing (no future or cross-bearing leakage).

Run after the feature table exists::

    python -m src.models.train_rul_lobo
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.paths import ensure_output_dirs, load_config, resolve

# Columns that are labels or indices, not condition-monitoring features.
_NON_FEATURE = {"bearing", "minute", "rul_hours", "health"}


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _NON_FEATURE]


def _rolling_slope(values: np.ndarray) -> float:
    """Linear-trend slope over a 1-D window (0 for a single point)."""
    if values.size < 2:
        return 0.0
    return float(np.polyfit(np.arange(values.size), values, 1)[0])


def add_trend_features(df: pd.DataFrame, base_feats: List[str], window: int) -> pd.DataFrame:
    """Append per-bearing backward-looking rolling mean + slope for each feature.

    Computed within each ``bearing`` group, so no future or cross-bearing leakage.
    """
    out = df.copy()
    for col in base_feats:
        grp = out.groupby("bearing")[col]
        out[f"{col}_rollmean"] = grp.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        out[f"{col}_rollslope"] = grp.transform(
            lambda s: s.rolling(window, min_periods=1).apply(_rolling_slope, raw=True)
        )
    return out


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    xj = cfg["xjtu"]

    features_path = resolve(xj["processed_features"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"找不到 XJTU 特徵表：{features_path}\n"
            "請先執行 python -m src.data.build_xjtu_dataset。"
        )
    df = pd.read_parquet(features_path)
    base_feats = _feature_columns(df)
    window = xj.get("lobo_window", 5)
    use_trend = bool(xj.get("lobo_use_trend", False))
    if use_trend:
        df = add_trend_features(df, base_feats, window)
    feats = _feature_columns(df)  # base (+ rolling mean/slope trend features if enabled)
    bearings = list(xj["bearings"])
    mode = f"{len(base_feats)} 瞬時 + 趨勢 window={window}" if use_trend else f"{len(base_feats)} 瞬時"
    print(f"[1/2] LOBO 監督式 RUL：{len(bearings)} 顆軸承，特徵 {len(feats)} 維（{mode}）")

    per_bearing: List[Dict] = []
    all_true, all_pred = [], []
    pred_rows: List[Dict] = []
    for held in bearings:
        train = df[df["bearing"] != held]
        test = df[df["bearing"] == held]
        if len(test) == 0:
            continue
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(train[feats], train["rul_hours"])
        pred = model.predict(test[feats])
        yt = test["rul_hours"].to_numpy()
        mae = float(mean_absolute_error(yt, pred))
        rmse = float(np.sqrt(mean_squared_error(yt, pred)))
        r2 = float(r2_score(yt, pred))
        per_bearing.append({
            "held_out": held, "n_test": int(len(test)),
            "mae_hours": mae, "rmse_hours": rmse, "r2": r2,
        })
        all_true.append(yt)
        all_pred.append(pred)
        for m, t, p in zip(test["minute"].to_numpy(), yt, pred):
            pred_rows.append({"bearing": held, "minute": int(m),
                              "rul_true": float(t), "rul_pred": float(p)})
        print(f"    -> 留 {held}（n={len(test)}）："
              f"MAE={mae:.3f}h RMSE={rmse:.3f}h R2={r2:.3f}")

    yt_all = np.concatenate(all_true)
    yp_all = np.concatenate(all_pred)
    pooled = {
        "mae_hours": float(mean_absolute_error(yt_all, yp_all)),
        "rmse_hours": float(np.sqrt(mean_squared_error(yt_all, yp_all))),
        "r2": float(r2_score(yt_all, yp_all)),
        "mean_per_bearing_mae_hours": float(np.mean([b["mae_hours"] for b in per_bearing])),
    }
    print(f"[2/2] 合併（所有留一測試點）：MAE={pooled['mae_hours']:.3f}h "
          f"RMSE={pooled['rmse_hours']:.3f}h R2={pooled['r2']:.3f}")

    out_json = resolve(xj["lobo_metrics"])
    out_csv = resolve(xj["lobo_predictions"])
    pd.DataFrame(pred_rows).to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "leave_one_bearing_out_supervised_rul",
                "model": "RandomForestRegressor(n_estimators=300, random_state=42)",
                "trend_features": use_trend,
                "features": feats,
                "per_bearing": per_bearing,
                "pooled": pooled,
            },
            f,
            indent=2,
        )
    print(f"    -> 指標：{out_json}")
    print(f"    -> 預測：{out_csv}")
    return out_json


if __name__ == "__main__":
    run()
