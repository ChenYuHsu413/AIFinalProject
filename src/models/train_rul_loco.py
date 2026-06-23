"""Leave-one-condition-out supervised RUL on XJTU-SY (Module B+, cross-condition).

The harder cross-operating-condition test: train on bearings from two operating
conditions, test on the held-out condition's bearings, rotating over all three.
Unlike leave-one-bearing-out (same conditions appear in train and test), here the
test condition's speed/load is *unseen* in training, so any drop reveals
operating-condition **domain shift** — an honest limit of condition monitoring
across regimes.

Reuses the same feature set and model as ``train_rul_lobo.py`` for a fair
contrast (leave-one-bearing-out vs leave-one-condition-out).

Run after the feature table exists::

    python -m src.models.train_rul_loco
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.train_rul_lobo import _feature_columns
from src.utils.paths import ensure_output_dirs, load_config, resolve


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
    feats = _feature_columns(df)
    conditions = [c["name"] for c in xj["conditions"]]
    print(f"[1/2] LOCO 監督式 RUL：{len(conditions)} 工況留一，特徵 {len(feats)} 維")

    per_condition: List[Dict] = []
    all_true, all_pred = [], []
    pred_rows: List[Dict] = []
    for held in conditions:
        train = df[df["condition"] != held]
        test = df[df["condition"] == held]
        if len(test) == 0:
            continue
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(train[feats], train["rul_hours"])
        pred = model.predict(test[feats])
        yt = test["rul_hours"].to_numpy()
        mae = float(mean_absolute_error(yt, pred))
        rmse = float(np.sqrt(mean_squared_error(yt, pred)))
        r2 = float(r2_score(yt, pred))
        per_condition.append({
            "held_out_condition": held,
            "n_test_bearings": int(test["bearing"].nunique()),
            "n_test": int(len(test)),
            "mae_hours": mae, "rmse_hours": rmse, "r2": r2,
        })
        all_true.append(yt)
        all_pred.append(pred)
        for b, m, t, p in zip(test["bearing"], test["minute"].to_numpy(), yt, pred):
            pred_rows.append({"condition": held, "bearing": b, "minute": int(m),
                              "rul_true": float(t), "rul_pred": float(p)})
        print(f"    -> 留 {held}（{test['bearing'].nunique()} 顆, n={len(test)}）："
              f"MAE={mae:.3f}h RMSE={rmse:.3f}h R2={r2:.3f}")

    yt_all = np.concatenate(all_true)
    yp_all = np.concatenate(all_pred)
    pooled = {
        "mae_hours": float(mean_absolute_error(yt_all, yp_all)),
        "rmse_hours": float(np.sqrt(mean_squared_error(yt_all, yp_all))),
        "r2": float(r2_score(yt_all, yp_all)),
        "mean_per_condition_mae_hours": float(np.mean([c["mae_hours"] for c in per_condition])),
    }
    print(f"[2/2] 合併（所有留一工況測試點）：MAE={pooled['mae_hours']:.3f}h "
          f"RMSE={pooled['rmse_hours']:.3f}h R2={pooled['r2']:.3f}")

    out_json = resolve(xj["loco_metrics"])
    out_csv = resolve(xj["loco_predictions"])
    pd.DataFrame(pred_rows).to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "leave_one_condition_out_supervised_rul",
                "model": "RandomForestRegressor(n_estimators=300, random_state=42)",
                "features": feats,
                "per_condition": per_condition,
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
