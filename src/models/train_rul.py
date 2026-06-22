"""Train a Remaining-Useful-Life (RUL) regressor on the IMS Set 2 table (Module B).

Step 4 of docs/MODULE_B_IMS_PLAN.md.  Reads the per-snapshot feature table built
by ``src.data.build_ims_dataset``, adds sliding-window trend features, fits a tree
regressor mapping features -> RUL (hours), and persists the model + metrics +
a health-curve prediction CSV for the dashboard.

Two rules that keep this honest:
  * **Temporal split** — the first 70% of the run trains, the last 30% tests.
    A random split would leak the future into the past on a single run.
  * **Sliding-window features** — rolling mean and slope over the last N snapshots
    capture the *trend* (e.g. kurtosis creeping up), which a single snapshot can't.

Run after the feature table exists::

    python -m src.models.train_rul
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.paths import ensure_output_dirs, load_config, resolve

LABEL_COLUMNS = ("rul_hours", "health")


def _slope(window: np.ndarray) -> float:
    """Least-squares slope of the values in a window (per snapshot step)."""
    x = np.arange(window.size, dtype=float)
    return float(np.polyfit(x, window, 1)[0])


def add_sliding_window_features(
    df: pd.DataFrame, window: int, feature_cols: List[str] | None = None
) -> pd.DataFrame:
    """Append rolling mean & slope over the last ``window`` snapshots per feature.

    The first ``window - 1`` rows have an incomplete window and are dropped, so
    the table loses a few of the earliest (healthy) snapshots — an acceptable
    trade for clean trend features.
    """
    cols = feature_cols or [c for c in df.columns if c not in LABEL_COLUMNS]
    out = df.copy()
    for c in cols:
        roll = out[c].rolling(window)
        out[f"{c}_roll_mean"] = roll.mean()
        out[f"{c}_roll_slope"] = roll.apply(_slope, raw=True)
    return out.dropna()


def temporal_split(
    df: pd.DataFrame, test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted frame into (train, test) by position — no shuffling."""
    n_test = int(round(len(df) * test_frac))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae_hours": float(mean_absolute_error(y_true, y_pred)),
        "rmse_hours": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    ims = cfg["ims"]
    random_state = cfg["modeling"]["random_state"]

    print("[1/5] 載入 IMS 特徵表...")
    features_path = resolve(ims["processed_features"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"找不到 IMS 特徵表：{features_path}\n"
            "請先放好 Set 2 資料並執行 python -m src.data.build_ims_dataset。"
        )
    df = pd.read_parquet(features_path).sort_index()
    print(f"    -> 特徵表維度 = {df.shape}")

    print(f"[2/5] 加入滑動視窗特徵（window = {ims['sliding_window']}）...")
    df = add_sliding_window_features(df, ims["sliding_window"])
    feature_cols = [c for c in df.columns if c not in LABEL_COLUMNS]
    print(f"    -> 加窗後維度 = {df.shape}（{len(feature_cols)} 個特徵）")

    print(f"[3/5] 時間切分（後 {ims['rul_test_frac']:.0%} 為測試）...")
    train_df, test_df = temporal_split(df, ims["rul_test_frac"])
    X_train, y_train = train_df[feature_cols], train_df["rul_hours"]
    X_test, y_test = test_df[feature_cols], test_df["rul_hours"]
    print(f"    -> train {X_train.shape}, test {X_test.shape}")

    print("[4/5] 訓練 RUL 回歸器（RandomForest / GradientBoosting）...")
    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }
    results: Dict[str, Dict] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        m = _metrics(y_test.values, model.predict(X_test))
        results[name] = {"model": model, "metrics": m}
        print(
            f"    - {name:<20s} MAE={m['mae_hours']:.2f}h "
            f"RMSE={m['rmse_hours']:.2f}h R2={m['r2']:.3f}"
        )

    best_name = min(results, key=lambda n: results[n]["metrics"]["mae_hours"])
    best = results[best_name]
    print(f"    -> 最佳：{best_name}（MAE 最低）")

    print("[5/5] 儲存模型、指標與健康曲線預測...")
    bundle = {
        "model": best["model"],
        "model_name": best_name,
        "feature_columns": feature_cols,
        "metrics": best["metrics"],
        "sliding_window": ims["sliding_window"],
        "target_bearing": ims["target_bearing"],
    }
    joblib.dump(bundle, resolve(ims["rul_model"]))

    with open(resolve(ims["rul_metrics"]), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "metrics": best["metrics"],
                "all_models": {n: r["metrics"] for n, r in results.items()},
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
            },
            f,
            indent=2,
        )

    # Health-curve CSV for the dashboard: predicted RUL -> predicted health,
    # using the same linear scale as the labels (health = rul / rul_max * 100).
    rul_max = float(df["rul_hours"].iloc[0])
    pred_rul = best["model"].predict(X_test)
    pd.DataFrame(
        {
            "timestamp": test_df.index,
            "rul_true": y_test.values,
            "rul_pred": pred_rul,
            "health_true": test_df["health"].values,
            "health_pred": np.clip(pred_rul / rul_max * 100.0, 0.0, 100.0),
        }
    ).to_csv(resolve(ims["rul_predictions"]), index=False)

    print(f"    -> 模型已儲存：{resolve(ims['rul_model'])}")
    print(f"    -> 指標已儲存：{resolve(ims['rul_metrics'])}")
    print(f"    -> 健康曲線預測已儲存：{resolve(ims['rul_predictions'])}")
    return resolve(ims["rul_model"])


if __name__ == "__main__":
    run()
