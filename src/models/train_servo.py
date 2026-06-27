"""Train the Module Servo reference models (offline).

Produces the artifacts the server loads for inference (it never trains large
models online):
  * ``servo.clf_model``      — health-state classifier bundle
  * ``servo.reg_model``      — DV regressor bundle
  * ``servo.feature_config`` — feature set / columns / label map / healthy
                               baseline stats / DV risk bands / metrics
  * ``servo.clf_metrics`` / ``servo.reg_metrics`` — eval JSON for the dashboard

The classifier is chosen by stratified-CV macro-F1 across ``servo.enabled_models``;
the regressor uses Random Forest (robust default for the DV target).

Run (after ``python -m src.data.build_servo_dataset``)::

    python -m src.models.train_servo
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from src.features.servo_features import HEALTH_LABELS, feature_set_columns
from src.models.servo_simulator import build_classifier, build_regressor
from src.servo.field_glossary import HEALTH_LABEL_ZH
from src.utils.paths import ensure_output_dirs, load_config, resolve


def _healthy_baseline(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Per-feature mean/std over healthy (LN) runs -> z-score for top features."""
    ln = df[df["ylabel"] == "LN"]
    base = {}
    for c in cols:
        mu = float(ln[c].mean()) if len(ln) else float(df[c].mean())
        sd = float(ln[c].std()) if len(ln) else float(df[c].std())
        base[c] = {"mean": mu, "std": sd if sd > 1e-9 else 1.0}
    return base


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    sv = cfg["servo"]
    rs = int(sv.get("random_state", 42))

    feat_path = resolve(sv["processed_features"])
    if not feat_path.exists():
        raise FileNotFoundError(
            f"找不到 Servo 特徵表：{feat_path}\n請先執行 python -m src.data.build_servo_dataset。"
        )
    df = pd.read_parquet(feat_path)
    feature_set = sv.get("reference_feature_set", "engineered")
    cols = feature_set_columns(feature_set)
    labels = [c for c in HEALTH_LABELS if c in set(df["ylabel"])]
    print(f"[Servo] 特徵組 {feature_set}（{len(cols)} 維）、{len(df)} 段、類別 {labels}")

    # --- classifier: pick best by stratified-CV macro-F1 ---
    X, y = df[cols], df["ylabel"]
    min_class = int(y.value_counts().min())
    if min_class < 2:
        rare = sorted(y.value_counts()[lambda s: s < 2].index.tolist())
        raise ValueError(
            f"類別 {rare} 僅有 1 段，無法做分層交叉驗證。"
            "請確認真實資料每個健康類別至少有 2 段，或調整 ylabel_map / 聚合粒度。"
        )
    n_splits = max(2, min(int(sv.get("cv_folds", 5)), min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    per_model: Dict[str, float] = {}
    best_name, best_macro = None, -np.inf
    for name in sv.get("enabled_models", ["random_forest"]):
        try:
            scores = cross_val_score(
                build_classifier(name, rs), X, y, cv=skf,
                scoring="f1_macro", n_jobs=-1)
        except Exception as e:  # pragma: no cover
            print(f"    {name:>20}: 跳過（{type(e).__name__}）")
            continue
        macro = float(scores.mean())
        per_model[name] = macro
        print(f"    {name:>20}: CV macro-F1={macro:.3f}")
        if macro > best_macro:
            best_name, best_macro = name, macro
    if best_name is None:
        raise RuntimeError("沒有可用的分類器。")

    y_cv = cross_val_predict(build_classifier(best_name, rs), X, y, cv=skf)
    clf_eval = {
        "model": best_name,
        "feature_set": feature_set,
        "accuracy": float(accuracy_score(y, y_cv)),
        "macro_f1": float(f1_score(y, y_cv, average="macro", labels=labels, zero_division=0)),
        "labels": labels,
        "label_zh": {k: HEALTH_LABEL_ZH[k] for k in labels},
        "confusion_matrix": confusion_matrix(y, y_cv, labels=labels).tolist(),
        "per_model_macro_f1": per_model,
        "n": int(len(y)),
        "placeholder": bool(sv.get("placeholder", True)),
    }
    clf_pipe = build_classifier(best_name, rs).fit(X, y)

    # --- regressor for DV (held-out CV predictions for honest metrics) ---
    reg_name = "random_forest"
    yv = df["DV"]
    from sklearn.model_selection import KFold, cross_val_predict as _cvp
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    dv_cv = _cvp(build_regressor(reg_name, rs), X, yv, cv=kf)
    reg_eval = {
        "model": reg_name,
        "feature_set": feature_set,
        "mae": float(mean_absolute_error(yv, dv_cv)),
        "rmse": float(np.sqrt(np.mean((yv.to_numpy() - dv_cv) ** 2))),
        "r2": float(r2_score(yv, dv_cv)),
        "n": int(len(yv)),
        "placeholder": bool(sv.get("placeholder", True)),
    }
    reg_pipe = build_regressor(reg_name, rs).fit(X, yv)
    print(f"    回歸 {reg_name}: MAE={reg_eval['mae']:.3f} R²={reg_eval['r2']:.3f}")

    # --- persist artifacts (compress=3: keeps the RF regressor well under
    #     GitHub's 50 MB file recommendation; ~96 MB -> ~29 MB) ---
    joblib.dump({"pipeline": clf_pipe, "feature_columns": cols, "labels": labels,
                 "model_name": best_name, "metrics": clf_eval},
                resolve(sv["clf_model"]), compress=3)
    joblib.dump({"pipeline": reg_pipe, "feature_columns": cols,
                 "model_name": reg_name, "metrics": reg_eval},
                resolve(sv["reg_model"]), compress=3)

    feature_config = {
        "feature_set": feature_set,
        "feature_columns": cols,
        "labels": labels,
        "label_zh": {k: HEALTH_LABEL_ZH[k] for k in labels},
        "dv_risk": sv.get("dv_risk", {"low_max": 0.33, "medium_max": 0.66}),
        "healthy_baseline": _healthy_baseline(df, cols),
        "clf_model": best_name,
        "reg_model": reg_name,
        "clf_macro_f1": clf_eval["macro_f1"],
        "reg_r2": reg_eval["r2"],
        "placeholder": bool(sv.get("placeholder", True)),
    }
    resolve(sv["feature_config"]).write_text(
        json.dumps(feature_config, indent=2, ensure_ascii=False), encoding="utf-8")
    resolve(sv["clf_metrics"]).write_text(
        json.dumps(clf_eval, indent=2, ensure_ascii=False), encoding="utf-8")
    resolve(sv["reg_metrics"]).write_text(
        json.dumps(reg_eval, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Servo] 最佳分類器 {best_name}（macro-F1={clf_eval['macro_f1']:.3f}）"
          f"；模型/設定已寫入 outputs/。")
    if sv.get("placeholder", True):
        print("    [!] 以 placeholder 合成資料訓練；下載真實 PHM 後請重訓並設 placeholder=false。")
    return resolve(sv["clf_model"])


if __name__ == "__main__":
    run()
