"""Paderborn fault classification + artificial->real generalization (Module C).

Two evaluations on the same feature table, echoing Module B+'s LOBO-vs-LOCO:

  1. **baseline** — stratified CV over healthy + ARTIFICIAL-fault measurements
     (the easy, in-distribution number); picks the best estimator by macro-F1.
  2. **artificial->real** — train the best estimator on healthy + artificial,
     test on the held-out REAL accelerated-lifetime damage.  The gap to the
     baseline quantifies the artificial->real domain shift (the headline).

Reuses the Module A registry / transformer.  Run after the feature table exists::

    python -m src.models.train_paderborn
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data.preprocess import make_column_transformer
from src.models.model_registry import NEEDS_SCALING, available_models, build
from src.utils.paths import ensure_output_dirs, load_config, resolve

LABELS = ["healthy", "outer", "inner"]  # fixed order for confusion matrices
_LABEL_COLS = {"condition", "bearing_code", "fault_class", "damage_origin", "measurement"}


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Numeric feature columns = everything that is not a label / index column."""
    return [c for c in df.columns if c not in _LABEL_COLS]


def split_artificial_real(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train = healthy + artificial, test = real-damage)."""
    train = df[df["damage_origin"].isin(["healthy", "artificial"])].reset_index(drop=True)
    real = df[df["damage_origin"] == "real"].reset_index(drop=True)
    return train, real


def _make_pipe(name: str, feats: List[str], random_state: int) -> Pipeline:
    ct = make_column_transformer(feats, [], scale_numeric=name in NEEDS_SCALING)
    return Pipeline([("pre", ct), ("clf", build(name, random_state))])


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    pb = cfg["paderborn"]

    features_path = resolve(pb["processed_features"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"找不到 Paderborn 特徵表：{features_path}\n"
            "請先執行 python -m src.data.build_paderborn_dataset。"
        )
    df = pd.read_parquet(features_path)
    feats = feature_columns(df)
    train_df, real_df = split_artificial_real(df)
    rs = int(pb.get("random_state", 42))
    models = available_models(pb.get("enabled_models", ["random_forest"]))
    print(f"[Module C] 特徵 {len(feats)} 維；訓練(健康+人工) {len(train_df)} 列、"
          f"真實損傷測試 {len(real_df)} 列；候選模型 {models}")

    X_tr, y_tr = train_df[feats], train_df["fault_class"]
    min_class = int(y_tr.value_counts().min())
    n_splits = max(2, min(int(pb.get("cv_folds", 5)), min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)

    # --- 1) baseline: stratified CV over healthy + artificial, pick best model ---
    print(f"[1] baseline 分層 CV（{n_splits} 折）...")
    per_model: Dict[str, float] = {}
    best_name, best_macro = None, -np.inf
    for name in models:
        try:
            y_cv = cross_val_predict(_make_pipe(name, feats, rs), X_tr, y_tr, cv=skf)
        except Exception as e:  # e.g. a binary-only estimator on a 3-class target
            print(f"    {name:>20}: 跳過（{type(e).__name__}）")
            continue
        macro = float(f1_score(y_tr, y_cv, average="macro"))
        per_model[name] = macro
        print(f"    {name:>20}: macro-F1={macro:.3f}")
        if macro > best_macro:
            best_name, best_macro = name, macro
    if best_name is None:
        raise RuntimeError(
            f"沒有任何候選模型可用於多類別分類（嘗試了 {models}）。"
        )

    y_cv_best = cross_val_predict(_make_pipe(best_name, feats, rs), X_tr, y_tr, cv=skf)
    baseline = {
        "model": best_name,
        "accuracy": float(accuracy_score(y_tr, y_cv_best)),
        "macro_f1": float(f1_score(y_tr, y_cv_best, average="macro")),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_tr, y_cv_best, labels=LABELS).tolist(),
        "per_model_macro_f1": per_model,
        "n": int(len(y_tr)),
    }

    # --- 2) artificial -> real generalization with the chosen model ---
    best_pipe = _make_pipe(best_name, feats, rs).fit(X_tr, y_tr)
    generalization, pred_rows = None, []
    if len(real_df):
        y_real = real_df["fault_class"]
        y_pred = best_pipe.predict(real_df[feats])
        generalization = {
            "model": best_name,
            "accuracy": float(accuracy_score(y_real, y_pred)),
            "macro_f1": float(f1_score(y_real, y_pred, average="macro",
                                       labels=LABELS, zero_division=0)),
            "labels": LABELS,
            "confusion_matrix": confusion_matrix(y_real, y_pred, labels=LABELS).tolist(),
            "n": int(len(y_real)),
        }
        for code, cond, m, yt, yp in zip(real_df["bearing_code"], real_df["condition"],
                                         real_df["measurement"], y_real, y_pred):
            pred_rows.append({"bearing_code": code, "condition": cond, "measurement": int(m),
                              "y_true": yt, "y_pred": yp})
        print(f"[2] artificial->real（{best_name}）：baseline macro-F1={baseline['macro_f1']:.3f}"
              f" -> real macro-F1={generalization['macro_f1']:.3f}")
    else:
        print("[2] 無真實損傷測試資料（config 未配置 real_* 或檔案缺失），跳過泛化評估。")

    summary = {
        "best_model": best_name,
        "baseline_macro_f1": baseline["macro_f1"],
        "generalization_macro_f1": (generalization["macro_f1"] if generalization else None),
        "gap": (baseline["macro_f1"] - generalization["macro_f1"]) if generalization else None,
    }

    # --- persist model + metrics + real-set predictions ---
    joblib.dump(
        {"pipeline": best_pipe, "feature_columns": feats, "labels": LABELS,
         "model_name": best_name},
        resolve(pb["best_model"]),
    )
    out_json = resolve(pb["metrics"])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "paderborn_fault_classification_artificial_to_real",
                "features": feats,
                "results": {"baseline": baseline, "artificial_to_real": generalization},
                "summary": summary,
            },
            f, indent=2,
        )
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(resolve(pb["predictions"]), index=False)
    print(f"    -> 指標：{out_json}")
    print(f"    -> 模型：{resolve(pb['best_model'])}")
    return out_json


if __name__ == "__main__":
    run()
