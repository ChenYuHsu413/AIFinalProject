"""實驗 E1：class_weight 對伺服馬達健康分類的影響（預設 vs 'balanced'）。

沿用 train_servo.py 的資料載入與 split 邏輯（讀 config → servo_features.parquet →
以 ``split`` 欄切 train/test），不重新切分、不呼叫 run()、不覆寫 servo_clf.joblib。

對 LogisticRegression 與 RandomForestClassifier 各跑兩版：預設 vs class_weight='balanced'
（其餘超參數沿用專案工廠：LR max_iter=2000；RF n_estimators=200, n_jobs=-1），
random_state=42。每版報告：
  * 訓練集 5-fold 分層 CV 的 macro-F1（平均±標準差）
  * 留出測試 800 段的 macro-F1、per-class F1、accuracy、混淆矩陣

結果寫 outputs/metrics/e1_class_weight.json（風格對齊 servo_clf_eval.json），並印對比表。

執行：
    ./.venv/Scripts/python.exe scripts/run_e1_class_weight.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp950 主控台防亂碼
except (AttributeError, ValueError):
    pass

# 讓 scripts/ 直接執行時也找得到 src 套件
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 沿用專案模組（與 train_servo.py 相同來源）
from src.features.servo_features import HEALTH_LABELS, feature_set_columns
from src.servo.field_glossary import HEALTH_LABEL_ZH
from src.utils.paths import load_config, resolve

RANDOM_STATE = 42
CV_FOLDS = 5
OUT_JSON = Path("outputs/metrics/e1_class_weight.json")


def load_split():
    """複製 train_servo.run() 的資料載入與 split 邏輯（不重新切分）。"""
    cfg = load_config()
    sv = cfg["servo"]
    feat_path = resolve(sv["processed_features"])
    if not feat_path.exists():
        raise FileNotFoundError(
            f"找不到 Servo 特徵表：{feat_path}\n請先執行 python -m src.data.build_servo_dataset。"
        )
    import pandas as pd
    df = pd.read_parquet(feat_path)
    # 本實驗系列（E1/E2/E3）在 engineered 空間進行，pin 死以自包含實驗條件——
    # 勿隨 config 的 reference_feature_set 變動（2026-07-10 已轉 full），否則重跑
    # 會與 e1_class_weight.json 的存檔結果對不上，破壞重現性。
    feature_set = "engineered"
    cols = feature_set_columns(feature_set)

    has_split = "split" in df.columns and {"train", "test"} <= set(df["split"])
    if not has_split:
        raise RuntimeError("特徵表缺少 train/test split；E1 需要留出測試集。")
    df_tr = df[df["split"] == "train"].reset_index(drop=True)
    df_te = df[df["split"] == "test"].reset_index(drop=True)
    labels = [c for c in HEALTH_LABELS if c in set(df_tr["ylabel"])]
    return df_tr, df_te, cols, feature_set, labels


def make_pipeline(model: str, balanced: bool) -> Pipeline:
    """只切換 class_weight，其餘超參數沿用專案工廠設定；同樣包 StandardScaler。"""
    cw = "balanced" if balanced else None
    if model == "logistic_regression":
        est = LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE)
    elif model == "random_forest":
        est = RandomForestClassifier(
            n_estimators=200, class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE)
    else:
        raise ValueError(model)
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def evaluate(model: str, balanced: bool, df_tr, df_te, cols, labels) -> dict:
    X_tr, y_tr = df_tr[cols], df_tr["ylabel"]
    X_te, y_te = df_te[cols], df_te["ylabel"]

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(make_pipeline(model, balanced), X_tr, y_tr,
                         cv=skf, scoring="f1_macro", n_jobs=-1)

    pipe = make_pipeline(model, balanced).fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    per_class = f1_score(y_te, y_pred, average=None, labels=labels, zero_division=0)

    return {
        "model": model,
        "class_weight": "balanced" if balanced else None,
        "variant": "balanced" if balanced else "default",
        "cv_macro_f1_mean": float(cv.mean()),
        "cv_macro_f1_std": float(cv.std()),
        "test_macro_f1": float(f1_score(y_te, y_pred, average="macro",
                                        labels=labels, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_te, y_pred)),
        "test_per_class_f1": {lab: float(v) for lab, v in zip(labels, per_class)},
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=labels).tolist(),
    }


def print_table(results: list[dict], labels: list[str]) -> None:
    zh = {lab: HEALTH_LABEL_ZH.get(lab, lab) for lab in labels}
    name = {"logistic_regression": "LogReg", "random_forest": "RandForest"}

    print("=" * 78)
    print("實驗 E1：class_weight 對比（5-fold 分層 CV / 留出測試 800 段）")
    print("=" * 78)
    header = (f"{'模型':<12}{'版本':<10}{'CV macro-F1 (μ±σ)':<22}"
              f"{'Test macro-F1':<15}{'Test Acc':<10}")
    print(header)
    print("-" * 78)
    for r in results:
        cv = f"{r['cv_macro_f1_mean']:.3f}±{r['cv_macro_f1_std']:.3f}"
        print(f"{name[r['model']]:<12}{r['variant']:<10}{cv:<22}"
              f"{r['test_macro_f1']:<15.3f}{r['test_accuracy']:<10.3f}")

    print("\nper-class F1（留出測試）：")
    head = f"{'模型/版本':<24}" + "".join(f"{lab}({zh[lab]})"[:11].ljust(12) for lab in labels)
    print(head)
    print("-" * 78)
    for r in results:
        row = f"{name[r['model']] + '/' + r['variant']:<24}"
        row += "".join(f"{r['test_per_class_f1'][lab]:<12.3f}" for lab in labels)
        print(row)

    # 每個模型 balanced 相對 default 的 Δ（重點看少數類 LO）
    print("\nΔ（balanced − default）：")
    for model in ["logistic_regression", "random_forest"]:
        d = next(r for r in results if r["model"] == model and r["variant"] == "default")
        b = next(r for r in results if r["model"] == model and r["variant"] == "balanced")
        dmac = b["test_macro_f1"] - d["test_macro_f1"]
        parts = [f"{lab} {b['test_per_class_f1'][lab] - d['test_per_class_f1'][lab]:+.3f}"
                 for lab in labels]
        print(f"  {name[model]:<12} macro-F1 {dmac:+.3f} | " + "  ".join(parts))
    print()


def main() -> None:
    df_tr, df_te, cols, feature_set, labels = load_split()
    print(f"[E1] 特徵組 {feature_set}（{len(cols)} 維）、訓練 {len(df_tr)} 段、"
          f"留出測試 {len(df_te)} 段、類別 {labels}\n")

    results = []
    for model in ["logistic_regression", "random_forest"]:
        for balanced in (False, True):
            results.append(evaluate(model, balanced, df_tr, df_te, cols, labels))

    print_table(results, labels)

    out = {
        "experiment": "E1_class_weight",
        "feature_set": feature_set,
        "eval": "holdout_test",
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "labels": labels,
        "label_zh": {k: HEALTH_LABEL_ZH[k] for k in labels},
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "results": results,
    }
    out_path = resolve(str(OUT_JSON))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已寫入：{OUT_JSON}")


if __name__ == "__main__":
    main()
