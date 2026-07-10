"""實驗 E4：特徵組對照 + FLAML AutoML —— 檢驗「瓶頸在特徵可分性而非樣本量」的假設。

背景：E2/E3 已確認四種過採樣與 class_weight='balanced' 等效，留出 macro-F1 全部卡在
0.755–0.759，LN↔LO 互相誤判不動。新假設：engineered 特徵組對 LN/LO 的可分性不足。

唯讀實驗：沿用 train_servo.run() 的載入與 split（見 run_e1/e2_e3），絕不重新切分、
絕不動測試集、不覆寫任何現有 artifact。

Part A（快）：LogisticRegression + class_weight='balanced'（max_iter=2000, random_state=42,
  StandardScaler pipeline），分別用 engineered(7 維) 與 full(21 維) 特徵組。
Part B：FLAML AutoML，engineered / full 各一次（task=classification, metric=macro_f1,
  time_budget=600, seed=42），只餵 train 665 段、FLAML 內部自驗，最終在留出 test 800 段評估。

執行：./.venv/Scripts/python.exe scripts/run_e4_automl_and_features.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.servo_features import FEATURE_SETS, HEALTH_LABELS, feature_set_columns
from src.servo.field_glossary import HEALTH_LABEL_ZH
from src.utils.paths import load_config, resolve

RANDOM_STATE = 42
TIME_BUDGET = 600
OUT_JSON = "outputs/metrics/e4_automl_features.json"
E2E3_JSON = "outputs/metrics/e2_e3_resampling.json"


def load_split():
    cfg = load_config()
    sv = cfg["servo"]
    feat_path = resolve(sv["processed_features"])
    if not feat_path.exists():
        raise FileNotFoundError(f"找不到 Servo 特徵表：{feat_path}")
    df = pd.read_parquet(feat_path)
    if not ("split" in df.columns and {"train", "test"} <= set(df["split"])):
        raise RuntimeError("特徵表缺少 train/test split。")
    df_tr = df[df["split"] == "train"].reset_index(drop=True)
    df_te = df[df["split"] == "test"].reset_index(drop=True)
    labels = [c for c in HEALTH_LABELS if c in set(df_tr["ylabel"])]
    return df_tr, df_te, labels


def _per_class(y_true, y_pred, labels) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {lab: {"precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)}


def _test_block(y_true, y_pred, labels) -> dict:
    return {
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                        labels=labels, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_per_class": _per_class(y_true, y_pred, labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# --- Part A：特徵組對照 ---
def part_a(feature_set: str, df_tr, df_te, labels) -> dict:
    cols = feature_set_columns(feature_set)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipe.fit(df_tr[cols], df_tr["ylabel"])
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    y_pred = pipe.predict(df_te[cols])
    res = {
        "part": "A_feature_set",
        "id": f"LR_balanced_{feature_set}",
        "feature_set": feature_set,
        "n_features": len(cols),
        "model": "logistic_regression",
        "class_weight": "balanced",
        "converged": bool(converged),
        **_test_block(df_te["ylabel"], y_pred, labels),
    }
    print(f"[Part A] {res['id']:<24} test macro-F1={res['test_macro_f1']:.3f}"
          f"  收斂={'是' if converged else '否'}")
    return res


# --- Part B：FLAML AutoML ---
def part_b(feature_set: str, df_tr, df_te, labels) -> dict:
    from flaml import AutoML

    cols = feature_set_columns(feature_set)
    X_tr, y_tr = df_tr[cols], df_tr["ylabel"]

    automl = AutoML()
    settings = dict(task="classification", metric="macro_f1",
                    time_budget=TIME_BUDGET, seed=RANDOM_STATE,
                    verbose=1, log_file_name="")
    print(f"\n[Part B] FLAML 開跑（{feature_set}, {len(cols)} 維, time_budget={TIME_BUDGET}s）…")
    automl.fit(X_train=X_tr, y_train=y_tr, **settings)

    y_pred = automl.predict(df_te[cols])
    models_tried = list(getattr(automl, "best_config_per_estimator", {}).keys())
    res = {
        "part": "B_flaml",
        "id": f"FLAML_{feature_set}",
        "feature_set": feature_set,
        "n_features": len(cols),
        "flaml": {
            "best_estimator": automl.best_estimator,
            "best_config": automl.best_config,
            "best_loss": float(automl.best_loss),
            "best_val_macro_f1": float(1.0 - automl.best_loss),
            "models_tried": models_tried,
            "candidate_list": list(automl.estimator_list),
            "time_budget_s": TIME_BUDGET,
        },
        **_test_block(df_te["ylabel"], y_pred, labels),
    }
    print(f"[Part B] FLAML_{feature_set:<12} best={automl.best_estimator}"
          f"  內部驗證 macro-F1={1 - automl.best_loss:.3f}"
          f"  test macro-F1={res['test_macro_f1']:.3f}")
    return res


def _baseline_ref() -> float:
    p = resolve(E2E3_JSON)
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d["results"]:
            if r["id"] == "E0_balanced":
                return float(r["test_macro_f1"])
    return 0.757


def print_table(results: list[dict], baseline: float) -> None:
    print("\n" + "=" * 82)
    print("實驗 E4：特徵組 + AutoML 對比（留出測試 800 段）")
    print("=" * 82)
    print(f"{'組別':<26}{'特徵組':<14}{'test macF1':<12}{'LO recall':<11}{'LN prec':<9}")
    print("-" * 82)
    print(f"{'E0_balanced (參考)':<26}{'engineered':<14}{baseline:<12.3f}{'0.500':<11}{'0.606':<9}")
    print("-" * 82)
    for r in results:
        lo = r["test_per_class"]["LO"]["recall"]
        ln = r["test_per_class"]["LN"]["precision"]
        print(f"{r['id']:<26}{r['feature_set']:<14}{r['test_macro_f1']:<12.3f}"
              f"{lo:<11.3f}{ln:<9.3f}")
    print()


def main() -> None:
    df_tr, df_te, labels = load_split()
    print(f"[E4] 訓練 {len(df_tr)} 段、留出測試 {len(df_te)} 段、類別 {labels}")
    print(f"     特徵組：engineered={len(feature_set_columns('engineered'))} 維、"
          f"full={len(feature_set_columns('full'))} 維\n")

    results = []
    for fs in ["engineered", "full"]:
        results.append(part_a(fs, df_tr, df_te, labels))
    for fs in ["engineered", "full"]:
        results.append(part_b(fs, df_tr, df_te, labels))

    baseline = _baseline_ref()
    print_table(results, baseline)

    out = {
        "experiment": "E4_automl_features",
        "eval": "holdout_test",
        "random_state": RANDOM_STATE,
        "labels": labels,
        "label_zh": {k: HEALTH_LABEL_ZH[k] for k in labels},
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "e0_balanced_ref": baseline,
        "notes": [
            "Part B（FLAML）僅以 train 665 段擬合，由 FLAML 內部驗證（資料量 <30k，"
            "FLAML 預設 5-fold CV）挑模型與超參；best_loss = 1 − 內部驗證 macro-F1。"
            "所有組別的最終比較一律以留出 test 800 段為準（test 全為真實樣本、未參與任何訓練/調參）。",
            "full 特徵組維度較高（21 維），LogisticRegression 以 max_iter=2000 擬合；"
            "converged 欄位如實記錄是否觸發 ConvergenceWarning（False=未收斂）。",
        ],
        "results": results,
    }
    out_path = resolve(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    print(f"已寫入：{OUT_JSON}")


if __name__ == "__main__":
    main()
