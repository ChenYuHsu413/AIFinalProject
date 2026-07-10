"""實驗 E2/E3：過採樣（SMOTE 家族）對伺服馬達健康分類 LO 少數類的影響。

背景：訓練集 LO 僅 65 段（其他三類各 ~200），test 四類均衡各 200。
E1 已知 LogisticRegression + class_weight='balanced' 留出 macro-F1 = 0.757，
default（不加權）= 0.636。本實驗只用 LR，比較「重採樣取代/疊加加權」。

唯讀實驗：沿用 train_servo.run() 的資料載入與 split 邏輯（見 run_e1_class_weight.py），
絕不重新切分、絕不動測試集、不覆寫任何現有 artifact。

10 組（全部 LogisticRegression, max_iter=2000, random_state=42）：
  對照組：E0_default (cw=None) / E0_balanced (cw='balanced')
  重採樣 × {cw=None, cw='balanced'}：SMOTE / BorderlineSMOTE / ADASYN / SMOTETomek
  皆把 LO 補到 200（sampling_strategy={'LO':200}；ADASYN/SMOTETomek 數量可能浮動）。

關鍵防洩漏：CV 把「整個 imblearn Pipeline」交給 StratifiedKFold，重採樣只發生在
每個 fold 的訓練部分；留出評估則在完整 train 上 fit、對 test 預測。

執行：./.venv/Scripts/python.exe scripts/run_e2_e3_resampling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp950 主控台防亂碼
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 讓 scripts/ 找得到 src

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.features.servo_features import HEALTH_LABELS, feature_set_columns
from src.servo.field_glossary import HEALTH_LABEL_ZH
from src.utils.paths import load_config, resolve

RANDOM_STATE = 42
CV_FOLDS = 5
LO_TARGET = 200
OUT_JSON = "outputs/metrics/e2_e3_resampling.json"
OUT_FIG = "outputs/figures/e2_e3_test_macro_f1.png"

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# --- 資料載入與 split（複用 train_servo.run() 同套流程）---
def load_split():
    cfg = load_config()
    sv = cfg["servo"]
    feat_path = resolve(sv["processed_features"])
    if not feat_path.exists():
        raise FileNotFoundError(f"找不到 Servo 特徵表：{feat_path}")
    df = pd.read_parquet(feat_path)
    # 本實驗系列（E1/E2/E3）在 engineered 空間進行，pin 死以自包含實驗條件——
    # 勿隨 config 的 reference_feature_set 變動（2026-07-10 已轉 full），否則重跑
    # 會與 e2_e3_resampling.json 的存檔結果對不上，破壞重現性。
    feature_set = "engineered"
    cols = feature_set_columns(feature_set)
    if not ("split" in df.columns and {"train", "test"} <= set(df["split"])):
        raise RuntimeError("特徵表缺少 train/test split。")
    df_tr = df[df["split"] == "train"].reset_index(drop=True)
    df_te = df[df["split"] == "test"].reset_index(drop=True)
    labels = [c for c in HEALTH_LABELS if c in set(df_tr["ylabel"])]
    return df_tr, df_te, cols, feature_set, labels


# --- 重採樣器工廠（每次回傳全新實例，避免 CV 與 sanity-check 共用狀態）---
def make_sampler(method: str):
    strat = {"LO": LO_TARGET}
    if method == "none":
        return None
    if method == "SMOTE":
        return SMOTE(sampling_strategy=strat, k_neighbors=5, random_state=RANDOM_STATE)
    if method == "BorderlineSMOTE":
        return BorderlineSMOTE(sampling_strategy=strat, k_neighbors=5, random_state=RANDOM_STATE)
    if method == "ADASYN":
        return ADASYN(sampling_strategy=strat, random_state=RANDOM_STATE)
    if method == "SMOTETomek":
        return SMOTETomek(
            smote=SMOTE(sampling_strategy=strat, k_neighbors=5, random_state=RANDOM_STATE),
            random_state=RANDOM_STATE)
    raise ValueError(method)


def build_pipeline(method: str, balanced: bool) -> ImbPipeline:
    """StandardScaler → (重採樣) → LogisticRegression。對照組省略重採樣步驟。"""
    cw = "balanced" if balanced else None
    clf = LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE)
    steps = [("scaler", StandardScaler())]
    sampler = make_sampler(method)
    if sampler is not None:
        steps.append(("resample", sampler))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def resampled_train_dist(method: str, X, y, labels) -> dict:
    """在完整 train 上單獨跑一次重採樣，記錄補樣後的類別分布（sanity check）。"""
    sampler = make_sampler(method)
    if sampler is None:
        counts = y.value_counts()
    else:
        Xs = StandardScaler().fit_transform(X)  # 與 pipeline 內順序一致
        _, ys = sampler.fit_resample(Xs, y)
        counts = pd.Series(ys).value_counts()
    return {lab: int(counts.get(lab, 0)) for lab in labels}


def evaluate(exp: dict, df_tr, df_te, cols, labels) -> dict:
    method, balanced = exp["method"], exp["class_weight"] == "balanced"
    X_tr, y_tr = df_tr[cols], df_tr["ylabel"]
    X_te, y_te = df_te[cols], df_te["ylabel"]

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(build_pipeline(method, balanced), X_tr, y_tr,
                         cv=skf, scoring="f1_macro")  # 整條 pipeline 進 CV → 無洩漏

    pipe = build_pipeline(method, balanced).fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    p, r, f, s = precision_recall_fscore_support(
        y_te, y_pred, labels=labels, zero_division=0)
    per_class = {lab: {"precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])}
                 for i, lab in enumerate(labels)}

    return {
        "id": exp["id"],
        "method": method,
        "class_weight": exp["class_weight"],
        "cv_macro_f1_mean": float(cv.mean()),
        "cv_macro_f1_std": float(cv.std()),
        "test_macro_f1": float(f1_score(y_te, y_pred, average="macro",
                                        labels=labels, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_te, y_pred)),
        "test_per_class": per_class,
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=labels).tolist(),
        "resampled_train_dist": resampled_train_dist(method, X_tr, y_tr, labels),
    }


def print_table(results: list[dict]) -> None:
    print("=" * 92)
    print("實驗 E2/E3：過採樣對比（LogReg；5-fold CV / 留出測試 800 段）")
    print("=" * 92)
    print(f"{'組別':<26}{'cw':<10}{'CV macro-F1 (μ±σ)':<20}"
          f"{'Test macF1':<12}{'LO recall':<11}{'LN prec':<9}")
    print("-" * 92)
    for r in results:
        cv = f"{r['cv_macro_f1_mean']:.3f}±{r['cv_macro_f1_std']:.3f}"
        cw = r["class_weight"] or "None"
        lo_rec = r["test_per_class"]["LO"]["recall"]
        ln_prec = r["test_per_class"]["LN"]["precision"]
        print(f"{r['id']:<26}{cw:<10}{cv:<20}"
              f"{r['test_macro_f1']:<12.3f}{lo_rec:<11.3f}{ln_prec:<9.3f}")
    print()


def plot_macro_f1(results: list[dict], baseline: float) -> Path:
    order = list(reversed(results))  # barh 由下往上，反轉讓 E0 在最上
    ids = [r["id"] + ("+bal" if r["class_weight"] == "balanced" and not r["id"].startswith("E0")
                      else "") for r in order]
    vals = [r["test_macro_f1"] for r in order]
    colors = ["#7F8C8D" if r["id"].startswith("E0")
              else ("#C0392B" if r["class_weight"] == "balanced" else "#2E86C1")
              for r in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(ids, vals, color=colors)
    ax.axvline(baseline, color="#27AE60", linestyle="--", linewidth=1.5,
               label=f"E0_balanced 基準 = {baseline:.3f}")
    for b, v in zip(bars, vals):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel("留出測試 macro-F1")
    ax.set_title("伺服馬達健康分類——重採樣方法留出 macro-F1 對比（LogReg）", fontsize=13)
    ax.set_xlim(0.55, max(vals) + 0.03)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = resolve(OUT_FIG)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    df_tr, df_te, cols, feature_set, labels = load_split()
    print(f"[E2/E3] 特徵組 {feature_set}（{len(cols)} 維）、訓練 {len(df_tr)} 段、"
          f"留出測試 {len(df_te)} 段、類別 {labels}\n")

    experiments = [
        {"id": "E0_default", "method": "none", "class_weight": None},
        {"id": "E0_balanced", "method": "none", "class_weight": "balanced"},
    ]
    for m in ["SMOTE", "BorderlineSMOTE", "ADASYN", "SMOTETomek"]:
        experiments.append({"id": m, "method": m, "class_weight": None})
        experiments.append({"id": m, "method": m, "class_weight": "balanced"})

    results = [evaluate(exp, df_tr, df_te, cols, labels) for exp in experiments]

    print_table(results)
    baseline = next(r["test_macro_f1"] for r in results if r["id"] == "E0_balanced")
    fig_path = plot_macro_f1(results, baseline)

    out = {
        "experiment": "E2_E3_resampling",
        "feature_set": feature_set,
        "eval": "holdout_test",
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "labels": labels,
        "label_zh": {k: HEALTH_LABEL_ZH[k] for k in labels},
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "lo_target": LO_TARGET,
        "notes": [
            "重採樣（SMOTE/BorderlineSMOTE/ADASYN/SMOTETomek）僅作用於訓練資料；"
            "留出 test 800 段全為真實樣本，未經任何合成或改動。",
            "訓練集的 LO 全部來自 train_noisy_LO 檔，與 test 的 LO 可能存在分布差異"
            "（domain shift）；重採樣只在特徵空間內插補、無法解決分布差異，"
            "僅緩解 LO 樣本量不足的問題。",
            "CV 將整條 imblearn Pipeline 交給 StratifiedKFold，重採樣只發生在每個 fold "
            "的訓練分割，避免資料洩漏；resampled_train_dist 為在完整 train 上另跑一次"
            "的補樣結果（sanity check，ADASYN/SMOTETomek 數量會浮動）。",
        ],
        "results": results,
    }
    out_path = resolve(OUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已寫入：{OUT_JSON}")
    print(f"已存圖：{Path(fig_path).relative_to(resolve('.'))}")


if __name__ == "__main__":
    main()
