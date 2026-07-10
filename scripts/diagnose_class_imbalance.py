"""伺服馬達健康狀態分類——類別不平衡診斷腳本（唯讀，不動訓練程式）。

用途：
  1. 統計 servo_features.parquet 中 train/test split 各類別（ylabel）的樣本數。
  2. 讀 servo_clf_eval.json，印出留出測試的 per-class precision/recall/F1 與混淆矩陣；
     若 JSON 內無混淆矩陣，則載入 servo_clf.joblib 對留出集重算一次。
  3. 用 matplotlib 產生兩張圖到 outputs/figures/：
       (a) 訓練集類別分布長條圖   servo_train_class_dist.png
       (b) 混淆矩陣熱圖           servo_confusion_heatmap.png

執行：
    python scripts/diagnose_class_imbalance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 確保繁體中文在 Windows 主控台（cp950）也不亂碼
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import matplotlib
matplotlib.use("Agg")  # 無視窗環境也能存圖
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 專案路徑（本檔位於 scripts/，專案根在上一層）---
ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "servo_features.parquet"
EVAL_JSON = ROOT / "outputs" / "metrics" / "servo_clf_eval.json"
CLF_MODEL = ROOT / "outputs" / "models" / "servo_clf.joblib"
FIG_DIR = ROOT / "outputs" / "figures"

# 健康等級由輕到重的顯示順序（與訓練程式 HEALTH_LABELS 一致）
LABEL_ORDER = ["LN", "LO", "MED", "HI"]
LABEL_ZH_FALLBACK = {"LN": "健康", "LO": "輕度退化", "MED": "中度退化", "HI": "高度退化"}

# 讓 matplotlib 能顯示繁體中文（Windows 內建微軟正黑體）
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _ordered_labels(present: set[str]) -> list[str]:
    """依 LABEL_ORDER 排序，並保留資料中實際出現的類別。"""
    ordered = [c for c in LABEL_ORDER if c in present]
    ordered += sorted(present - set(ordered))  # 任何非預期類別接在後面
    return ordered


# ---------------------------------------------------------------------------
# 步驟 1：split × 類別 樣本數統計
# ---------------------------------------------------------------------------
def count_split_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    labels = _ordered_labels(set(df["ylabel"]))
    splits = [s for s in ["train", "test"] if s in set(df["split"])]
    splits += [s for s in sorted(set(df["split"])) if s not in splits]

    table = pd.DataFrame(0, index=labels, columns=splits, dtype=int)
    for sp in splits:
        vc = df.loc[df["split"] == sp, "ylabel"].value_counts()
        for lab in labels:
            table.at[lab, sp] = int(vc.get(lab, 0))
    table["總計"] = table.sum(axis=1)
    table.loc["總計"] = table.sum(axis=0)

    print("=" * 60)
    print("步驟 1：train / test split 各類別樣本數")
    print("=" * 60)
    zh = {lab: LABEL_ZH_FALLBACK.get(lab, lab) for lab in labels}
    disp = table.copy()
    disp.index = [f"{lab}（{zh[lab]}）" if lab in zh else lab for lab in table.index]
    print(disp.to_string())

    # 不平衡比（訓練集最多 / 最少類別）
    if "train" in splits:
        tr = table.loc[labels, "train"]
        if tr.min() > 0:
            print(f"\n訓練集不平衡比（多數/少數）：{tr.max() / tr.min():.2f} "
                  f"（最多 {tr.idxmax()}={tr.max()}，最少 {tr.idxmin()}={tr.min()}）")
    print()
    return table, labels


# ---------------------------------------------------------------------------
# 步驟 2：per-class precision/recall/F1 與混淆矩陣
# ---------------------------------------------------------------------------
def _per_class_from_cm(cm: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """由混淆矩陣（列=真實、行=預測）推算 per-class precision/recall/F1。"""
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    pred_sum = cm.sum(axis=0)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(tp), where=denom > 0)
    return pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1,
         "support": support.astype(int)},
        index=labels,
    )


def _recompute_cm_from_model(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """fallback：載入 joblib，對 parquet 的 test split 重算混淆矩陣。"""
    import joblib
    from sklearn.metrics import confusion_matrix

    bundle = joblib.load(CLF_MODEL)
    pipe = bundle["pipeline"]
    cols = bundle["feature_columns"]
    model_labels = bundle.get("labels", labels)

    df = pd.read_parquet(FEATURES)
    df_te = df[df["split"] == "test"]
    y_true = df_te["ylabel"]
    y_pred = pipe.predict(df_te[cols])
    cm = confusion_matrix(y_true, y_pred, labels=model_labels)
    return cm, list(model_labels)


def report_metrics(ev: dict) -> tuple[np.ndarray, list[str], dict]:
    labels = ev.get("labels") or LABEL_ORDER
    label_zh = ev.get("label_zh", {lab: LABEL_ZH_FALLBACK.get(lab, lab) for lab in labels})

    cm = ev.get("confusion_matrix")
    if cm is None:
        print("[!] JSON 內無混淆矩陣 → 載入 servo_clf.joblib 對留出集重算。\n")
        cm, labels = _recompute_cm_from_model(labels)
    cm = np.asarray(cm)

    print("=" * 60)
    print(f"步驟 2：留出測試 per-class 指標（eval={ev.get('eval', 'holdout_test')}, "
          f"n={int(cm.sum())}）")
    print("=" * 60)

    pc = _per_class_from_cm(cm, labels)
    disp = pc.copy()
    disp.index = [f"{lab}（{label_zh.get(lab, lab)}）" for lab in labels]
    disp["precision"] = disp["precision"].map("{:.3f}".format)
    disp["recall"] = disp["recall"].map("{:.3f}".format)
    disp["f1"] = disp["f1"].map("{:.3f}".format)
    print(disp.to_string())

    acc = np.diag(cm).sum() / cm.sum() if cm.sum() else 0.0
    macro_f1 = pc["f1"].mean()
    print(f"\nAccuracy={acc:.3f}   Macro-F1={macro_f1:.3f}"
          + (f"   （JSON 記載 macro_f1={ev['macro_f1']:.3f}）" if "macro_f1" in ev else ""))

    print("\n混淆矩陣（列=真實，行=預測）：")
    cm_df = pd.DataFrame(cm, index=[f"真:{l}" for l in labels],
                         columns=[f"預:{l}" for l in labels])
    print(cm_df.to_string())
    print()
    return cm, labels, label_zh


# ---------------------------------------------------------------------------
# 步驟 3：畫圖
# ---------------------------------------------------------------------------
def plot_train_distribution(table: pd.DataFrame, labels: list[str], label_zh: dict) -> Path:
    if "train" not in table.columns:
        print("[!] 無 train split，跳過類別分布長條圖。")
        return None
    counts = [int(table.at[lab, "train"]) for lab in labels]
    xticks = [f"{lab}\n{label_zh.get(lab, lab)}" for lab in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(xticks, counts, color=["#2E86C1", "#28B463", "#F39C12", "#C0392B"][:len(labels)])
    ax.set_title("伺服馬達健康狀態——訓練集類別分布", fontsize=13)
    ax.set_xlabel("健康等級")
    ax.set_ylabel("樣本數（段）")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c),
                ha="center", va="bottom", fontsize=10)
    ax.margins(y=0.12)
    fig.tight_layout()
    out = FIG_DIR / "servo_train_class_dist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion_heatmap(cm: np.ndarray, labels: list[str], label_zh: dict) -> Path:
    xt = [f"{lab}\n{label_zh.get(lab, lab)}" for lab in labels]
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="樣本數")

    ax.set_xticks(range(len(labels)), labels=xt)
    ax.set_yticks(range(len(labels)), labels=xt)
    ax.set_xlabel("預測類別")
    ax.set_ylabel("真實類別")
    ax.set_title("伺服馬達健康分類——留出測試混淆矩陣", fontsize=13)

    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "servo_confusion_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    if not FEATURES.exists():
        raise FileNotFoundError(f"找不到特徵表：{FEATURES}")
    if not EVAL_JSON.exists():
        raise FileNotFoundError(f"找不到評估結果：{EVAL_JSON}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES)
    table, _ = count_split_classes(df)

    ev = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    cm, labels, label_zh = report_metrics(ev)

    print("=" * 60)
    print("步驟 3：輸出圖檔")
    print("=" * 60)
    p1 = plot_train_distribution(table, labels, label_zh)
    p2 = plot_confusion_heatmap(cm, labels, label_zh)
    for p in (p1, p2):
        if p:
            print(f"  已存：{p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
