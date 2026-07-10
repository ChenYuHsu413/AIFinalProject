"""E5 + 特徵組選擇驗證 —— full 空間的擴充方法對決 + FLAML 公平戰。

延續 E4（full 21 維留出 macro-F1 0.819 >> engineered 7 維 0.757）。唯讀實驗：沿用
train_servo.run() 的載入與 split（train 665 / 留出 test 800），不重新切分、不動測試集。

Task 1 — 特徵組選擇的 CV 驗證（最先做）：train 上 5-fold StratifiedKFold 比較
  LR+balanced 在 engineered vs full 的 CV macro-F1，證明「選 full」可只依訓練集決定，
  0.819 僅為事後驗證，排除 test-set selection 疑慮。
Task 2 — E5：full 空間三組（皆 LR max_iter=2000, rs=42, StandardScaler, full 特徵）
  a. E0_full         cw='balanced'、無擴充（應重現 0.819）
  b. SMOTE_full      SMOTE(LO→200, k=5) + cw=None，imblearn Pipeline、CV 防洩漏
  c. CTGAN_full      以 65 筆真實 LO 訓練 CTGANSynthesizer(epochs=300) 生成 135 筆合成 LO
                     補到 200 + cw=None；含 KDE 品質圖 + TSTR
Task 3 — E4c：FLAML + full + time_budget=600 + sample_weight='balanced'（公平一戰）。

執行：./.venv/Scripts/python.exe scripts/run_e5_and_validation.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.features.servo_features import HEALTH_LABELS, feature_set_columns
from src.servo.field_glossary import HEALTH_LABEL_ZH
from src.utils.paths import load_config, resolve

RANDOM_STATE = 42
TIME_BUDGET = 600
LO_TARGET = 200
FEATURE_SET = "full"
OUT_JSON = "outputs/metrics/e5_validation.json"
KDE_FIG = "outputs/figures/e5_ctgan_kde_full.png"
E4_JSON = "outputs/metrics/e4_automl_features.json"

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_split():
    sv = load_config()["servo"]
    feat_path = resolve(sv["processed_features"])
    df = pd.read_parquet(feat_path)
    df_tr = df[df["split"] == "train"].reset_index(drop=True)
    df_te = df[df["split"] == "test"].reset_index(drop=True)
    labels = [c for c in HEALTH_LABELS if c in set(df_tr["ylabel"])]
    return df_tr, df_te, labels


def per_class(y_true, y_pred, labels) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {lab: {"precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)}


def test_block(y_true, y_pred, labels) -> dict:
    return {
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                        labels=labels, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_per_class": per_class(y_true, y_pred, labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# ---------------------------------------------------------------------------
# Task 1：特徵組選擇的 CV 驗證（只依訓練集）
# ---------------------------------------------------------------------------
def task1_feature_validation(df_tr) -> dict:
    print("=" * 78)
    print("Task 1：特徵組選擇 CV 驗證（train 665 段，5-fold，LR+balanced）")
    print("=" * 78)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    for fs in ["engineered", "full"]:
        cols = feature_set_columns(fs)
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    random_state=RANDOM_STATE))])
        sc = cross_val_score(pipe, df_tr[cols], df_tr["ylabel"],
                             cv=skf, scoring="f1_macro")
        out[fs] = {"n_features": len(cols), "cv_macro_f1_mean": float(sc.mean()),
                   "cv_macro_f1_std": float(sc.std()), "folds": [round(float(x), 4) for x in sc]}
        print(f"  {fs:<12}({len(cols):>2}維): CV macro-F1 = {sc.mean():.3f} ± {sc.std():.3f}")
    selected = "full" if out["full"]["cv_macro_f1_mean"] >= out["engineered"]["cv_macro_f1_mean"] else "engineered"
    delta = out["full"]["cv_macro_f1_mean"] - out["engineered"]["cv_macro_f1_mean"]
    print(f"  → 依訓練集 CV 選擇：{selected}（full − engineered = {delta:+.3f}）\n")
    return {
        "method": "5-fold StratifiedKFold macro-F1 on TRAIN only (LR + class_weight='balanced', rs=42)",
        "engineered": out["engineered"],
        "full": out["full"],
        "selected": selected,
        "cv_delta_full_minus_engineered": float(delta),
        "decision_rule": "以訓練集 CV macro-F1 平均較高者為準；選擇不參考留出 test，"
                         "故留出 0.819 僅為事後驗證，排除 test-set selection。",
    }


# ---------------------------------------------------------------------------
# Task 2：E5 full 空間三組
# ---------------------------------------------------------------------------
def e0_full(df_tr, df_te, cols, labels) -> dict:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=RANDOM_STATE))]).fit(
        df_tr[cols], df_tr["ylabel"])
    r = {"id": "E0_full", "method": "none", "class_weight": "balanced",
         **test_block(df_te["ylabel"], pipe.predict(df_te[cols]), labels)}
    print(f"[E5] {r['id']:<14} test macro-F1={r['test_macro_f1']:.3f}")
    return r


def smote_full(df_tr, df_te, cols, labels) -> dict:
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy={"LO": LO_TARGET}, k_neighbors=5,
                        random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, class_weight=None,
                                   random_state=RANDOM_STATE)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(pipe, df_tr[cols], df_tr["ylabel"], cv=skf, scoring="f1_macro")
    pipe.fit(df_tr[cols], df_tr["ylabel"])
    r = {"id": "SMOTE_full", "method": "SMOTE(LO->200,k=5)", "class_weight": None,
         "cv_macro_f1_mean": float(cv.mean()), "cv_macro_f1_std": float(cv.std()),
         **test_block(df_te["ylabel"], pipe.predict(df_te[cols]), labels)}
    print(f"[E5] {r['id']:<14} test macro-F1={r['test_macro_f1']:.3f}"
          f"  (CV {cv.mean():.3f}±{cv.std():.3f})")
    return r


def ctgan_full(df_tr, df_te, cols, labels) -> tuple[dict, dict]:
    """以 65 筆真實 LO 訓練 CTGAN、生成合成 LO；補到 200 訓練 + KDE 品質圖 + TSTR。"""
    import torch
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    lo_real = df_tr[df_tr["ylabel"] == "LO"][cols].reset_index(drop=True)
    n_real = len(lo_real)
    n_gen = LO_TARGET - n_real  # 65 -> 200 需補 135

    md = SingleTableMetadata()
    md.detect_from_dataframe(lo_real)
    synth = CTGANSynthesizer(md, epochs=300, verbose=False, enforce_min_max_values=True)
    print(f"[E5] CTGAN 訓練中（{n_real} 筆真實 LO、{len(cols)} 維、epochs=300）…")
    synth.fit(lo_real)
    lo_syn = synth.sample(n_gen).reset_index(drop=True)

    # (1) KDE 品質圖：每個特徵 真實 vs 合成 疊圖
    _plot_kde(lo_real, lo_syn, cols)

    # (2) 補到 200 併入訓練集，cw=None
    aug_lo = pd.concat([lo_real, lo_syn], ignore_index=True)
    aug_lo["ylabel"] = "LO"
    others = df_tr[df_tr["ylabel"] != "LO"][cols + ["ylabel"]]
    aug = pd.concat([others, aug_lo[cols + ["ylabel"]]], ignore_index=True)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight=None,
                                                random_state=RANDOM_STATE))]).fit(
        aug[cols], aug["ylabel"])
    res = {"id": "CTGAN_full", "method": f"CTGAN(+{n_gen} synth LO, epochs=300)",
           "class_weight": None,
           **test_block(df_te["ylabel"], pipe.predict(df_te[cols]), labels)}
    print(f"[E5] {res['id']:<14} test macro-F1={res['test_macro_f1']:.3f}")

    # (3) TSTR：僅合成 LO + 真實其他類 訓練，留出測 LO 的 F1
    lo_syn_tstr = synth.sample(LO_TARGET).reset_index(drop=True)
    lo_syn_tstr["ylabel"] = "LO"
    tstr_train = pd.concat([others, lo_syn_tstr[cols + ["ylabel"]]], ignore_index=True)
    tstr_pipe = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                     random_state=RANDOM_STATE))]).fit(
        tstr_train[cols], tstr_train["ylabel"])
    tstr_pred = tstr_pipe.predict(df_te[cols])
    tstr_pc = per_class(df_te["ylabel"], tstr_pred, labels)
    quality = {
        "n_real_lo": int(n_real),
        "n_synth_generated": int(n_gen),
        "epochs": 300,
        "kde_figure": KDE_FIG,
        "tstr": {
            "desc": "訓練=合成 LO(200) + 真實 LN/MED/HI；測=留出 test",
            "lo_f1": float(tstr_pc["LO"]["f1"]),
            "lo_recall": float(tstr_pc["LO"]["recall"]),
            "macro_f1": float(f1_score(df_te["ylabel"], tstr_pred,
                                       average="macro", labels=labels, zero_division=0)),
        },
    }
    print(f"[E5] CTGAN TSTR: LO F1={quality['tstr']['lo_f1']:.3f}"
          f"  (合成 LO 訓練、真實 LO 測試)")
    return res, quality


def _plot_kde(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str]) -> None:
    from scipy.stats import gaussian_kde
    ncol = 5
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 2.4))
    axes = axes.ravel()
    for i, c in enumerate(cols):
        ax = axes[i]
        for data, color, name in [(real[c].to_numpy(), "#2E86C1", "真實"),
                                  (syn[c].to_numpy(), "#C0392B", "合成")]:
            try:
                xs = np.linspace(data.min(), data.max(), 200)
                ax.plot(xs, gaussian_kde(data)(xs), color=color, label=name, lw=1.3)
            except Exception:
                ax.hist(data, bins=15, density=True, color=color, alpha=0.4, label=name)
        ax.set_title(c, fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("CTGAN 合成 LO vs 真實 LO —— 各特徵分布（full 21 維，65 筆訓練）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = resolve(KDE_FIG)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[E5] 已存 CTGAN KDE 品質圖：{KDE_FIG}")


# ---------------------------------------------------------------------------
# Task 3：E4c FLAML 公平戰（full + sample_weight balanced）
# ---------------------------------------------------------------------------
def flaml_weighted_full(df_tr, df_te, cols, labels) -> dict:
    from flaml import AutoML
    X_tr, y_tr = df_tr[cols], df_tr["ylabel"]
    sw = compute_sample_weight("balanced", y_tr)
    automl = AutoML()
    print(f"\n[E4c] FLAML + full + sample_weight='balanced'（time_budget={TIME_BUDGET}s）…")
    automl.fit(X_train=X_tr, y_train=y_tr, sample_weight=sw,
               task="classification", metric="macro_f1",
               time_budget=TIME_BUDGET, seed=RANDOM_STATE, verbose=1, log_file_name="")
    y_pred = automl.predict(df_te[cols])
    r = {"id": "FLAML_weighted_full", "method": "FLAML+sample_weight(balanced)",
         "class_weight": "sample_weight",
         "flaml": {
             "best_estimator": automl.best_estimator,
             "best_config": automl.best_config,
             "best_val_macro_f1": float(1.0 - automl.best_loss),
             "models_tried": list(getattr(automl, "best_config_per_estimator", {}).keys()),
         },
         **test_block(df_te["ylabel"], y_pred, labels)}
    print(f"[E4c] best={automl.best_estimator}  內部驗證={1 - automl.best_loss:.3f}"
          f"  test macro-F1={r['test_macro_f1']:.3f}")
    return r


def _engineered_history() -> dict:
    """從 e4_automl_features.json 取 engineered LR+balanced 歷史對照。"""
    p = resolve(E4_JSON)
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in d["results"]:
        if r.get("id") == "LR_balanced_engineered":
            return {"id": "engineered(LR+bal, 歷史)", "feature_set": "engineered",
                    "test_macro_f1": r["test_macro_f1"], "test_per_class": r["test_per_class"]}
    return {}


def print_table(results: list[dict], hist: dict) -> None:
    print("\n" + "=" * 84)
    print("E5 總表：full 空間擴充方法對決（留出 test 800 段）")
    print("=" * 84)
    print(f"{'組別':<24}{'特徵組':<12}{'test macF1':<12}{'LO recall':<11}{'LN prec':<9}")
    print("-" * 84)
    if hist:
        pc = hist["test_per_class"]
        print(f"{hist['id']:<24}{'engineered':<12}{hist['test_macro_f1']:<12.3f}"
              f"{pc['LO']['recall']:<11.3f}{pc['LN']['precision']:<9.3f}")
        print("-" * 84)
    for r in results:
        pc = r["test_per_class"]
        print(f"{r['id']:<24}{'full':<12}{r['test_macro_f1']:<12.3f}"
              f"{pc['LO']['recall']:<11.3f}{pc['LN']['precision']:<9.3f}")
    print()


def main() -> None:
    warnings.filterwarnings("ignore")
    df_tr, df_te, labels = load_split()
    cols = feature_set_columns(FEATURE_SET)
    print(f"[E5] 訓練 {len(df_tr)} 段、留出 test {len(df_te)} 段、特徵組 {FEATURE_SET}"
          f"（{len(cols)} 維）、類別 {labels}\n")

    fsv = task1_feature_validation(df_tr)

    print("=" * 78)
    print("Task 2：E5 full 空間三組")
    print("=" * 78)
    r_e0 = e0_full(df_tr, df_te, cols, labels)
    r_smote = smote_full(df_tr, df_te, cols, labels)
    r_ctgan, ctgan_quality = ctgan_full(df_tr, df_te, cols, labels)

    r_flaml = flaml_weighted_full(df_tr, df_te, cols, labels)

    results = [r_e0, r_smote, r_ctgan, r_flaml]
    hist = _engineered_history()
    print_table(results, hist)

    out = {
        "experiment": "E5_and_validation",
        "eval": "holdout_test",
        "random_state": RANDOM_STATE,
        "feature_set": FEATURE_SET,
        "labels": labels,
        "label_zh": {k: HEALTH_LABEL_ZH[k] for k in labels},
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "feature_selection_validation": fsv,
        "results": results,
        "reference_engineered": hist,
        "ctgan_quality": ctgan_quality,
        "notes": [
            "所有擴充（SMOTE / CTGAN）僅作用於訓練資料；留出 test 800 段全為真實樣本、"
            "未參與任何訓練或合成。",
            "CTGAN 僅以 65 筆真實 LO 訓練，屬小樣本；在 21 維連續特徵空間下生成品質有限"
            "（見 e5_ctgan_kde_full.png 與 TSTR）。合成樣本仍侷限於既有特徵分布，"
            "無法跨越 train/test 的 LO domain shift。",
            "Task 1 的特徵組選擇僅依訓練集 5-fold CV，與留出 test 無關；留出 0.819 為事後驗證。",
            "E4c 給 FLAML 傳入 sample_weight='balanced'，排除『AutoML 輸在沒拿到不平衡處理』的質疑。",
        ],
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
