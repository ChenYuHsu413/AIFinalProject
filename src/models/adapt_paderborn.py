"""CE1: domain adaptation for the Paderborn artificial->real gap (Module C).

The Module C MVP (``train_paderborn``) shows a near-total collapse when a
classifier trained on healthy + ARTIFICIAL (EDM/engraved) faults is tested on
REAL accelerated-lifetime damage: baseline 5-fold CV macro-F1 ~= 1.00 ->
artificial->real macro-F1 ~= 0.20 (gap ~0.80).  This module diagnoses that gap
as **domain shift** and tries three remedies on the *same* artificial->real
split, then writes a baseline-vs-methods ablation:

  1. **coral** — align the source (healthy+artificial) feature covariance to the
     real-damage target covariance (classic CORAL, reused from the XJTU B+ track).
     Uses target features only, never target labels -> unsupervised DA.
  2. **transductive_zscore** — standardise the source by its own stats and the
     target by its *own unlabelled* stats.  Also target-label-free.
  3. **few_shot** — admit k real samples per damaged class into training and test
     on the remaining real samples; a learning curve over k.  **This uses real
     labels** (disclosed honestly) -> few-shot, not zero-shot.

Honesty notes baked into the output JSON:
  * coral / transductive_zscore are label-free for the target (legitimate
    unsupervised DA); few_shot uses a few real labels (disclosed).
  * The real test set is **all damaged (no healthy class)**, so a 3-class
    macro-F1 mechanically includes a 0-support healthy term.  We therefore also
    report a 2-class outer/inner macro-F1 (``binary_f1``) for a fair read of the
    actual outer-vs-inner discriminability.

Run after the Paderborn feature table exists::

    python -m src.models.adapt_paderborn
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.models.eval_xjtu_domain_adapt import coral_align, per_condition_zscore
from src.models.train_paderborn import (
    LABELS,
    _make_pipe,
    feature_columns,
    split_artificial_real,
)
from src.utils.paths import ensure_output_dirs, load_config, resolve

DAMAGED = ["outer", "inner"]  # real-damage test set has no healthy class


def _macro_f1(y_true, y_pred) -> float:
    """3-class macro-F1 over the fixed label order (healthy + 0-support OK)."""
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))


def _binary_f1(y_true, y_pred) -> float:
    """Fair outer-vs-inner macro-F1 (drops the phantom 0-support healthy term).

    A ``healthy`` prediction on a real (damaged) sample still counts as a miss
    for its true class, so this neither rewards nor invents the absent class.
    """
    return float(f1_score(y_true, y_pred, average="macro", labels=DAMAGED, zero_division=0))


def _eval(model_name: str, feats: List[str], Xtr: np.ndarray, ytr,
          Xte: np.ndarray, yte, rs: int) -> Dict:
    """Fit ``model_name`` on transformed source features, score on real target."""
    tr_df = pd.DataFrame(Xtr, columns=feats)
    te_df = pd.DataFrame(Xte, columns=feats)
    pipe = _make_pipe(model_name, feats, rs).fit(tr_df, ytr)
    y_pred = pipe.predict(te_df)
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(yte, y_pred)),
        "macro_f1": _macro_f1(yte, y_pred),
        "binary_f1": _binary_f1(yte, y_pred),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(yte, y_pred, labels=LABELS).tolist(),
        "n": int(len(yte)),
    }


def _pick_best_model(train_df: pd.DataFrame, feats: List[str], models: List[str],
                     n_splits: int, rs: int) -> str:
    """Best baseline model by stratified-CV macro-F1 over healthy + artificial.

    Mirrors ``train_paderborn`` so the ablation fixes ONE classifier and the only
    thing varying across methods is the feature transform / few-shot labels.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = train_df[feats], train_df["fault_class"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    best_name, best_macro = None, -np.inf
    for name in models:
        try:
            y_cv = cross_val_predict(_make_pipe(name, feats, rs), X, y, cv=skf)
        except Exception:
            continue
        macro = float(f1_score(y, y_cv, average="macro"))
        if macro > best_macro:
            best_name, best_macro = name, macro
    if best_name is None:
        raise RuntimeError(f"沒有可用的多類別模型（嘗試了 {models}）。")
    return best_name


def _few_shot_curve(train_df: pd.DataFrame, real_df: pd.DataFrame, feats: List[str],
                    model_name: str, ks: List[int], seeds: int, rs: int,
                    align: bool = False, coral_reg: float = 1.0) -> List[Dict]:
    """Learning curve: admit k real samples per damaged class into training.

    For each k we draw ``seeds`` independent samples (per-class, leakage-free:
    the drawn rows are removed from the test set) and report mean ± std macro-F1.

    ``align=True`` first CORAL-recolours the (healthy+artificial) source onto the
    full real-feature covariance (target X only, label-free) before admitting the
    k real shots — tests whether unsupervised alignment *plus* a few labels beats
    few-shot alone.  The k shots and the test rows stay in raw real space.
    """
    curve: List[Dict] = []
    by_class = {c: real_df[real_df["fault_class"] == c] for c in DAMAGED}
    max_k = min(len(g) for g in by_class.values()) - 1  # leave >=1 per class to test
    src_feats = train_df[feats].to_numpy(dtype=float)
    if align:
        # Align once to the whole real pool's covariance (unsupervised, draw-free).
        src_feats, _ = coral_align(src_feats, train_df["condition"].to_numpy(),
                                   real_df[feats].to_numpy(dtype=float), coral_reg)
    src_X = pd.DataFrame(src_feats, columns=feats)
    src_y = train_df["fault_class"].reset_index(drop=True)
    for k in ks:
        if k > max_k:
            break
        macro, binary, n_tests = [], [], []
        for s in range(seeds):
            rng = np.random.RandomState(rs + s)
            shot_idx, test_mask = [], pd.Series(True, index=real_df.index)
            for c, g in by_class.items():
                chosen = rng.choice(g.index.to_numpy(), size=k, replace=False)
                shot_idx.extend(chosen.tolist())
                test_mask.loc[chosen] = False
            shots = real_df.loc[shot_idx]
            test = real_df[test_mask]
            tr_X = pd.concat([src_X, shots[feats].reset_index(drop=True)], ignore_index=True)
            tr_y = pd.concat([src_y, shots["fault_class"].reset_index(drop=True)], ignore_index=True)
            pipe = _make_pipe(model_name, feats, rs).fit(tr_X, tr_y)
            y_pred = pipe.predict(test[feats])
            macro.append(_macro_f1(test["fault_class"], y_pred))
            binary.append(_binary_f1(test["fault_class"], y_pred))
            n_tests.append(int(len(test)))
        curve.append({
            "k_per_class": int(k),
            "macro_f1_mean": float(np.mean(macro)),
            "macro_f1_std": float(np.std(macro)),
            "binary_f1_mean": float(np.mean(binary)),
            "binary_f1_std": float(np.std(binary)),
            "n_test_mean": float(np.mean(n_tests)),
            "seeds": seeds,
        })
    return curve


def _feature_diagnosis(train_df: pd.DataFrame, real_df: pd.DataFrame, feats: List[str],
                       best_pipe) -> Dict:
    """Why unsupervised alignment can't help: per-feature shift vs discriminability.

    For each feature, the **standardised mean shift** (Cohen's-d-like) between
    artificial and real damage *within the same fault class* measures how far the
    feature's distribution moves under the artificial->real shift; the baseline
    classifier's **importance** measures how much it relies on that feature.  A
    positive shift-vs-importance correlation means the discriminative axes
    themselves move — so a linear (covariance) realignment cannot recover them.
    """
    art = train_df[train_df["damage_origin"] == "artificial"]
    importances = _baseline_importances(best_pipe, feats, train_df)
    per_feature: List[Dict] = []
    for j, f in enumerate(feats):
        shifts = []
        for c in DAMAGED:
            a = art[art["fault_class"] == c][f].to_numpy(dtype=float)
            r = real_df[real_df["fault_class"] == c][f].to_numpy(dtype=float)
            if len(a) < 2 or len(r) < 2:
                continue
            pooled = np.sqrt((a.var(ddof=1) + r.var(ddof=1)) / 2.0) + 1e-9
            shifts.append(abs(a.mean() - r.mean()) / pooled)
        per_feature.append({
            "feature": f,
            "importance": float(importances[j]),
            "shift": float(np.mean(shifts)) if shifts else 0.0,
        })
    imp = np.array([p["importance"] for p in per_feature])
    shf = np.array([p["shift"] for p in per_feature])
    corr = _spearman(imp, shf)
    ranked = sorted(per_feature, key=lambda p: p["importance"], reverse=True)
    return {
        "per_feature": per_feature,
        "spearman_importance_vs_shift": float(corr),
        "top_discriminative": ranked[:5],
        "note": ("逐特徵 artificial→real 標準化均值位移（同類別內）對 baseline 重要度的 Spearman 相關。"
                 "正相關＝越被 baseline 倚重的判別特徵、位移越大，故線性協方差對齊（CORAL）無法復原判別軸。"),
    }


def _baseline_importances(best_pipe, feats: List[str], train_df: pd.DataFrame) -> np.ndarray:
    """Feature importances from the fitted baseline classifier.

    Uses tree ``feature_importances_`` when available (RF/GB — no scaling, so the
    order matches ``feats``); otherwise falls back to univariate mutual info on
    the source, so the diagnosis works for any chosen model.
    """
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_, dtype=float)
    from sklearn.feature_selection import mutual_info_classif
    return mutual_info_classif(train_df[feats], train_df["fault_class"], random_state=0)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (rank then Pearson) — no scipy dependency."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    pb = cfg["paderborn"]
    da = pb.get("domain_adapt", {})
    methods = da.get("methods", ["coral", "transductive_zscore", "few_shot"])
    reg = float(da.get("coral_reg", 1.0))
    ks = list(da.get("few_shot_k", [1, 3, 5, 10]))
    seeds = int(da.get("few_shot_seeds", 5))

    features_path = resolve(pb["processed_features"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"找不到 Paderborn 特徵表：{features_path}\n"
            "請先執行 python -m src.data.build_paderborn_dataset。"
        )
    df = pd.read_parquet(features_path)
    feats = feature_columns(df)
    train_df, real_df = split_artificial_real(df)
    if real_df.empty:
        raise RuntimeError("無真實損傷測試資料（config 未配置 real_* 或檔案缺失）。")
    rs = int(pb.get("random_state", 42))
    models = list(pb.get("enabled_models", ["random_forest"]))

    min_class = int(train_df["fault_class"].value_counts().min())
    n_splits = max(2, min(int(pb.get("cv_folds", 5)), min_class))
    best_name = _pick_best_model(train_df, feats, models, n_splits, rs)
    print(f"[CE1] 領域自適應：特徵 {len(feats)} 維；源(健康+人工) {len(train_df)} 列、"
          f"真實損傷 {len(real_df)} 列；固定分類器 {best_name}；手段 {methods}")

    Xs = train_df[feats].to_numpy(dtype=float)
    Xt = real_df[feats].to_numpy(dtype=float)
    ys, yt = train_df["fault_class"], real_df["fault_class"]
    cond = train_df["condition"].to_numpy()

    results: Dict[str, Dict] = {}

    # --- baseline: no adaptation (reproduces the MVP ~0.20 self-consistency) ---
    print("[1] baseline（無自適應）...")
    results["baseline"] = _eval(best_name, feats, Xs, ys, Xt, yt, rs)
    print(f"    -> macro-F1={results['baseline']['macro_f1']:.3f} "
          f"(outer/inner {results['baseline']['binary_f1']:.3f})")

    # baseline pipe (refit on raw source) for the feature-transferability diagnosis
    baseline_pipe = _make_pipe(best_name, feats, rs).fit(pd.DataFrame(Xs, columns=feats), ys)

    if "coral" in methods:
        print("[2] coral（協方差對齊，僅用 target 特徵）...")
        Xs_a, Xt_a = coral_align(Xs, cond, Xt, reg)
        results["coral"] = _eval(best_name, feats, Xs_a, ys, Xt_a, yt, rs)
        print(f"    -> macro-F1={results['coral']['macro_f1']:.3f} "
              f"(outer/inner {results['coral']['binary_f1']:.3f})")

    if "transductive_zscore" in methods:
        print("[3] transductive_zscore（源/目標各自標準化）...")
        Xs_z, Xt_z = per_condition_zscore(Xs, cond, Xt)
        results["transductive_zscore"] = _eval(best_name, feats, Xs_z, ys, Xt_z, yt, rs)
        print(f"    -> macro-F1={results['transductive_zscore']['macro_f1']:.3f} "
              f"(outer/inner {results['transductive_zscore']['binary_f1']:.3f})")

    if "few_shot" in methods:
        print(f"[4] few_shot（每類 k 筆真實標籤，k={ks}，{seeds} 抽樣平均）...")
        curve = _few_shot_curve(train_df, real_df, feats, best_name, ks, seeds, rs)
        curve_coral = _few_shot_curve(train_df, real_df, feats, best_name, ks, seeds, rs,
                                      align=True, coral_reg=reg)
        results["few_shot"] = {
            "model": best_name,
            "seeds": seeds,
            "uses_real_labels": True,
            "curve": curve,
            "curve_coral": curve_coral,  # CORAL-aligned source + k real shots
        }
        for pt, ptc in zip(curve, curve_coral):
            print(f"    k={pt['k_per_class']:>2}/class -> macro-F1="
                  f"{pt['macro_f1_mean']:.3f}±{pt['macro_f1_std']:.3f}"
                  f"  | +CORAL {ptc['macro_f1_mean']:.3f}")

    # --- feature-transferability diagnosis (why unsupervised alignment fails) ---
    print("[5] feature diagnosis（位移 vs 重要度）...")
    results["diagnosis"] = _feature_diagnosis(train_df, real_df, feats, baseline_pipe)
    print(f"    -> Spearman(importance, shift)="
          f"{results['diagnosis']['spearman_importance_vs_shift']:.3f}")

    # --- summary: best label-free remedy + best few-shot point ---
    unsup = {k: results[k]["macro_f1"] for k in ("coral", "transductive_zscore") if k in results}
    best_unsup = max(unsup, key=unsup.get) if unsup else None
    fs_curve = results.get("few_shot", {}).get("curve", [])
    best_fs = max(fs_curve, key=lambda p: p["macro_f1_mean"]) if fs_curve else None
    diag = results.get("diagnosis", {})
    summary = {
        "best_model": best_name,
        "baseline_macro_f1": results["baseline"]["macro_f1"],
        "baseline_binary_f1": results["baseline"]["binary_f1"],
        "best_unsup_method": best_unsup,
        "best_unsup_macro_f1": (unsup[best_unsup] if best_unsup else None),
        "few_shot_best_k": (best_fs["k_per_class"] if best_fs else None),
        "few_shot_best_macro_f1": (best_fs["macro_f1_mean"] if best_fs else None),
        "spearman_importance_vs_shift": diag.get("spearman_importance_vs_shift"),
        "honesty": {
            "unsupervised_methods": ["coral", "transductive_zscore"],
            "uses_real_labels": ["few_shot"],
            "real_test_has_no_healthy": True,
            "note": ("CORAL/z-score 僅用 target 未標註特徵（無監督 DA）；few_shot 用了少量真實標籤。"
                     "真實測試集無 healthy 類，3 類 macro-F1 含一個 0 分項，故另報 outer/inner 二類 binary_f1。"),
        },
    }
    print(f"[done] baseline macro-F1={summary['baseline_macro_f1']:.3f}；"
          f"最佳無監督={best_unsup}（{summary['best_unsup_macro_f1']}）；"
          f"few-shot 最佳 k={summary['few_shot_best_k']}")

    out_json = resolve(da.get("metrics", "outputs/metrics/paderborn_domain_adapt.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "paderborn_artificial_to_real_domain_adaptation",
                "features": feats,
                "coral_reg": reg,
                "results": results,
                "summary": summary,
            },
            f, indent=2,
        )
    print(f"    -> 指標：{out_json}")
    return out_json


if __name__ == "__main__":
    run()
