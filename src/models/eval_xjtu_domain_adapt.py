"""E1: cross-condition domain adaptation for supervised RUL (Module B+).

The leave-one-condition-out (LOCO) baseline collapses to pooled R^2 ~= -1.22:
trained on two operating conditions, a RandomForest cannot predict the held-out
condition's absolute remaining hours because (a) lifetimes differ ~20x across
conditions and (b) the feature distribution shifts with speed/load.  This module
diagnoses that as **domain shift** and tries three remedies on the *same* LOCO
split, then writes a baseline-vs-methods ablation:

  1. **lifetime_ratio** — relabel the target as remaining-life *fraction* (0..1)
     instead of absolute hours.  Directly attacks the lifetime-scale mismatch.
     Restoring a fraction to hours needs a life estimate; we report three numbers:
       * ``r2_ratio_space``  — evaluated in fraction units (leakage-free).
       * ``r2_deploy``       — restored with the *source-mean* life, a constant
         known at inference (leakage-free; the honest, deployable number).
       * ``r2_oracle``       — restored with the target bearing's *true* life
         (uses a label-derived quantity -> an UPPER BOUND, flagged as leaky).

  2. **transductive_zscore** — standardise each condition's features by its own
     statistics (the target by its own *unlabelled* feature stats).  Legitimate
     transductive / unsupervised DA: uses target X only, never target RUL.

  3. **coral** — align the pooled-source feature covariance to the target's
     covariance (classic CORAL, pure numpy).  Also uses target X only.

All transforms are label-free for the target.  Run after the feature table::

    python -m src.models.eval_xjtu_domain_adapt
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.train_rul_lobo import _feature_columns
from src.utils.paths import ensure_output_dirs, load_config, resolve

RANDOM_STATE = 42

# A feature transform takes (X_train, condition_of_each_train_row, X_test) and
# returns aligned (X_train, X_test).  It must never see the target labels.
Transform = Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def _rf() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae_hours": float(mean_absolute_error(y_true, y_pred)),
        "rmse_hours": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Label-free feature transforms (target uses unlabelled features only)
# ---------------------------------------------------------------------------
def per_condition_zscore(
    X_train: np.ndarray, cond_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardise each source condition by its own stats; the target by its own.

    Aligns the per-condition feature offset/scale so a value that is "high for
    its own regime" maps to the same standardised value across conditions.  The
    target is standardised with **its own unlabelled feature statistics** only —
    no target RUL is used, so this is transductive / unsupervised DA.
    """
    Xtr = np.empty_like(X_train, dtype=float)
    for c in np.unique(cond_train):
        m = cond_train == c
        mu = X_train[m].mean(axis=0)
        sd = X_train[m].std(axis=0) + 1e-8
        Xtr[m] = (X_train[m] - mu) / sd
    mu_t = X_test.mean(axis=0)
    sd_t = X_test.std(axis=0) + 1e-8
    Xte = (X_test - mu_t) / sd_t
    return Xtr, Xte


def _sqrtm_sym(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return (V * np.sqrt(w)) @ V.T


def _inv_sqrtm_sym(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def coral_align(
    X_train: np.ndarray, cond_train: np.ndarray, X_test: np.ndarray, reg: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """CORAL: recolour the source so its covariance matches the target's.

    Whitens the (centred) source by its own covariance, then re-colours with the
    target covariance, and recentres to the target mean.  Uses the target feature
    covariance/mean only (no labels).  ``cond_train`` is unused (whole-source
    alignment) but kept for a uniform transform signature.
    """
    mu_s = X_train.mean(axis=0)
    mu_t = X_test.mean(axis=0)
    Xs = X_train - mu_s
    n_feat = X_train.shape[1]
    Cs = np.cov(Xs, rowvar=False) + reg * np.eye(n_feat)
    Ct = np.cov(X_test - mu_t, rowvar=False) + reg * np.eye(n_feat)
    Xs_aligned = Xs @ _inv_sqrtm_sym(Cs) @ _sqrtm_sym(Ct)
    return Xs_aligned + mu_t, X_test.astype(float)


# ---------------------------------------------------------------------------
# LOCO evaluation variants
# ---------------------------------------------------------------------------
def _loco_hours(
    df: pd.DataFrame, feats: List[str], conditions: List[str],
    transform: Optional[Transform] = None,
) -> Dict:
    """LOCO on the absolute-hours target, optionally with a feature transform."""
    per_condition: List[Dict] = []
    all_true, all_pred = [], []
    for held in conditions:
        train = df[df["condition"] != held]
        test = df[df["condition"] == held]
        if len(test) == 0:
            continue
        Xtr = train[feats].to_numpy(dtype=float)
        Xte = test[feats].to_numpy(dtype=float)
        if transform is not None:
            Xtr, Xte = transform(Xtr, train["condition"].to_numpy(), Xte)
        model = _rf()
        model.fit(Xtr, train["rul_hours"].to_numpy())
        pred = model.predict(Xte)
        yt = test["rul_hours"].to_numpy()
        m = _metrics(yt, pred)
        m["held_out_condition"] = held
        per_condition.append(m)
        all_true.append(yt)
        all_pred.append(pred)
    yt_all = np.concatenate(all_true)
    yp_all = np.concatenate(all_pred)
    return {"per_condition": per_condition, "pooled": _metrics(yt_all, yp_all)}


def _bearing_life(df: pd.DataFrame) -> pd.Series:
    """Total life (hours) of each bearing = its maximum RUL (RUL at onset of life)."""
    return df.groupby("bearing")["rul_hours"].transform("max")


def _loco_ratio(df: pd.DataFrame, feats: List[str], conditions: List[str]) -> Dict:
    """LOCO with a remaining-life-*fraction* target; three honesty-tiered scores.

    Returns pooled ``r2_ratio_space`` (fraction units, leakage-free),
    ``r2_deploy`` (restored with source-mean life, leakage-free) and
    ``r2_oracle`` (restored with the target bearing's true life — upper bound).
    """
    df = df.copy()
    df["life"] = _bearing_life(df)
    df["rul_frac"] = (df["rul_hours"] / df["life"]).clip(0.0, 1.0)

    per_condition: List[Dict] = []
    frac_true, frac_pred = [], []
    hrs_true, hrs_deploy, hrs_oracle = [], [], []
    for held in conditions:
        train = df[df["condition"] != held]
        test = df[df["condition"] == held]
        if len(test) == 0:
            continue
        model = _rf()
        model.fit(train[feats].to_numpy(dtype=float), train["rul_frac"].to_numpy())
        pf = np.clip(model.predict(test[feats].to_numpy(dtype=float)), 0.0, 1.0)

        # source-mean life = mean over TRAINING bearings (a constant known at
        # inference -> deployable, leakage-free).
        src_life = float(train.groupby("bearing")["life"].first().mean())
        deploy = pf * src_life
        oracle = pf * test["life"].to_numpy()  # target true life -> upper bound

        ft = test["rul_frac"].to_numpy()
        yt = test["rul_hours"].to_numpy()
        per_condition.append({
            "held_out_condition": held,
            "n": int(len(test)),
            "r2_ratio_space": float(r2_score(ft, pf)),
            "r2_deploy": float(r2_score(yt, deploy)),
            "mae_deploy": float(mean_absolute_error(yt, deploy)),
            "r2_oracle": float(r2_score(yt, oracle)),
            "mae_oracle": float(mean_absolute_error(yt, oracle)),
            "source_mean_life_hours": src_life,
        })
        frac_true.append(ft); frac_pred.append(pf)
        hrs_true.append(yt); hrs_deploy.append(deploy); hrs_oracle.append(oracle)

    ft_all = np.concatenate(frac_true); pf_all = np.concatenate(frac_pred)
    yt_all = np.concatenate(hrs_true)
    dep_all = np.concatenate(hrs_deploy); ora_all = np.concatenate(hrs_oracle)
    pooled = {
        "r2_ratio_space": float(r2_score(ft_all, pf_all)),
        "mae_ratio": float(mean_absolute_error(ft_all, pf_all)),
        # headline (comparable to baseline hours R^2): deployable restore
        "r2": float(r2_score(yt_all, dep_all)),
        "mae_hours": float(mean_absolute_error(yt_all, dep_all)),
        "rmse_hours": float(np.sqrt(mean_squared_error(yt_all, dep_all))),
        "r2_oracle": float(r2_score(yt_all, ora_all)),
        "mae_oracle_hours": float(mean_absolute_error(yt_all, ora_all)),
        "n": int(len(yt_all)),
    }
    return {"per_condition": per_condition, "pooled": pooled}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    xj = cfg["xjtu"]
    da = xj.get("domain_adapt", {})
    methods = da.get("methods", ["lifetime_ratio", "transductive_zscore", "coral"])
    reg = float(da.get("coral_reg", 1.0))

    features_path = resolve(xj["processed_features"])
    if not features_path.exists():
        raise FileNotFoundError(
            f"找不到 XJTU 特徵表：{features_path}\n"
            "請先執行 python -m src.data.build_xjtu_dataset。"
        )
    df = pd.read_parquet(features_path)
    feats = _feature_columns(df)
    conditions = [c["name"] for c in xj["conditions"]]
    print(f"[E1] 跨工況自適應 RUL：{len(conditions)} 工況留一，特徵 {len(feats)} 維，手段 {methods}")

    results: Dict[str, Dict] = {}
    print("[1] baseline（無自適應）...")
    results["baseline"] = _loco_hours(df, feats, conditions)
    print(f"    -> pooled R2={results['baseline']['pooled']['r2']:.3f} "
          f"MAE={results['baseline']['pooled']['mae_hours']:.2f}h")

    if "lifetime_ratio" in methods:
        print("[2] lifetime_ratio（壽命比例目標）...")
        results["lifetime_ratio"] = _loco_ratio(df, feats, conditions)
        p = results["lifetime_ratio"]["pooled"]
        print(f"    -> ratio-space R2={p['r2_ratio_space']:.3f} | "
              f"deploy R2={p['r2']:.3f} | oracle R2={p['r2_oracle']:.3f}")

    if "transductive_zscore" in methods:
        print("[3] transductive_zscore（工況感知標準化）...")
        results["transductive_zscore"] = _loco_hours(df, feats, conditions, per_condition_zscore)
        print(f"    -> pooled R2={results['transductive_zscore']['pooled']['r2']:.3f}")

    if "coral" in methods:
        print("[4] coral（協方差對齊）...")
        results["coral"] = _loco_hours(
            df, feats, conditions, lambda a, b, c: coral_align(a, b, c, reg))
        print(f"    -> pooled R2={results['coral']['pooled']['r2']:.3f}")

    # Best leakage-free method by pooled (hours) R^2 — oracle is excluded as it
    # uses a label-derived quantity.
    comparable = {k: v["pooled"]["r2"] for k, v in results.items()}
    best = max(comparable, key=comparable.get)
    summary = {
        "baseline_r2": results["baseline"]["pooled"]["r2"],
        "best_method": best,
        "best_r2": comparable[best],
        "ranking": sorted(comparable.items(), key=lambda kv: kv[1], reverse=True),
    }
    print(f"[done] 最佳（零洩漏，依 hours R2）：{best} = {comparable[best]:.3f} "
          f"（baseline {results['baseline']['pooled']['r2']:.3f}）")

    out_json = resolve(da.get("da_metrics", "outputs/metrics/xjtu_domain_adapt.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "leave_one_condition_out_domain_adaptation",
                "model": "RandomForestRegressor(n_estimators=300, random_state=42)",
                "features": feats,
                "coral_reg": reg,
                "results": results,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"    -> 指標：{out_json}")
    return out_json


if __name__ == "__main__":
    run()
