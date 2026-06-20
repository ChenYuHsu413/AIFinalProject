"""Optuna-based hyperparameter tuning for the top-N models.

Pipeline
--------
1. Read ``outputs/metrics/model_comparison.csv`` and take the top-N rows by
   the configured ``scoring_for_best`` metric (default F1).
2. For each of those (model, feature_set) pairs, run an Optuna study that
   maximises the same metric on stratified k-fold CV of the *training* fold.
3. Refit the best trial on the full training fold, score it on the held-out
   test fold, and record the result.
4. Pick the overall winner; if it beats the previous ``best_model.joblib``,
   archive the old one to ``best_model_pretuned.joblib`` and replace it.

Run::

    python -m src.models.tune
"""
from __future__ import annotations

import json
import time
import warnings
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data.load_data import load_raw
from src.data.preprocess import make_column_transformer, make_split
from src.models.evaluate import _proba  # reuse the probability helper
from src.models.model_registry import NEEDS_SCALING
from src.models.train import build_feature_sets
from src.utils.paths import ensure_output_dirs, load_config, resolve

# Optional libraries — same gate as model_registry
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier  # type: ignore
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Per-model search spaces + factory
# ---------------------------------------------------------------------------
SEARCH_SPACES: Dict[str, Callable[[optuna.Trial, int], Any]] = {}


def _gb(trial: optuna.Trial, rs: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        random_state=rs,
    )


def _rf(trial: optuna.Trial, rs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100),
        max_depth=trial.suggest_int("max_depth", 4, 24),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced",
        n_jobs=-1,
        random_state=rs,
    )


def _lr(trial: optuna.Trial, rs: int) -> LogisticRegression:
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 10.0, log=True),
        solver=trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
        max_iter=2000,
        class_weight="balanced",
        random_state=rs,
    )


def _svm(trial: optuna.Trial, rs: int) -> SVC:
    return SVC(
        C=trial.suggest_float("C", 1e-2, 50.0, log=True),
        gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=rs,
    )


def _mlp(trial: optuna.Trial, rs: int) -> MLPClassifier:
    layer1 = trial.suggest_int("layer1", 16, 128, step=16)
    layer2 = trial.suggest_int("layer2", 0, 64, step=16)
    layers = (layer1, layer2) if layer2 else (layer1,)
    return MLPClassifier(
        hidden_layer_sizes=layers,
        alpha=trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        learning_rate_init=trial.suggest_float("lr_init", 1e-4, 1e-2, log=True),
        max_iter=500,
        random_state=rs,
    )


SEARCH_SPACES["gradient_boosting"] = _gb
SEARCH_SPACES["random_forest"] = _rf
SEARCH_SPACES["logistic_regression"] = _lr
SEARCH_SPACES["svm"] = _svm
SEARCH_SPACES["mlp"] = _mlp

if HAS_XGB:
    def _xgb(trial: optuna.Trial, rs: int) -> "XGBClassifier":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=rs,
        )
    SEARCH_SPACES["xgboost"] = _xgb

if HAS_LGBM:
    def _lgbm(trial: optuna.Trial, rs: int) -> "LGBMClassifier":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=50),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            class_weight="balanced",
            n_jobs=-1,
            random_state=rs,
            verbose=-1,
        )
    SEARCH_SPACES["lightgbm"] = _lgbm


# ---------------------------------------------------------------------------
# Single-study runner
# ---------------------------------------------------------------------------
def _build_pipeline(model_name: str, X_train_cols, fs_columns, estimator) -> Pipeline:
    cat = [c for c in ["Type"] if c in fs_columns]
    num = [c for c in fs_columns if c not in cat]
    scale = model_name in NEEDS_SCALING
    ct = make_column_transformer(num, cat, scale_numeric=scale)
    return Pipeline([("pre", ct), ("clf", estimator)])


def _tune_model(model_name: str, fs, split, cfg: dict) -> Dict[str, Any]:
    if model_name not in SEARCH_SPACES:
        print(f"    -> skip {model_name}: 沒有定義搜尋空間")
        return {}
    rs = cfg["modeling"]["random_state"]
    scoring = cfg["modeling"]["scoring_for_best"]
    n_trials = cfg["tuning"]["n_trials"]
    cv_folds = cfg["tuning"]["cv_folds"]
    timeout = cfg["tuning"].get("timeout_per_model")

    X_train = split.X_train[fs.columns]
    y_train = split.y_train
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
    factory = SEARCH_SPACES[model_name]

    def objective(trial: optuna.Trial) -> float:
        estimator = factory(trial, rs)
        pipe = _build_pipeline(model_name, X_train.columns, fs.columns, estimator)
        scores = cross_val_score(pipe, X_train, y_train, scoring=scoring,
                                  cv=cv, n_jobs=-1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=rs)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=False)
    dt = time.time() - t0
    print(f"    -> {n_trials} trials in {dt:.1f}s, best CV {scoring} = {study.best_value:.4f}")
    print(f"       best params = {study.best_params}")

    # Refit best on full train, score on held-out test
    best_estimator = factory(_FrozenTrial(study.best_params), rs)
    final_pipe = _build_pipeline(model_name, X_train.columns, fs.columns, best_estimator)
    final_pipe.fit(X_train, y_train)
    X_test = split.X_test[fs.columns]
    y_test = split.y_test
    y_proba = _proba(final_pipe, X_test)
    y_pred = final_pipe.predict(X_test)

    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
        precision_score, recall_score, roc_auc_score,
    )
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }

    trial_rows = [
        {
            "model_name": model_name,
            "feature_set": fs.name,
            "trial": t.number,
            "cv_score": t.value,
            **{f"param__{k}": v for k, v in t.params.items()},
        }
        for t in study.trials
        if t.value is not None
    ]
    return {
        "model_name": model_name,
        "feature_set": fs.name,
        "best_params": study.best_params,
        "best_cv_score": float(study.best_value),
        "test_metrics": test_metrics,
        "final_pipeline": final_pipe,
        "trials": trial_rows,
    }


class _FrozenTrial:
    """Tiny shim so we can re-call the factory with already-decided params."""
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def suggest_int(self, name, *_args, **_kw):
        return self.params[name]

    def suggest_float(self, name, *_args, **_kw):
        return self.params[name]

    def suggest_categorical(self, name, *_args, **_kw):
        return self.params[name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    ensure_output_dirs()
    cfg = load_config()
    scoring = cfg["modeling"]["scoring_for_best"]
    top_n = cfg["tuning"]["top_n_models"]

    # ---- read previous comparison ----
    comp_path = resolve(cfg["paths"]["metrics_csv"])
    if not comp_path.exists():
        raise FileNotFoundError(
            f"找不到 {comp_path}。請先執行 `python -m src.models.train`。"
        )
    comparison = pd.read_csv(comp_path)
    targets = (
        comparison.sort_values(scoring, ascending=False)
        .drop_duplicates(subset=["model_name"])
        .head(top_n)[["model_name", "feature_set", scoring]]
        .reset_index(drop=True)
    )
    print(f"[1/4] 挑選 top-{top_n} 模型進行調參（依 {scoring}）：")
    print(targets.to_string(index=False))

    # ---- build split + feature sets ----
    print("[2/4] 重建 train/test split 與特徵組合...")
    df = load_raw()
    split = make_split(df, include_engineered=True)
    feature_sets = {fs.name: fs for fs in build_feature_sets(split, cfg)}

    # ---- run studies ----
    print(f"[3/4] 對每個模型跑 Optuna ({cfg['tuning']['n_trials']} trials × {cfg['tuning']['cv_folds']}-fold CV)...")
    studies: List[Dict[str, Any]] = []
    all_trials: List[Dict[str, Any]] = []
    for _, row in targets.iterrows():
        mn = row["model_name"]
        fs_name = row["feature_set"]
        if fs_name not in feature_sets:
            print(f"    -> skip {mn}/{fs_name}: 找不到該特徵組合")
            continue
        print(f"  - 調參 {mn} / {fs_name}（原始 {scoring} = {row[scoring]:.4f}）")
        result = _tune_model(mn, feature_sets[fs_name], split, cfg)
        if result:
            studies.append(result)
            all_trials.extend(result["trials"])

    if not studies:
        print("沒有任何模型完成調參。結束。")
        return

    # ---- save trial history ----
    history_path = resolve(cfg["paths"]["tuning_history"])
    pd.DataFrame(all_trials).to_csv(history_path, index=False)
    print(f"    -> trial 紀錄寫入 {history_path}")

    # ---- pick overall winner ----
    print("[4/4] 比較調參後的測試集結果...")
    summary = pd.DataFrame([
        {
            "model_name": s["model_name"],
            "feature_set": s["feature_set"],
            "best_cv_score": s["best_cv_score"],
            **s["test_metrics"],
        }
        for s in studies
    ]).sort_values(scoring, ascending=False, ignore_index=True)
    print(summary.to_string(index=False))

    winner = max(studies, key=lambda s: s["test_metrics"][scoring])
    # ---- compare with current persisted best ----
    pre_path = resolve(cfg["paths"]["best_model"])
    pre_metric = None
    if pre_path.exists():
        try:
            prev = joblib.load(pre_path)
            pre_metric = prev["metrics"].get(scoring)
        except Exception:
            pre_metric = None

    if pre_metric is not None and winner["test_metrics"][scoring] <= pre_metric:
        print(
            f"調參結果 ({winner['test_metrics'][scoring]:.4f}) 未勝過原始最佳 "
            f"({pre_metric:.4f})。保留原始 best_model.joblib。"
        )
    else:
        # backup old, replace
        if pre_path.exists():
            backup = resolve(cfg["paths"]["pretuned_model_backup"])
            joblib.dump(joblib.load(pre_path), backup)
            print(f"    -> 原始 best_model 備份至 {backup}")
        bundle = {
            "pipeline": winner["final_pipeline"],
            "feature_columns": list(feature_sets[winner["feature_set"]].columns),
            "model_name": winner["model_name"],
            "feature_set": winner["feature_set"],
            "metrics": winner["test_metrics"],
        }
        joblib.dump(bundle, pre_path)
        meta = {
            "model_name": bundle["model_name"],
            "feature_set": bundle["feature_set"],
            "feature_columns": bundle["feature_columns"],
            "metrics": bundle["metrics"],
            "selection_criterion": scoring,
            "tuned": True,
            "best_params": winner["best_params"],
        }
        with open(resolve(cfg["paths"]["best_model_meta"]), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(
            f"    -> 採用調參後的 {bundle['model_name']} / {bundle['feature_set']}"
            f"（{scoring} {winner['test_metrics'][scoring]:.4f}）。"
        )

    # save params for inspection
    tuned_params_path = resolve(cfg["paths"]["tuned_params"])
    with open(tuned_params_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "model_name": s["model_name"],
                    "feature_set": s["feature_set"],
                    "best_params": s["best_params"],
                    "best_cv_score": s["best_cv_score"],
                    "test_metrics": s["test_metrics"],
                }
                for s in studies
            ],
            f, indent=2, ensure_ascii=False,
        )
    print(f"    -> 所有調參結果寫入 {tuned_params_path}")


if __name__ == "__main__":
    run()
