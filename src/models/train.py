"""Train every (model x feature-set) combination and persist the best one.

Outputs
-------
outputs/metrics/model_comparison.csv
    Tabular comparison of every run.
outputs/models/best_model.joblib
    The single best estimator, wrapped in a ``Pipeline`` with the
    ``ColumnTransformer`` it was trained with, plus the feature names it
    expects.
outputs/models/best_model_meta.json
    Metadata (model name, feature set, metrics) for the best model so the
    Streamlit / FastAPI layer can describe it without re-loading the pipeline.
outputs/figures/*.png
    Model comparison and feature-set comparison charts.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.load_data import load_raw
from src.data.preprocess import (
    DataSplit,
    get_feature_names,
    make_column_transformer,
    make_split,
)
from src.features.feature_engineering import ENGINEERED_COLUMNS
from src.features.feature_selection import (
    select_kbest,
    select_rf_importance,
    select_rfe,
)
from src.models.model_registry import (
    NEEDS_SCALING,
    available_models,
    build,
)
from src.utils.paths import ensure_output_dirs, load_config, resolve
from src.visualization.plots import (
    plot_feature_count_vs_metric,
    plot_metric_comparison,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ---------------------------------------------------------------------------
# Feature-set definitions
# ---------------------------------------------------------------------------
@dataclass
class FeatureSet:
    name: str
    description: str
    columns: List[str] = field(default_factory=list)


def build_feature_sets(split: DataSplit, cfg: dict) -> List[FeatureSet]:
    """Create the comparison feature sets.

    Selectors that need a fitted matrix are run on the *training* fold only,
    using a temporary fully-scaled+encoded matrix.  Their output is the set
    of column names to keep — the actual pipeline is rebuilt later with the
    real ``ColumnTransformer``.
    """
    sets: List[FeatureSet] = []
    raw_numeric = [c for c in split.numeric_cols if c not in ENGINEERED_COLUMNS]
    cat = split.categorical_cols

    # A — raw only
    sets.append(FeatureSet(
        name="A_baseline",
        description="Original raw features only (Type + 5 numeric).",
        columns=raw_numeric + cat,
    ))

    # B — raw + engineered
    sets.append(FeatureSet(
        name="B_engineered",
        description="Raw + 5 engineered features.",
        columns=raw_numeric + ENGINEERED_COLUMNS + cat,
    ))

    # For C/D/E we score on a transformed training matrix that already
    # contains engineered features.
    full_pipe = make_column_transformer(split.numeric_cols, cat, scale_numeric=True)
    X_train_mat = full_pipe.fit_transform(split.X_train)
    feature_names_full = get_feature_names(full_pipe)
    X_train_df = pd.DataFrame(X_train_mat, columns=feature_names_full)
    y_train = split.y_train

    def _back_to_source(selected_names: List[str]) -> List[str]:
        """Translate transformed names back to source columns.

        ``num__Torque [Nm]`` -> ``Torque [Nm]``;
        ``cat__Type_L`` -> ``Type``.  We keep a source column whenever ANY of
        its derived transformed columns survives selection.
        """
        keep: List[str] = []
        for n in selected_names:
            if n.startswith("num__"):
                src = n[len("num__"):]
                if src not in keep:
                    keep.append(src)
            elif n.startswith("cat__"):
                if "Type" not in keep:
                    keep.append("Type")
            else:
                if n not in keep:
                    keep.append(n)
        return keep

    for spec in cfg["feature_sets"]:
        if spec["method"] == "raw" or spec["method"] == "engineered":
            continue  # already added above
        k = spec.get("k", 8)
        if spec["method"] == "selectkbest":
            res = select_kbest(X_train_df, y_train, k=k)
        elif spec["method"] == "rfe":
            res = select_rfe(X_train_df, y_train, k=k)
        elif spec["method"] == "rf_importance":
            res = select_rf_importance(X_train_df, y_train, k=k)
        else:
            raise ValueError(f"Unknown selector method: {spec['method']}")
        sets.append(FeatureSet(
            name=spec["name"],
            description=spec["description"],
            columns=_back_to_source(res.selected),
        ))
    return sets


# ---------------------------------------------------------------------------
# Single (model, feature-set) training run
# ---------------------------------------------------------------------------
def _evaluate(estimator, X_test, y_test) -> Dict[str, float]:
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        # Map decision function to a pseudo-probability in [0, 1]
        d = estimator.decision_function(X_test)
        y_proba = (d - d.min()) / (d.max() - d.min() + 1e-9)
    else:
        y_proba = y_pred.astype(float)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def _train_one(
    model_name: str, fs: FeatureSet, split: DataSplit, random_state: int
) -> Dict[str, Any]:
    cat = [c for c in split.categorical_cols if c in fs.columns]
    num = [c for c in fs.columns if c not in cat]
    scale = model_name in NEEDS_SCALING
    ct = make_column_transformer(num, cat, scale_numeric=scale)
    estimator = build(model_name, random_state=random_state)
    pipe = Pipeline([("pre", ct), ("clf", estimator)])

    X_train = split.X_train[fs.columns].copy()
    X_test = split.X_test[fs.columns].copy()
    pipe.fit(X_train, split.y_train)
    metrics = _evaluate(pipe, X_test, split.y_test)
    return {
        "pipeline": pipe,
        "metrics": metrics,
        "feature_columns": fs.columns,
        "model_name": model_name,
        "feature_set": fs.name,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run() -> Path:
    ensure_output_dirs()
    cfg = load_config()
    random_state = cfg["modeling"]["random_state"]

    print("[1/5] 載入原始資料集...")
    df = load_raw()
    print(f"    -> 資料維度 = {df.shape}")

    print("[2/5] 建立 train/test 切分與衍生特徵...")
    split = make_split(df, include_engineered=True)
    print(f"    -> X_train {split.X_train.shape}, X_test {split.X_test.shape}")
    print(f"    -> 訓練集故障比例 = {split.y_train.mean():.4f}")
    print(f"    -> 測試集故障比例 = {split.y_test.mean():.4f}")

    print("[3/5] 建立特徵組合...")
    feature_sets = build_feature_sets(split, cfg)
    for fs in feature_sets:
        print(f"    - {fs.name}: {len(fs.columns)} 欄 -> {fs.columns}")

    print("[4/5] 訓練模型...")
    model_names = available_models(cfg["modeling"]["enabled_models"])
    print(f"    -> 啟用模型：{model_names}")

    rows: List[Dict[str, Any]] = []
    fitted: Dict[str, Any] = {}
    for fs in feature_sets:
        for mn in model_names:
            try:
                res = _train_one(mn, fs, split, random_state=random_state)
            except Exception as e:  # pragma: no cover - defensive
                print(f"      ! {mn}/{fs.name} 訓練失敗：{e}")
                continue
            rows.append({
                "model_name": mn,
                "feature_set": fs.name,
                "feature_count": len(fs.columns),
                "selected_features": ";".join(fs.columns),
                **res["metrics"],
            })
            fitted[(mn, fs.name)] = res
            m = res["metrics"]
            print(
                f"    - {mn:<22s} | {fs.name:<22s} "
                f"acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
                f"rec={m['recall']:.3f} f1={m['f1']:.3f} "
                f"roc={m['roc_auc']:.3f} pr={m['pr_auc']:.3f}"
            )

    comparison = pd.DataFrame(rows).sort_values(
        cfg["modeling"]["scoring_for_best"], ascending=False, ignore_index=True
    )
    metrics_csv = resolve(cfg["paths"]["metrics_csv"])
    comparison.to_csv(metrics_csv, index=False)
    print(f"    -> 比較表已寫入 {metrics_csv}")

    # ---- pick best & persist
    print("[5/5] 挑選最佳模型並儲存產物...")
    best = comparison.iloc[0].to_dict()
    best_key = (best["model_name"], best["feature_set"])
    best_artefact = fitted[best_key]

    bundle = {
        "pipeline": best_artefact["pipeline"],
        "feature_columns": best_artefact["feature_columns"],
        "model_name": best_artefact["model_name"],
        "feature_set": best_artefact["feature_set"],
        "metrics": best_artefact["metrics"],
    }
    model_path = resolve(cfg["paths"]["best_model"])
    joblib.dump(bundle, model_path)

    meta = {
        "model_name": bundle["model_name"],
        "feature_set": bundle["feature_set"],
        "feature_columns": bundle["feature_columns"],
        "metrics": bundle["metrics"],
        "selection_criterion": cfg["modeling"]["scoring_for_best"],
    }
    meta_path = resolve(cfg["paths"]["best_model_meta"])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"    -> 最佳模型 = {bundle['model_name']} / {bundle['feature_set']}")
    print(f"    -> 模型已儲存：{model_path}")
    print(f"    -> 中繼資料已儲存：{meta_path}")

    # ---- comparison plots
    fig_dir = resolve(cfg["paths"]["outputs_figures"])
    for metric in ["recall", "f1", "roc_auc", "pr_auc"]:
        plot_metric_comparison(comparison, metric, fig_dir / f"compare_{metric}.png")
    for metric in ["f1", "recall", "pr_auc"]:
        plot_feature_count_vs_metric(
            comparison, metric, fig_dir / f"feature_count_vs_{metric}.png"
        )
    print(f"    -> 圖表已輸出到 {fig_dir}")

    return model_path


if __name__ == "__main__":
    run()
