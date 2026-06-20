"""Second-stage failure-type classifiers (TWF / HDF / PWF / OSF / RNF).

The first-stage model decides *whether* a unit is likely to fail.  This
module trains a *separate* binary classifier per failure type so that, given
the same operating snapshot, we can also estimate *which* failure mode is
the most plausible cause.

Each per-type model uses:
* the same raw + engineered features as the first stage,
* RandomForest with ``class_weight="balanced"`` (per-type positives are rare),
* a stratified 80/20 split,
* a metric report covering n positives, precision, recall, F1, ROC-AUC, PR-AUC.

Note on **RNF (Random failure)**: by construction this label has no
deterministic signal in the AI4I dataset. The model is expected to perform
near chance on it; we report the result honestly rather than hiding it.

Run::

    python -m src.models.train_failure_types
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load_data import load_raw
from src.data.preprocess import make_column_transformer
from src.features.feature_engineering import add_engineered_features
from src.utils.paths import ensure_output_dirs, load_config, resolve


def _prepare_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop IDs / Machine failure / OTHER failure-type cols and add engineered features."""
    id_cols = cfg["columns"]["id_columns"]
    failure_types = cfg["columns"]["failure_types"]
    primary = cfg["columns"]["target_primary"]
    drop = list(id_cols) + list(failure_types) + [primary]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return add_engineered_features(X)


def _train_one(failure_type: str, df: pd.DataFrame, cfg: dict) -> Dict[str, Any]:
    X = _prepare_features(df, cfg)
    y = df[failure_type].astype(int)

    categorical_cols = ["Type"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Some failure types (RNF) have very few positives; only stratify when safe.
    stratify_arg = y if y.sum() >= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=stratify_arg,
    )

    ct = make_column_transformer(numeric_cols, categorical_cols, scale_numeric=False)
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1,
        random_state=cfg["modeling"]["random_state"],
    )
    pipe = Pipeline([("pre", ct), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    has_pos = y_test.nunique() > 1
    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_positives_train": int(y_train.sum()),
        "n_positives_test": int(y_test.sum()),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if has_pos else float("nan"),
        "pr_auc": float(average_precision_score(y_test, y_proba)) if has_pos else float("nan"),
    }
    return {
        "pipeline": pipe,
        "feature_columns": list(X.columns),
        "metrics": metrics,
    }


def run() -> None:
    ensure_output_dirs()
    cfg = load_config()
    df = load_raw()

    failure_types: List[str] = cfg["columns"]["failure_types"]
    print(f"[1/3] 載入資料：shape = {df.shape}")
    print(f"    -> 故障類型計數：" + ", ".join(
        f"{ft}={int(df[ft].sum())}" for ft in failure_types
    ))

    print("[2/3] 對每個故障類型分別訓練二元分類器...")
    bundle: Dict[str, Dict[str, Any]] = {}
    rows = []
    for ft in failure_types:
        res = _train_one(ft, df, cfg)
        bundle[ft] = res
        m = res["metrics"]
        rows.append({"failure_type": ft, **m})
        print(
            f"    - {ft}: pos_train={m['n_positives_train']:>3} pos_test={m['n_positives_test']:>3}"
            f"  prec={m['precision']:.3f} rec={m['recall']:.3f} f1={m['f1']:.3f}"
            f"  roc={m['roc_auc']:.3f} pr={m['pr_auc']:.3f}"
        )

    print("[3/3] 儲存模型與比較表...")
    model_path = resolve(cfg["paths"]["failure_type_model"])
    joblib.dump(bundle, model_path)
    metrics_path = resolve(cfg["paths"]["failure_type_metrics"])
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"    -> 模型儲存於 {model_path}")
    print(f"    -> 指標寫入 {metrics_path}")


if __name__ == "__main__":
    run()
