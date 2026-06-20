"""Post-training evaluation utilities.

Re-runs the best model on the test split to produce the confusion matrix,
ROC curve, precision-recall curve and (where available) feature-importance
plots.  Also computes permutation importance on the test fold.

Run after ``src.models.train`` (so that ``outputs/models/best_model.joblib``
exists)::

    python -m src.models.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.data.load_data import load_raw
from src.data.preprocess import make_split
from src.features.feature_selection import permutation_importance_scores
from src.utils.paths import ensure_output_dirs, load_config, resolve
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)


def _proba(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    d = estimator.decision_function(X)
    return (d - d.min()) / (d.max() - d.min() + 1e-9)


def _native_feature_importance(pipe) -> pd.DataFrame | None:
    """Return a DataFrame of native feature importances if the model exposes them."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    try:
        feature_names = list(pre.get_feature_names_out())
    except Exception:
        return None

    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        scores = np.abs(np.ravel(clf.coef_))
    else:
        return None
    if len(scores) != len(feature_names):
        return None
    return pd.DataFrame({"feature": feature_names, "importance": scores}).sort_values(
        "importance", ascending=False, ignore_index=True
    )


def run() -> Dict:
    ensure_output_dirs()
    cfg = load_config()
    fig_dir = resolve(cfg["paths"]["outputs_figures"])
    metrics_dir = resolve(cfg["paths"]["outputs_metrics"])

    bundle = joblib.load(resolve(cfg["paths"]["best_model"]))
    pipe = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    df = load_raw()
    split = make_split(df, include_engineered=True)
    X_test = split.X_test[feature_columns]
    y_test = split.y_test

    y_pred = pipe.predict(X_test)
    y_proba = _proba(pipe, X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Persist (y_true, y_proba) so the Streamlit dashboard can compute live
    # confusion matrices for any decision threshold without re-running inference.
    pd.DataFrame({"y_true": y_test.values, "y_proba": y_proba}).to_csv(
        metrics_dir / "test_predictions.csv", index=False
    )

    plot_confusion_matrix(cm, ["No failure", "Failure"], fig_dir / "confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plot_roc_curve(fpr, tpr, fig_dir / "roc_curve.png")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plot_pr_curve(precision, recall, fig_dir / "pr_curve.png")

    # Native importance (RF / GBM / linear coefs)
    importance_df = _native_feature_importance(pipe)
    if importance_df is not None:
        plot_feature_importance(
            importance_df.head(15), fig_dir / "feature_importance_native.png"
        )
        importance_df.to_csv(metrics_dir / "feature_importance_native.csv", index=False)

    # Permutation importance on a transformed test matrix
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_test_trans = pd.DataFrame(
        pre.transform(X_test), columns=list(pre.get_feature_names_out())
    )
    perm = permutation_importance_scores(clf, X_test_trans, y_test)
    perm.to_csv(metrics_dir / "feature_importance_permutation.csv", index=False)
    plot_feature_importance(
        perm.rename(columns={"importance_mean": "importance"}).head(15),
        fig_dir / "feature_importance_permutation.png",
    )

    out = {
        "model_name": bundle["model_name"],
        "feature_set": bundle["feature_set"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "summary_metrics": bundle["metrics"],
    }
    with open(metrics_dir / "best_model_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("評估完成。")
    print(f"  最佳模型 = {bundle['model_name']} / {bundle['feature_set']}")
    print(f"  混淆矩陣：\n{cm}")
    print(f"  評估結果已寫入 {metrics_dir / 'best_model_eval.json'}")

    # Auto-regenerate the Model Card so the markdown stays in sync with the
    # actual deployed model.
    try:
        from src.models.model_card import generate as _generate_card
        card_path = _generate_card()
        print(f"  模型卡已寫入 {card_path}")
    except Exception as e:  # pragma: no cover - defensive
        print(f"  ! 模型卡產生失敗：{e}")
    return out


if __name__ == "__main__":
    run()
