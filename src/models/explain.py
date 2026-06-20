"""Per-prediction explainability using SHAP TreeExplainer.

Only the *best* persisted model is explained. Only tree-based models are
supported here (TreeExplainer is fast and exact). For non-tree best models
the helper raises ``NotImplementedError`` and the UI falls back gracefully.

The values returned are log-odds contributions (the natural SHAP space for
binary classifiers), so summing ``shap_values`` and adding ``base_value``
reproduces the model's raw output, not the probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.predict import load_model, prepare_input


# Tree-based estimators TreeExplainer can handle.
_TREE_MODELS = {
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "DecisionTreeClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "ExtraTreesClassifier",
}


@dataclass
class ShapExplanation:
    feature_names: List[str]
    feature_values: List[float]
    shap_values: List[float]
    base_value: float
    model_output: float


_EXPLAINER: Any = None
_FEATURE_NAMES: Optional[List[str]] = None


def _pretty(name: str) -> str:
    """Strip the ColumnTransformer prefix (``num__`` / ``cat__``)."""
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def is_supported() -> bool:
    """Return True iff the best model is something TreeExplainer can handle."""
    try:
        bundle = load_model()
    except FileNotFoundError:
        return False
    clf_name = type(bundle.pipeline.named_steps["clf"]).__name__
    return clf_name in _TREE_MODELS


def _ensure_explainer():
    """Build (and cache) the TreeExplainer for the best model."""
    global _EXPLAINER, _FEATURE_NAMES
    if _EXPLAINER is not None:
        return _EXPLAINER, _FEATURE_NAMES

    import shap

    bundle = load_model()
    pipe = bundle.pipeline
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    clf_name = type(clf).__name__
    if clf_name not in _TREE_MODELS:
        raise NotImplementedError(
            f"目前 SHAP 解釋僅支援樹模型；目前最佳模型為 {clf_name}。"
        )
    _EXPLAINER = shap.TreeExplainer(clf)
    _FEATURE_NAMES = [_pretty(n) for n in pre.get_feature_names_out()]
    return _EXPLAINER, _FEATURE_NAMES


def _binary_positive_class(arr: np.ndarray) -> np.ndarray:
    """Normalise SHAP outputs into a (n_samples, n_features) positive-class slice.

    SHAP's API for binary classifiers has evolved across versions:
    - some return a single (n, k) array (already positive-class log-odds),
    - some return a list of two arrays (one per class),
    - some return (n, k, 2) ndarray.
    """
    if isinstance(arr, list):
        if len(arr) == 2:
            return np.asarray(arr[1])
        return np.asarray(arr[0])
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # (n, k, n_classes) -> positive class
        return arr[:, :, -1]
    return arr


def _scalar_base_value(base: Any) -> float:
    """Normalise SHAP's expected_value into a single positive-class scalar."""
    if isinstance(base, (list, tuple)):
        return float(base[1]) if len(base) == 2 else float(base[0])
    arr = np.asarray(base).ravel()
    if arr.size == 1:
        return float(arr[0])
    if arr.size == 2:
        return float(arr[1])
    return float(arr[0])


def explain_record(record: Dict[str, Any]) -> ShapExplanation:
    """Compute SHAP contributions for a single raw input record."""
    bundle = load_model()
    explainer, feature_names = _ensure_explainer()

    pre = bundle.pipeline.named_steps["pre"]
    df = prepare_input(record)[bundle.feature_columns]
    X_trans = np.asarray(pre.transform(df), dtype=float)

    sv = _binary_positive_class(explainer.shap_values(X_trans))
    base = _scalar_base_value(explainer.expected_value)
    contrib = sv[0]
    model_output = float(base + contrib.sum())

    return ShapExplanation(
        feature_names=list(feature_names),
        feature_values=[float(v) for v in X_trans[0]],
        shap_values=[float(v) for v in contrib],
        base_value=float(base),
        model_output=model_output,
    )
