"""Feature-selection helpers used during the model-comparison sweep.

All selectors operate on already pre-processed (numeric, encoded) matrices.
They are designed to be fit on TRAINING data only, then applied to held-out
test data to avoid leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression


@dataclass
class SelectionResult:
    method: str
    selected: List[str]
    scores: pd.DataFrame  # columns: feature, score


def select_kbest(X_train: pd.DataFrame, y_train: pd.Series, k: int) -> SelectionResult:
    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    selected = [c for c, keep in zip(X_train.columns, mask) if keep]
    scores = pd.DataFrame({"feature": X_train.columns, "score": selector.scores_})
    scores = scores.sort_values("score", ascending=False, ignore_index=True)
    return SelectionResult(method="selectkbest", selected=selected, scores=scores)


def select_rfe(X_train: pd.DataFrame, y_train: pd.Series, k: int) -> SelectionResult:
    base = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    selector = RFE(base, n_features_to_select=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    mask = selector.support_
    selected = [c for c, keep in zip(X_train.columns, mask) if keep]
    ranking = pd.DataFrame({"feature": X_train.columns, "score": -selector.ranking_})
    ranking = ranking.sort_values("score", ascending=False, ignore_index=True)
    return SelectionResult(method="rfe", selected=selected, scores=ranking)


def select_rf_importance(
    X_train: pd.DataFrame, y_train: pd.Series, k: int, random_state: int = 42
) -> SelectionResult:
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=random_state
    )
    rf.fit(X_train, y_train)
    importance = pd.DataFrame(
        {"feature": X_train.columns, "score": rf.feature_importances_}
    ).sort_values("score", ascending=False, ignore_index=True)
    selected = importance["feature"].head(min(k, len(importance))).tolist()
    return SelectionResult(method="rf_importance", selected=selected, scores=importance)


def permutation_importance_scores(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return permutation importance for an already-fitted estimator."""
    result = permutation_importance(
        estimator, X_test, y_test,
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring="f1",
    )
    return pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def filter_columns(df: pd.DataFrame, selected: Sequence[str]) -> pd.DataFrame:
    """Return ``df`` restricted to ``selected`` columns (in their original order)."""
    keep = [c for c in selected if c in df.columns]
    return df[keep].copy()
