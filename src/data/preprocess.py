"""Pre-processing: split, target/feature separation, ColumnTransformer.

The pre-processor is wrapped in a ``Pipeline`` and only fitted on the training
fold to prevent leakage from the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_engineering import (
    ENGINEERED_COLUMNS,
    add_engineered_features,
)
from src.utils.paths import load_config


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_cols: List[str]
    categorical_cols: List[str]


def strip_id_and_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID columns and per-failure-type columns to avoid leakage.

    The five failure-type columns (TWF/HDF/PWF/OSF/RNF) are themselves
    deterministic causes of ``Machine failure``; using them as features would
    leak the target.  See README for the second-stage analysis that *does*
    legitimately use them.
    """
    cfg = load_config()
    drop = list(cfg["columns"]["id_columns"]) + list(cfg["columns"]["failure_types"])
    return df.drop(columns=[c for c in drop if c in df.columns])


def split_X_y(
    df: pd.DataFrame, include_engineered: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Build the feature matrix and the binary target."""
    cfg = load_config()
    target = cfg["columns"]["target_primary"]

    df = strip_id_and_leakage(df)
    if include_engineered:
        df = add_engineered_features(df)

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    categorical_cols = list(cfg["columns"]["categorical"])
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return X, y, numeric_cols, categorical_cols


def make_split(
    df: pd.DataFrame, include_engineered: bool = True
) -> DataSplit:
    cfg = load_config()
    sp = cfg["split"]

    X, y, num_cols, cat_cols = split_X_y(df, include_engineered=include_engineered)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=sp["test_size"],
        random_state=sp["random_state"],
        stratify=y if sp["stratify"] else None,
    )
    return DataSplit(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
    )


def make_column_transformer(
    numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool = True
) -> ColumnTransformer:
    """Build a ``ColumnTransformer``.

    Numeric scaling is toggleable so that tree-based models can skip it.
    """
    # OneHotEncoder API changed: ``sparse_output`` in sklearn >= 1.2.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - very old sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = [("cat", ohe, categorical_cols)]
    if scale_numeric:
        transformers.insert(0, ("num", StandardScaler(), numeric_cols))
    else:
        transformers.insert(0, ("num", "passthrough", numeric_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(ct: ColumnTransformer) -> List[str]:
    """Return the (post-transform) feature names produced by ``ct``."""
    try:
        return list(ct.get_feature_names_out())
    except Exception:  # pragma: no cover - defensive fallback
        names: List[str] = []
        for name, trans, cols in ct.transformers_:
            if trans == "drop":
                continue
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols))
            else:
                names.extend(cols)
        return names


__all__ = [
    "DataSplit",
    "strip_id_and_leakage",
    "split_X_y",
    "make_split",
    "make_column_transformer",
    "get_feature_names",
    "ENGINEERED_COLUMNS",
]
