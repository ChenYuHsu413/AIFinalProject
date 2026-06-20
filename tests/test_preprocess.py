"""Tests for the preprocessing pipeline (no leakage, correct shapes)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocess import make_column_transformer, split_X_y


def _synth_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "UDI": np.arange(n),
            "Product ID": [f"L{i}" for i in range(n)],
            "Type": rng.choice(["L", "M", "H"], n),
            "Air temperature [K]": rng.normal(298, 2, n),
            "Process temperature [K]": rng.normal(309, 1.5, n),
            "Rotational speed [rpm]": rng.normal(1500, 200, n),
            "Torque [Nm]": rng.normal(40, 8, n),
            "Tool wear [min]": rng.integers(0, 250, n),
            "Machine failure": rng.integers(0, 2, n),
            "TWF": np.zeros(n, dtype=int),
            "HDF": np.zeros(n, dtype=int),
            "PWF": np.zeros(n, dtype=int),
            "OSF": np.zeros(n, dtype=int),
            "RNF": np.zeros(n, dtype=int),
        }
    )


def test_split_X_y_excludes_leaky_columns_and_includes_engineered():
    df = _synth_df()
    X, y, num, cat = split_X_y(df, include_engineered=True)
    leak = {"Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"}
    assert leak.isdisjoint(X.columns)
    assert "temp_diff" in X.columns
    assert cat == ["Type"]
    assert len(y) == len(df)
    assert set(num).issubset(set(X.columns))


def test_column_transformer_scales_numeric_only_when_requested():
    df = _synth_df()
    X, _, num, cat = split_X_y(df, include_engineered=True)
    ct_scaled = make_column_transformer(num, cat, scale_numeric=True)
    ct_raw = make_column_transformer(num, cat, scale_numeric=False)

    X_scaled = ct_scaled.fit_transform(X)
    X_raw = ct_raw.fit_transform(X)

    # Numeric block is the first slice in both — check standardisation behaviour.
    n_num = len(num)
    assert abs(X_scaled[:, :n_num].mean(axis=0)).max() < 1e-6
    assert abs(X_scaled[:, :n_num].std(axis=0) - 1.0).max() < 1e-2
    # Raw passthrough should NOT be unit-variance.
    assert (abs(X_raw[:, :n_num].std(axis=0) - 1.0).max() > 0.1)
