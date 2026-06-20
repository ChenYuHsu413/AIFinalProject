"""Tests for derived features and the leakage-prevention helper."""
from __future__ import annotations

import pandas as pd

from src.data.preprocess import strip_id_and_leakage
from src.features.feature_engineering import (
    ENGINEERED_COLUMNS,
    add_engineered_features,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": [1, 2],
            "Product ID": ["L1", "M2"],
            "Type": ["L", "M"],
            "Air temperature [K]": [298.1, 299.0],
            "Process temperature [K]": [308.6, 309.5],
            "Rotational speed [rpm]": [1551, 1408],
            "Torque [Nm]": [42.8, 46.3],
            "Tool wear [min]": [108, 3],
            "Machine failure": [0, 1],
            "TWF": [0, 0], "HDF": [0, 1], "PWF": [0, 0],
            "OSF": [0, 0], "RNF": [0, 0],
        }
    )


def test_engineered_features_added_with_expected_values():
    df = _sample_df()
    out = add_engineered_features(df)
    for c in ENGINEERED_COLUMNS:
        assert c in out.columns

    assert out.loc[0, "temp_diff"] == 308.6 - 298.1
    assert out.loc[0, "power_proxy"] == 42.8 * 1551
    assert out.loc[1, "wear_torque_interaction"] == 3 * 46.3
    assert out.loc[1, "temp_wear_interaction"] == 309.5 * 3


def test_strip_id_and_leakage_drops_id_and_failure_type_columns():
    df = _sample_df()
    out = strip_id_and_leakage(df)
    for col in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert col not in out.columns
    # Target & features must survive.
    assert "Machine failure" in out.columns
    assert "Torque [Nm]" in out.columns
