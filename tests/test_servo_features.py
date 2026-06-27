"""Module Servo — feature aggregation + feature-set definitions."""
import numpy as np
import pandas as pd
import pytest

from src.features.servo_features import (
    FEATURE_SETS,
    RAW_COLUMNS,
    add_position_error,
    all_feature_columns,
    build_feature_table,
    feature_set_columns,
    validate_raw_columns,
)


def _raw(n_per_run=20, runs=("LN", "HI")):
    rng = np.random.default_rng(0)
    rows = []
    for ri, lab in enumerate(runs):
        for _ in range(n_per_run):
            rows.append({
                "time": 0.0, "DV": 0.1 if lab == "LN" else 0.9,
                "rod_demand_pos": 50.0, "rod_actual_pos": 50.0 - rng.normal(0, 0.1),
                "torque": rng.normal(1, 0.1), "rotor_speed": rng.normal(2000, 5),
                "i_3p_a": rng.normal(3, 0.2), "i_3p_b": rng.normal(3, 0.2),
                "i_3p_c": rng.normal(3, 0.2), "direct": rng.normal(0.4, 0.05),
                "quadrature": rng.normal(1.2, 0.1), "run_index": ri,
                "transitions": 0, "del_pos": rng.normal(0.5, 0.02), "ylabel": lab,
            })
    return pd.DataFrame(rows)


def test_raw_columns_constant():
    assert RAW_COLUMNS[0] == "time" and "ylabel" in RAW_COLUMNS


def test_position_error_derived():
    df = add_position_error(_raw())
    assert "position_error" in df.columns


def test_build_feature_table_one_row_per_run():
    table = build_feature_table(_raw(runs=("LN", "MED", "HI")))
    assert len(table) == 3
    assert {"ylabel", "DV", "run_index"} <= set(table.columns)
    # every aggregated feature column should be present
    for c in all_feature_columns():
        assert c in table.columns


def test_feature_sets_resolve():
    for name in FEATURE_SETS:
        cols = feature_set_columns(name)
        assert cols, f"{name} has no columns"
    # full == union of the three component groups (de-duplicated)
    full = set(feature_set_columns("full"))
    for g in ("basic_motion", "current", "position_tracking"):
        assert set(feature_set_columns(g)) <= full


# --- real-data path protections ------------------------------------------------
def test_missing_required_column_raises():
    df = _raw().drop(columns=["rotor_speed"])
    with pytest.raises(ValueError, match="缺少必要欄位"):
        build_feature_table(df)


def test_all_nan_required_column_raises():
    df = _raw()
    df["torque"] = np.nan
    with pytest.raises(ValueError, match="整欄皆為"):
        validate_raw_columns(df)


def test_multi_file_run_index_kept_separate():
    # Two files, each with run_index 0/1 -> 4 distinct segments, not 2.
    a = _raw(runs=("LN", "HI"))
    b = _raw(runs=("LO", "MED"))
    a["__source_file__"] = "exp_a.csv"
    b["__source_file__"] = "exp_b.csv"
    table = build_feature_table(pd.concat([a, b], ignore_index=True))
    assert len(table) == 4
    assert set(table["ylabel"]) == {"LN", "HI", "LO", "MED"}
    assert table["run_index"].is_unique


def test_label_map_remaps_numeric_labels():
    df = _raw(runs=("LN", "HI"))
    df["ylabel"] = df["ylabel"].map({"LN": 0, "HI": 3})  # numeric encoded labels
    table = build_feature_table(df, label_map={0: "LN", 1: "LO", 2: "MED", 3: "HI"})
    assert set(table["ylabel"]) == {"LN", "HI"}


def test_unknown_label_raises():
    df = _raw(runs=("LN", "HI"))
    df["ylabel"] = df["ylabel"].map({"LN": 0, "HI": 3})  # no label_map provided
    with pytest.raises(ValueError, match="未知健康標籤"):
        build_feature_table(df)


def test_all_nan_ylabel_segment_raises():
    df = _raw(runs=("LN", "HI"))
    df.loc[df["run_index"] == 0, "ylabel"] = np.nan  # one run has no label
    with pytest.raises(ValueError, match="無法決定健康標籤"):
        build_feature_table(df)
