"""Module Servo — feature aggregation + feature-set definitions."""
import numpy as np
import pandas as pd

from src.features.servo_features import (
    FEATURE_SETS,
    RAW_COLUMNS,
    add_position_error,
    all_feature_columns,
    build_feature_table,
    feature_set_columns,
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
