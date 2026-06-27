"""Module Servo — training-simulator robustness on small / imbalanced data."""
import numpy as np
import pandas as pd

from src.features.servo_features import all_feature_columns
from src.models import servo_simulator as sim


def _feature_df(counts):
    """Build a feature table with the given per-class row counts."""
    rng = np.random.default_rng(0)
    cols = all_feature_columns()
    rows = []
    for lab, n in counts.items():
        sev = {"LN": 0.05, "LO": 0.35, "MED": 0.65, "HI": 0.92}[lab]
        for _ in range(n):
            feat = {c: float(rng.normal(sev, 0.1)) for c in cols}
            feat["ylabel"] = lab
            feat["DV"] = float(np.clip(sev + rng.normal(0, 0.05), 0, 1))
            rows.append(feat)
    return pd.DataFrame(rows)


def test_run_classification_survives_singleton_class():
    # HI has a single member -> stratified split would crash; the guard falls
    # back to a non-stratified split instead of raising.
    df = _feature_df({"LN": 20, "LO": 20, "MED": 20, "HI": 1})
    res = sim.run_classification(df, "engineered", "decision_tree", n_samples=1000)
    assert res["task"] == "classification"
    assert 0.0 <= res["accuracy"] <= 1.0


def test_run_regression_basic():
    df = _feature_df({"LN": 20, "LO": 20, "MED": 20, "HI": 20})
    res = sim.run_regression(df, "engineered", "ridge", n_samples=1000)
    assert res["task"] == "regression" and "rmse" in res
