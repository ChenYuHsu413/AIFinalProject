"""Streaming PHM-zip aggregator — online stats match aggregate_run; zero-guard."""
import numpy as np
import pandas as pd

from src.data import build_servo_from_zip as B
from src.features.servo_features import aggregate_run, all_feature_columns


def _raw(n=40):
    rng = np.random.default_rng(0)
    rows = []
    for ri, lab in ((0, "LN"), (1, "HI")):
        for _ in range(n):
            rows.append({
                "DV": 0.1 if lab == "LN" else 3000.0,
                "rod_demand_pos": 50.0, "rod_actual_pos": 50.0 - rng.normal(0, 0.1),
                "torque": rng.normal(1, 0.1), "rotor_speed": rng.normal(2000, 5),
                "i_3p_a": rng.normal(3, 0.2), "i_3p_b": rng.normal(3, 0.2),
                "i_3p_c": rng.normal(3, 0.2), "direct": rng.normal(0.4, 0.05),
                "quadrature": rng.normal(1.2, 0.1), "run_index": ri,
                "del_pos": rng.normal(0.5, 0.02), "ylabel": lab,
            })
    return pd.DataFrame(rows)


def test_streaming_matches_aggregate_run():
    raw = _raw()
    acc = {}
    B._process_chunk(raw.copy(), "f.csv", "train", acc)
    out = B._finalize_file(acc, "train").set_index("__ri")

    ref = {}
    for ri, seg in raw.groupby("run_index"):
        f = aggregate_run(seg)
        f["DV"] = float(pd.to_numeric(seg["DV"]).mean())
        ref[ri] = f

    for ri in ref:
        for c in all_feature_columns():
            assert abs(out.loc[ri, c] - ref[ri][c]) <= 1e-9 * (abs(ref[ri][c]) + 1e-9), c
        assert abs(out.loc[ri, "DV_raw"] - ref[ri]["DV"]) < 1e-9
    assert (out["split"] == "train").all()
    assert np.isfinite(out[all_feature_columns()].to_numpy()).all()


def test_zero_guard_all_nan_signal():
    """A run whose signal is entirely NaN must yield 0.0 (not inf/NaN)."""
    raw = _raw(20)
    raw.loc[raw["run_index"] == 0, "torque"] = np.nan
    acc = {}
    B._process_chunk(raw.copy(), "f.csv", "test", acc)
    out = B._finalize_file(acc, "test").set_index("__ri")
    for st in ("mean", "std", "min", "max", "rms"):
        assert out.loc[0, f"torque_{st}"] == 0.0
    assert np.isfinite(out[all_feature_columns()].to_numpy()).all()
