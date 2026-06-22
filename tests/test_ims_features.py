"""Tests for the IMS Module-B pipeline (features + RUL/health labels).

All tests run on small synthetic arrays — no IMS download required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.build_ims_dataset import add_rul_health
from src.data.load_ims import parse_timestamp
from src.features.vibration_features import (
    extract_file_features,
    time_domain_features,
)
from src.models.train_rul import add_sliding_window_features, temporal_split
from src.models.rul_extrapolation import (
    build_health_indicator,
    detect_fpt,
    extrapolate_rul,
)
from src.utils.paths import load_config


def test_kurtosis_rises_with_impulses():
    """A signal with sharp impacts must have far higher kurtosis than noise.

    This is the early-warning property the dynamic-health track relies on.
    """
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(20480)
    impulsive = noise.copy()
    impulsive[::512] += 12.0  # periodic impacts, like a nascent outer-race defect
    k_noise = time_domain_features(noise)["kurtosis"]
    k_impulsive = time_domain_features(impulsive)["kurtosis"]
    assert k_impulsive > k_noise + 1.0


def test_extract_file_features_shapes_and_keys():
    cfg = load_config()
    n = cfg["ims"]["n_bearings"]
    arr = np.random.default_rng(1).standard_normal((2048, n))
    row = extract_file_features(arr, cfg)
    # Target bearing has the full set incl. defect-frequency bands; others RMS only.
    target = cfg["ims"]["target_bearing"]
    assert f"b{target}_kurtosis" in row
    assert f"b{target}_band_BPFO" in row
    for b in range(1, n + 1):
        if b != target:
            assert list(row.keys()).count(f"b{b}_rms") == 1
            assert f"b{b}_kurtosis" not in row


def test_add_rul_health_is_monotonic_and_bounded():
    idx = pd.date_range("2004-02-12 10:32:39", periods=5, freq="10min")
    df = pd.DataFrame({"b1_rms": [1.0, 1.1, 1.3, 1.8, 3.0]}, index=idx)
    out = add_rul_health(df)
    # RUL decreases to exactly 0 at failure (last snapshot).
    assert out["rul_hours"].is_monotonic_decreasing
    assert out["rul_hours"].iloc[-1] == 0.0
    # Health is a 100 -> 0 rescaling and stays in range.
    assert out["health"].iloc[0] == 100.0
    assert out["health"].iloc[-1] == 0.0
    assert out["health"].between(0.0, 100.0).all()


def test_sliding_window_features_mean_and_slope():
    # A strictly linear ramp: rolling slope should equal the step (1.0),
    # rolling mean of the last 3 of [.., k-2, k-1, k] is (k-1).
    df = pd.DataFrame(
        {"b1_rms": np.arange(10.0), "rul_hours": np.arange(10.0)[::-1], "health": 0.0}
    )
    out = add_sliding_window_features(df, window=3)
    # First window-1 rows dropped; labels untouched as feature inputs.
    assert len(out) == 8
    assert "b1_rms_roll_mean" in out and "b1_rms_roll_slope" in out
    assert np.allclose(out["b1_rms_roll_slope"], 1.0)
    assert out["b1_rms_roll_mean"].iloc[0] == 1.0  # mean of [0,1,2]
    # Label columns must not be turned into window features.
    assert "rul_hours_roll_mean" not in out


def test_temporal_split_preserves_order_and_fraction():
    df = pd.DataFrame({"x": range(100)})
    train, test = temporal_split(df, test_frac=0.3)
    assert len(train) == 70 and len(test) == 30
    # No shuffle: test must be the *last* contiguous block.
    assert test["x"].iloc[0] == 70
    assert train["x"].iloc[-1] == 69


def test_health_indicator_maps_baseline_to_100_and_failure_to_0():
    # Flat healthy stretch, then a rising tail toward failure.
    raw = pd.Series([0.1] * 100 + list(np.linspace(0.1, 0.5, 50)))
    hi, health, hi_base, hi_fail = build_health_indicator(
        raw, smooth_window=3, baseline_n=100, fail_percentile=99
    )
    assert abs(hi_base - 0.1) < 1e-6
    assert health.iloc[0] == pytest.approx(100.0)   # healthy baseline
    assert health.iloc[-1] < 5.0                    # near failure


def test_detect_fpt_finds_onset_after_baseline():
    hi = pd.Series([1.0] * 100 + [5.0] * 20)  # clear jump at index 100
    fpt = detect_fpt(hi, baseline_n=100, n_sigma=3.0, consecutive=5)
    assert 100 <= fpt <= 105


def test_extrapolate_rul_converges_on_exponential_degradation():
    # Perfectly exponential HI -> the log-linear fit is exact, so the predicted
    # failure time (HI reaches hi_fail) should match the truth near end of life.
    hours = np.arange(0.0, 120.0)
    hi_base = 0.1
    hi = hi_base + 0.01 * np.exp(0.05 * hours)
    hi_fail = hi[-1]
    rul = extrapolate_rul(hours, hi, hi_base, hi_fail, fpt_idx=10,
                          min_points=5, window=40, max_life=200.0)
    # At hour 110, true RUL = 119 - 110 = 9; prediction should be very close.
    assert abs(rul[110] - (hours[-1] - hours[110])) < 1.0


def test_parse_timestamp_roundtrip():
    ts = parse_timestamp("2004.02.12.10.32.39")
    assert (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second) == (
        2004, 2, 12, 10, 32, 39,
    )
