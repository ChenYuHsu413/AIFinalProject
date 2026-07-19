"""情境 18 Phase 0 前置檢核的單元測試 —— 用合成訊號鎖住判定邏輯。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.s18.phase0_precheck import (
    SIGNAL_COLS,
    dead_zone_window_check,
    reversal_summary,
    sampling_summary,
)


def _run(v: np.ndarray, demand: np.ndarray, fs: float = 50_000.0) -> pd.DataFrame:
    n = len(v)
    return pd.DataFrame({
        "time": np.arange(n) / fs,
        "rod_demand_pos": demand,
        "rod_actual_pos": demand - 0.1,
        "del_pos": np.zeros(n),
        "torque": np.zeros(n),
        "rotor_speed": v,
        "run_index": 0,
        "transitions": 2,
    })


def test_no_dv_or_ylabel_in_signal_cols():
    """防洩漏鐵律 1:特徵管線讀取的欄位不得含標註欄。"""
    assert "DV" not in SIGNAL_COLS and "ylabel" not in SIGNAL_COLS


def test_sampling_summary_recovers_50khz():
    r = _run(np.zeros(1000), np.arange(1000, dtype=float))
    s = sampling_summary(r)
    assert abs(s["fs_hz"] - 50_000.0) < 1e-6
    assert s["dt_is_uniform"] and s["n_samples"] == 1000


def test_unidirectional_run_reports_no_reversal():
    """單向遞增的階梯:速度恆正 -> 零反轉、demand 單調。"""
    v = np.abs(np.sin(np.linspace(0, 6 * np.pi, 2000))) + 0.01
    demand = np.cumsum(np.ones(2000))
    rep = reversal_summary(_run(v, demand))
    assert rep["significant_reversals"] == 0
    assert rep["raw_sign_changes"] == 0
    assert rep["demand_monotonic"] is True


def test_bidirectional_run_counts_reversals():
    """三次真實換向的正弦速度 -> significant_reversals == 3。"""
    v = np.sin(np.linspace(0, 3 * np.pi, 3000)) * 100.0   # + - + -> 兩次換向
    demand = np.cumsum(v) / 1000.0
    rep = reversal_summary(_run(v, demand))
    assert rep["significant_reversals"] == 2
    assert rep["demand_monotonic"] is False


def test_hysteresis_threshold_suppresses_noise_crossings():
    """零附近的量測雜訊會製造大量假過零,遲滯判定必須濾掉。"""
    rng = np.random.default_rng(42)
    v = rng.normal(0, 1e-3, 5000)          # 純雜訊,無真實換向
    rep = reversal_summary(_run(v, np.arange(5000, dtype=float)))
    assert rep["raw_sign_changes"] > 100    # 原始判定被雜訊灌爆
    assert rep["significant_reversals"] <= 1


def test_dead_zone_window_translates_to_ms():
    c = dead_zone_window_check(50_000.0)
    assert abs(c["window_duration_ms"] - 0.4) < 1e-9
