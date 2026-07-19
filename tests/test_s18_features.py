"""情境 18 特徵 —— 參數化重實作的**逐位元等價測試** + 行為測試。

等價測試是「不重新實作」禁令的機器可驗證替代品:在原函數的參數設定下
(20 samples / 0.05 門檻 / 全速度過零 / 原退化公式),參數化版本必須與
`dsp_analytics` 原函數輸出 bit-identical。任何行為漂移都會讓測試失敗。

真實資料的等價測試涵蓋 3 個完整 LN run(各 299,951 samples),避免在短樣本
上碰巧相等;無法取得 zip 時自動跳過,合成訊號的等價測試仍會執行。
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.s18.dsp_analytics import AdvancedMechanicalDiagnostics as M
from src.s18.s18_features import (
    ANALYSIS_FEATURES,
    FEATURE_COLUMNS,
    commanded_reversal_events,
    compute_run_features,
    dead_zone_sensitivity,
    dead_zone_width_param,
    reversal_error_param,
    stiff_torque_slope,
    viscous_from_complement,
    zero_crossing_events,
)

ZIP = os.environ.get("FMCRD_ZIP", "C:/Users/alung/Downloads/FMCRD_Data.zip")
_CACHE = Path(__file__).parent / "_s18_ln_runs.pkl"

PARAMS = {
    "dead_zone": {"search_samples": 1000, "move_threshold": 0.05},
    "friction": {"viscous_speed_min": 100.0},
}


# ------------------------------------------------------------------ fixtures


def _synth(n=20_000, seed=0):
    """合成訊號:速度多次過零、位置有階梯與反轉。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    v = np.sin(t) * 120.0 + rng.normal(0, 2.0, n)
    d = np.cumsum(v) / 5_000.0 + 200.0
    a = d - 0.4 * np.sin(t) + rng.normal(0, 0.02, n)
    return d, a, v


@pytest.fixture(scope="module")
def ln_runs():
    """3 個完整 LN run;讀一次後快取,zip 不存在則跳過。"""
    if _CACHE.exists():
        return pickle.loads(_CACHE.read_bytes())
    if not Path(ZIP).exists():
        pytest.skip(f"FMCRD zip 不存在:{ZIP}")
    from src.s18.phase0_precheck import read_first_runs
    runs = read_first_runs(ZIP, n_runs=3)
    _CACHE.write_bytes(pickle.dumps(runs))
    return runs


# ------------------------------------------------- 逐位元等價(合成訊號)


def test_dead_zone_bit_identical_synthetic():
    d, a, v = _synth()
    ours = dead_zone_width_param(d, a, zero_crossing_events(v),
                                 search_samples=20, move_threshold=0.05,
                                 fallback=0.05)
    assert ours == M.dead_zone_width(d, a, v)


def test_reversal_error_bit_identical_synthetic():
    d, a, v = _synth()
    ours = reversal_error_param(d, a, zero_crossing_events(v),
                                fallback_on_empty=True)
    assert ours == M.reversal_error(d, a, v)


def test_zero_crossing_events_match_original_loop():
    """事件索引本身也必須與原函數的 for-loop 判定完全一致。"""
    _, _, v = _synth(5_000)
    ref = [i for i in range(1, len(v)) if v[i] * v[i - 1] < 0]
    assert np.array_equal(zero_crossing_events(v), np.array(ref))


def test_empty_events_fallback_reproduces_original():
    """無事件時,開啟 fallback 必須重現原函數的退化公式。"""
    n = 1000
    d = np.linspace(200, 220, n)
    a = d - 0.3
    v = np.ones(n) * 50.0                      # 恆正,無過零
    assert dead_zone_width_param(d, a, [], 20, 0.05, fallback=0.05) == \
        M.dead_zone_width(d, a, v)
    assert reversal_error_param(d, a, [], fallback_on_empty=True) == \
        M.reversal_error(d, a, v)


# --------------------------------------------- 逐位元等價(真實 LN run,全長)


def test_bit_identical_on_three_full_ln_runs(ln_runs):
    assert len(ln_runs) == 3
    for ri, run in sorted(ln_runs.items()):
        d = run["rod_demand_pos"].to_numpy()
        a = run["rod_actual_pos"].to_numpy()
        v = run["rotor_speed"].to_numpy()
        assert len(d) > 250_000, f"run {ri} 太短,等價測試失去意義"
        ev = zero_crossing_events(v)

        assert dead_zone_width_param(d, a, ev, 20, 0.05, fallback=0.05) == \
            M.dead_zone_width(d, a, v), f"dead_zone 在 run {ri} 不等價"
        assert reversal_error_param(d, a, ev, fallback_on_empty=True) == \
            M.reversal_error(d, a, v), f"reversal_error 在 run {ri} 不等價"


# ------------------------------------------------------------ 零事件 policy


def test_zero_events_yield_nan_not_default_constant():
    """本專案用法:無事件 -> NaN,不得落回 0.05 / mean|FE|*0.1。"""
    d = np.linspace(200, 220, 1000)
    a = d - 0.3
    assert np.isnan(dead_zone_width_param(d, a, [], 1000, 0.05))
    assert np.isnan(reversal_error_param(d, a, []))


def test_run_without_commanded_reversal_gets_nan(ln_runs):
    """單向指令的合成 run:_cmd 版本應為 NaN,_zc 版本仍可算。"""
    run = next(iter(ln_runs.values())).copy()
    n = len(run)
    # 覆寫成單向遞增的四階梯(無換向),速度保留原樣以確保 zc 事件存在
    steps = np.repeat(np.array([0.0, 5.0, 10.0, 15.0, 20.0]), n // 5 + 1)[:n]
    run["rod_demand_pos"] = 200.0 + steps
    feats = compute_run_features(run, PARAMS)
    assert feats["n_cmd_reversals"] == 0
    assert np.isnan(feats["BL_DeadZone_cmd"])
    assert np.isnan(feats["BL_ReversalErr_cmd"])
    assert feats["n_zero_crossings"] > 0


# ------------------------------------------------------------- 新增/改名特徵


def test_stiff_torque_slope_uses_torque():
    """STIFF_TorqueSlope 必須對 torque 敏感(原 dsp 函數完全沒用到 torque)。"""
    rng = np.random.default_rng(42)
    fe = rng.normal(0, 1, 5_000)
    assert stiff_torque_slope(fe, 3.0 * fe) == pytest.approx(3.0, abs=1e-9)
    assert stiff_torque_slope(fe, 7.0 * fe) == pytest.approx(7.0, abs=1e-9)
    # 對照:原函數換了 torque 也不動如山
    pos = rng.normal(0, 1, 5_000)
    assert M.force_displacement_slope(pos, fe, np.abs(3.0 * fe)) == \
        M.force_displacement_slope(pos, fe, np.abs(7.0 * fe))


def test_viscous_complement_recovers_slope():
    v = np.linspace(-200, 200, 4_000)
    tq = 0.02 * np.abs(v) + 1.0
    assert viscous_from_complement(v, tq, 100.0) == pytest.approx(0.02, abs=1e-9)


def test_viscous_nan_when_high_speed_band_empty():
    v = np.linspace(-50, 50, 1_000)          # 全部 < 100
    assert np.isnan(viscous_from_complement(v, np.ones(1_000), 100.0))


def test_commanded_reversal_events_counts_sign_flips():
    n = 5_000
    # 四階梯 + + - :一次換向
    d = np.concatenate([np.full(n, 200.0), np.full(n, 210.0),
                        np.full(n, 220.0), np.full(n, 205.0)])
    assert len(commanded_reversal_events(d)) == 1
    # 單向遞增:零換向
    up = np.concatenate([np.full(n, 200.0 + 10 * k) for k in range(4)])
    assert len(commanded_reversal_events(up)) == 0


# ------------------------------------------------------------------ 真實資料


def test_analysis_features_non_constant_across_ln_runs(ln_runs):
    """Phase 1 驗收:納入分析的特徵在 3 個 LN run 間不得退化成常數。"""
    rows = [compute_run_features(r, PARAMS) for _, r in sorted(ln_runs.items())]
    df = pd.DataFrame(rows)
    degenerate = [c for c in ANALYSIS_FEATURES
                  if df[c].notna().all() and float(df[c].std()) == 0.0]
    assert not degenerate, f"以下特徵退化成常數:{degenerate}"


def test_demand_is_piecewise_constant(ln_runs):
    """鎖住 BL_DeadZone 標「不適用」的結構性前提:指令是階梯,不是連續軌跡。"""
    for ri, run in sorted(ln_runs.items()):
        d = run["rod_demand_pos"].to_numpy()
        n_moves = int(np.sum(np.abs(np.diff(d)) > 1e-12))
        assert n_moves == 4, f"run {ri} 指令跳變數 {n_moves},預期 4"


def test_dead_zone_is_structurally_zero_regardless_of_window(ln_runs):
    """窗長 10/20/40 ms 全部給同一結果 -> 證明這不是調參問題而是結構性不適用。

    死區量的是「反轉後**指令**繼續移動」的位移;階梯指令在死區期間位移恆為 0,
    故 |d[j]-d[i]| 結構上必為 0(或搜不到移動而為 NaN),與窗長無關。
    """
    for ri, run in sorted(ln_runs.items()):
        s = dead_zone_sensitivity(run, [10.0, 20.0, 40.0], 50_000.0)
        vals = np.array(list(s.values()))
        finite = vals[np.isfinite(vals)]
        assert np.all(finite == 0.0), f"run {ri} 死區非零,不適用之判定需重審:{s}"
