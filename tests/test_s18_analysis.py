"""情境 18 分析工具測試 —— 用已知答案的合成資料鎖住統計行為。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.s18.s18_analysis import (
    auc,
    auc_with_ci,
    composite_score,
    ln_baseline,
    monotonicity,
    partial_spearman,
    spearman_permutation,
    trigger_rates,
    zscores,
)

FEATS = ["f1", "f2", "f3", "f4"]


def _labelled(n_per=50, seed=42):
    """LN<LO<MED<HI 的合成資料:f1 單調上升,f2 無關。"""
    rng = np.random.default_rng(seed)
    rows = []
    for k, lab in enumerate(["LN", "LO", "MED", "HI"]):
        for _ in range(n_per):
            rows.append({"ylabel": lab,
                         "f1": k + rng.normal(0, 0.3),
                         "f2": rng.normal(0, 1),
                         "f3": -k + rng.normal(0, 0.3),
                         "f4": k + rng.normal(0, 0.3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- AUC


def test_auc_perfect_separation_is_one():
    assert auc(np.arange(10, 20), np.arange(0, 10)) == 1.0


def test_auc_identical_distributions_is_half():
    x = np.arange(50, dtype=float)
    assert auc(x, x.copy()) == pytest.approx(0.5, abs=1e-9)


def test_auc_ci_brackets_point_estimate_and_reports_n():
    rng = np.random.default_rng(0)
    r = auc_with_ci(rng.normal(1, 1, 80), rng.normal(0, 1, 40), n_boot=500)
    assert r["ci_low"] <= r["auc"] <= r["ci_high"]
    assert r["n_pos"] == 80 and r["n_neg"] == 40


def test_auc_ci_ignores_nan_rows():
    a = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    r = auc_with_ci(a, np.array([0.0, 0.5]), n_boot=200)
    assert r["n_pos"] == 4


# --------------------------------------------------------- 單調性 / 偏相關


def test_monotonicity_detects_increasing_feature():
    r = monotonicity(_labelled(), "f1", n_perm=500)
    assert r["rho"] > 0.8 and r["p_perm"] < 0.01 and r["direction"] == "up"


def test_monotonicity_detects_decreasing_feature():
    r = monotonicity(_labelled(), "f3", n_perm=500)
    assert r["rho"] < -0.8 and r["direction"] == "down"


def test_monotonicity_null_feature_not_significant():
    r = monotonicity(_labelled(), "f2", n_perm=500)
    assert r["p_perm"] > 0.05


def test_monotonicity_reports_coverage_with_nan():
    df = _labelled()
    df.loc[df.index[:40], "f1"] = np.nan
    r = monotonicity(df, "f1", n_perm=200)
    assert r["coverage"] == pytest.approx(1 - 40 / len(df))
    assert r["n"] == len(df) - 40


def test_permutation_p_is_bounded_away_from_zero():
    """permutation p 下界為 1/(n_perm+1),不得回傳 0。"""
    r = spearman_permutation(np.arange(100), np.arange(100), n_perm=200)
    assert r["p_perm"] == pytest.approx(1 / 201)


def test_partial_correlation_kills_covariate_driven_association():
    """特徵與標的的關聯完全由共變量驅動時,偏相關應塌陷。"""
    rng = np.random.default_rng(1)
    cov = rng.normal(0, 1, 400)
    df = pd.DataFrame({"cov": cov,
                       "feat": cov + rng.normal(0, 0.05, 400),
                       "target": cov + rng.normal(0, 0.05, 400)})
    raw = spearman_permutation(df["feat"], df["target"], n_perm=300)
    par = partial_spearman(df, "feat", "target", "cov", n_perm=300)
    assert raw["rho"] > 0.9
    assert abs(par["rho_partial"]) < 0.3


def test_partial_correlation_preserves_genuine_association():
    rng = np.random.default_rng(2)
    cov = rng.normal(0, 1, 400)
    feat = rng.normal(0, 1, 400)
    df = pd.DataFrame({"cov": cov, "feat": feat,
                       "target": feat + 0.2 * cov + rng.normal(0, 0.1, 400)})
    par = partial_spearman(df, "feat", "target", "cov", n_perm=300)
    assert par["rho_partial"] > 0.8


# ----------------------------------------------------- 基線 / z / 複合分數


def test_baseline_uses_only_ln_rows():
    df = _labelled()
    b = ln_baseline(df, FEATS)
    ln = df[df.ylabel == "LN"]
    assert b.loc["f1", "mean"] == pytest.approx(ln["f1"].mean())
    assert b.loc["f1", "n"] == len(ln)


def test_baseline_raises_without_ln():
    with pytest.raises(ValueError):
        ln_baseline(_labelled().query("ylabel != 'LN'"), FEATS)


def test_zscore_of_baseline_mean_is_zero():
    df = _labelled()
    b = ln_baseline(df, FEATS)
    z = zscores(df, b, FEATS)
    assert z.loc[df.ylabel == "LN", "f1"].mean() == pytest.approx(0, abs=1e-9)


def test_composite_uses_top3_of_available_only():
    z = pd.DataFrame([{"a": 5.0, "b": 4.0, "c": 3.0, "d": 0.0}])
    assert composite_score(z)[0] == pytest.approx(4.0)      # (5+4+3)/3


def test_composite_ignores_nan_and_adjusts_denominator():
    """NaN 特徵不入 top-3 選擇,分母為實際可用特徵數。"""
    z = pd.DataFrame([{"a": 6.0, "b": np.nan, "c": 3.0, "d": 3.0}])
    assert composite_score(z)[0] == pytest.approx(4.0)      # (6+3+3)/3


def test_composite_nan_when_too_few_available():
    z = pd.DataFrame([{"a": 6.0, "b": np.nan, "c": np.nan, "d": np.nan}])
    assert np.isnan(composite_score(z, min_available=3)[0])


def test_trigger_rates_counts_and_order():
    s = pd.Series([0.0, 0.0, 9.0, 9.0])
    y = pd.Series(["LN", "LN", "HI", "HI"])
    t = trigger_rates(s, y, threshold=1.0)
    assert list(t.index) == ["LN", "HI"]
    assert t.loc["LN", "trigger_rate"] == 0.0
    assert t.loc["HI", "trigger_rate"] == 1.0
    assert t.loc["HI", "n"] == 2
