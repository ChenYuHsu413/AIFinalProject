"""Tests for the E1 cross-condition domain-adaptation transforms.

All tests run on small synthetic arrays — no XJTU download required.  They check
the two correctness-critical properties the plan calls out:
  * lifetime-ratio restoration multiplies fraction by life exactly;
  * the transductive z-score never uses the target labels and aligns each
    condition to zero-mean / unit-variance;
  * CORAL makes the aligned source covariance match the target's.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.models.eval_xjtu_domain_adapt import (
    _bearing_life,
    coral_align,
    per_condition_zscore,
)


def test_bearing_life_is_max_rul_per_bearing():
    df = pd.DataFrame({
        "bearing": ["A", "A", "A", "B", "B"],
        "rul_hours": [2.0, 1.0, 0.0, 40.0, 20.0],
    })
    life = _bearing_life(df).to_numpy()
    assert list(life) == [2.0, 2.0, 2.0, 40.0, 40.0]


def test_lifetime_ratio_restoration_is_exact():
    # fraction * life must restore the original hours exactly
    life = np.array([2.0, 2.0, 40.0])
    frac = np.array([1.0, 0.5, 0.25])
    restored = frac * life
    np.testing.assert_allclose(restored, [2.0, 1.0, 10.0])


def test_transductive_zscore_takes_no_target_labels():
    # the signature must not expose any y / label / rul parameter
    params = set(inspect.signature(per_condition_zscore).parameters)
    assert params == {"X_train", "cond_train", "X_test"}
    assert not any("y" in p or "rul" in p or "label" in p for p in params)


def test_transductive_zscore_centers_each_condition():
    rng = np.random.default_rng(0)
    # two source conditions with very different offset + scale
    Xa = rng.normal(loc=100.0, scale=5.0, size=(50, 3))
    Xb = rng.normal(loc=-20.0, scale=0.5, size=(50, 3))
    X_train = np.vstack([Xa, Xb])
    cond = np.array(["a"] * 50 + ["b"] * 50)
    X_test = rng.normal(loc=3000.0, scale=50.0, size=(40, 3))

    Xtr, Xte = per_condition_zscore(X_train, cond, X_test)
    # each source condition -> ~zero mean, ~unit std
    np.testing.assert_allclose(Xtr[:50].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(Xtr[50:].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(Xtr[:50].std(axis=0), 1.0, atol=1e-3)
    # target standardised by its OWN stats -> ~zero mean
    np.testing.assert_allclose(Xte.mean(axis=0), 0.0, atol=1e-6)


def test_coral_aligns_source_covariance_to_target():
    rng = np.random.default_rng(1)
    # source and target with different covariance structure
    As = rng.normal(size=(400, 3)) @ np.array([[2.0, 0.3, 0.0],
                                               [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 0.5]])
    At = rng.normal(size=(400, 3)) @ np.array([[0.4, 0.0, 0.0],
                                               [0.6, 1.5, 0.0],
                                               [0.0, 0.2, 3.0]])
    cond = np.array(["x"] * 200 + ["y"] * 200)
    Xs_aligned, Xt_out = coral_align(As, cond, At, reg=1e-6)
    # aligned-source covariance should closely match the target covariance
    cov_aligned = np.cov(Xs_aligned, rowvar=False)
    cov_target = np.cov(At, rowvar=False)
    assert np.linalg.norm(cov_aligned - cov_target) < 0.15 * np.linalg.norm(cov_target)
    # the target features are passed through unchanged
    np.testing.assert_allclose(Xt_out, At)


def test_coral_returns_finite_values():
    rng = np.random.default_rng(2)
    Xs = rng.normal(size=(100, 4))
    Xt = rng.normal(size=(80, 4))
    Xs_a, Xt_o = coral_align(Xs, np.array(["c"] * 100), Xt, reg=1.0)
    assert np.isfinite(Xs_a).all()
