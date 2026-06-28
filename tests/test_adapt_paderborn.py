"""Tests for CE1 Paderborn domain adaptation (adapt_paderborn).

All synthetic — no Paderborn download / ``.mat`` parsing.  These guard the two
properties that make the ablation honest: (1) the label-free transforms touch
target features only, and (2) few-shot never tests on a row it trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.eval_xjtu_domain_adapt import coral_align, per_condition_zscore
from src.models.adapt_paderborn import (
    _baseline_importances,
    _binary_f1,
    _feature_diagnosis,
    _few_shot_curve,
    _macro_f1,
    _make_pipe,
    _spearman,
)


def _cov_dist(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius distance between two feature covariance matrices."""
    return float(np.linalg.norm(np.cov(A, rowvar=False) - np.cov(B, rowvar=False)))


def test_coral_reduces_covariance_gap():
    """CORAL should move the source covariance toward the target's."""
    rng = np.random.RandomState(0)
    src = rng.normal(0, 1, size=(200, 4))
    # target = a different covariance (scaled + rotated)
    tgt = rng.normal(0, 1, size=(200, 4)) @ np.array(
        [[2.0, 0.5, 0, 0], [0.5, 1.5, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 1.0]]
    )
    before = _cov_dist(src, tgt)
    src_a, tgt_out = coral_align(src, np.zeros(len(src)), tgt, reg=1e-3)
    after = _cov_dist(src_a, tgt)
    assert after < before
    # target must be returned unchanged (label-free, target-X only)
    assert np.allclose(tgt_out, tgt)


def test_transforms_never_see_target_labels():
    """coral_align / per_condition_zscore take X only — no label argument exists.

    A structural guard: the transform signature is (X_train, cond_train, X_test);
    there is no slot for target y, so target labels cannot leak by construction.
    """
    rng = np.random.RandomState(1)
    Xs = rng.normal(size=(50, 3))
    Xt = rng.normal(size=(30, 3))
    cond = np.zeros(len(Xs))
    for fn in (coral_align, per_condition_zscore):
        Xa, Xta = fn(Xs, cond, Xt)
        assert Xa.shape == Xs.shape
        assert Xta.shape == Xt.shape


def _synth_paderborn(rng: np.random.RandomState):
    """Tiny synthetic source (healthy+artificial) / real frames with 3 features."""
    feats = ["f0", "f1", "f2"]

    def block(label, origin, n, loc):
        d = {f: rng.normal(loc, 1.0, size=n) for f in feats}
        d["fault_class"] = label
        d["damage_origin"] = origin
        d["condition"] = "N15_M07_F10"
        return pd.DataFrame(d)

    train = pd.concat([
        block("healthy", "healthy", 30, 0.0),
        block("outer", "artificial", 30, 3.0),
        block("inner", "artificial", 30, -3.0),
    ], ignore_index=True)
    real = pd.concat([
        block("outer", "real", 25, 2.0),
        block("inner", "real", 25, -2.0),
    ], ignore_index=True)
    return train, real, feats


def test_few_shot_no_leakage_and_monotone_size():
    """Few-shot draws k real rows into training and tests on the REST.

    Verifies the test count shrinks by exactly k*2 (two damaged classes) and the
    curve is produced for the requested ks.
    """
    rng = np.random.RandomState(2)
    train, real, feats = _synth_paderborn(rng)
    ks = [1, 3, 5]
    curve = _few_shot_curve(train, real, feats, "random_forest", ks, seeds=3, rs=42)
    assert [pt["k_per_class"] for pt in curve] == ks
    for pt in curve:
        # 50 real rows total, k removed per class (2 classes) -> 50 - 2k tested
        assert pt["n_test_mean"] == 50 - 2 * pt["k_per_class"]
        assert 0.0 <= pt["macro_f1_mean"] <= 1.0
        assert pt["macro_f1_std"] >= 0.0


def test_few_shot_caps_k_to_leave_test_rows():
    """k larger than (class size - 1) is dropped, never emptying the test set."""
    rng = np.random.RandomState(3)
    train, real, feats = _synth_paderborn(rng)  # 25 real per class
    curve = _few_shot_curve(train, real, feats, "random_forest", [10, 25, 100], seeds=2, rs=42)
    ks = [pt["k_per_class"] for pt in curve]
    assert 10 in ks and 25 not in ks and 100 not in ks  # 25 would leave 0 test rows


def test_metrics_helpers_label_handling():
    """binary_f1 ignores the absent healthy class; macro_f1 includes it (0-support)."""
    y_true = ["outer", "inner", "outer", "inner"]
    y_pred = ["outer", "inner", "healthy", "inner"]  # one leaks to healthy
    assert 0.0 < _binary_f1(y_true, y_pred) <= 1.0
    # 3-class macro averages over healthy(0), outer, inner -> strictly below binary
    assert _macro_f1(y_true, y_pred) < _binary_f1(y_true, y_pred)


# --- B: CORAL + few-shot combo --------------------------------------------
def test_few_shot_coral_combo_same_shape():
    """align=True returns a curve of the same shape as plain few-shot (no leakage)."""
    rng = np.random.RandomState(4)
    train, real, feats = _synth_paderborn(rng)
    ks = [1, 5]
    plain = _few_shot_curve(train, real, feats, "random_forest", ks, seeds=2, rs=42)
    combo = _few_shot_curve(train, real, feats, "random_forest", ks, seeds=2, rs=42, align=True)
    assert [p["k_per_class"] for p in combo] == ks
    for pp, pc in zip(plain, combo):
        assert pc["n_test_mean"] == pp["n_test_mean"]  # same leakage-free test size
        assert 0.0 <= pc["macro_f1_mean"] <= 1.0


# --- A: feature-transferability diagnosis ----------------------------------
def test_spearman_matches_known_sign():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _spearman(a, a) == pytest.approx(1.0)
    assert _spearman(a, a[::-1]) == pytest.approx(-1.0)
    assert _spearman(a, np.ones(4)) == 0.0  # degenerate -> 0, no NaN


def test_feature_diagnosis_structure_and_importance_source():
    """Diagnosis returns per-feature importance+shift, a bounded correlation, and
    falls back to mutual-info when the classifier exposes no importances."""
    rng = np.random.RandomState(5)
    train, real, feats = _synth_paderborn(rng)
    pipe = _make_pipe("random_forest", feats, 42).fit(train[feats], train["fault_class"])
    diag = _feature_diagnosis(train, real, feats, pipe)
    assert {p["feature"] for p in diag["per_feature"]} == set(feats)
    assert all("importance" in p and "shift" in p for p in diag["per_feature"])
    assert -1.0 <= diag["spearman_importance_vs_shift"] <= 1.0
    assert len(diag["top_discriminative"]) == min(5, len(feats))
    # RF exposes feature_importances_ -> array length matches feats
    imp = _baseline_importances(pipe, feats, train)
    assert len(imp) == len(feats)
