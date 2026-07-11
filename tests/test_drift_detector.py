"""S4 — drift detector structural guarantees (degradation ≠ drift).

(a) HI degradation does NOT false-trigger drift (in-distribution).
(b) An injected sensor-gain shift DOES trigger drift on v1's baseline (off-manifold).
(c) After retraining folds the drifted condition in, v2's baseline DIGESTS it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.monitor import drift_detector as dd
from src.models import servo_model_registry as registry
from src.utils.paths import load_config, resolve

_V1 = registry.version_dir("v1")
pytestmark = pytest.mark.skipif(
    not (_V1 / dd.DRIFT_BASELINE_FILE).exists(),
    reason="v1 drift baseline not built (run drift_detector.build_for_version)")


def _feat_df():
    df = pd.read_parquet(resolve(load_config()["servo"]["processed_features"]))
    return df[df["split"] == "train"], df[df["split"] == "test"]


def _windows(frame):
    """Turn feature rows into window dicts the detector consumes."""
    return [{"_features": dict(r), "_stream_t": float(i * 3), "_true": r["ylabel"]}
            for i, (_, r) in enumerate(frame.iterrows())]


def _run(detector, windows):
    events = []
    for w in windows:
        events += detector.update(w)
    return [e["type"] for e in events]


_CFG = {"sustain_windows": 3, "psi_window": 8, "psi_threshold": 0.2, "psi_min_features": 3}


def test_hi_degradation_does_not_false_trigger(tmp_path):
    _, te = _feat_df()
    det = dd.DriftDetector(_V1, config=_CFG, alerts_dir=tmp_path)
    hi = te[te["ylabel"] == "HI"].head(20)
    assert "drift_detected" not in _run(det, _windows(hi))
    assert not det.active


def test_injected_sensor_drift_triggers_on_v1(tmp_path):
    _, te = _feat_df()
    det = dd.DriftDetector(_V1, config=_CFG, alerts_dir=tmp_path)
    drifted = dd.inject_sensor_drift_features(te[te["ylabel"] == "HI"].head(20), gain=1.3)
    assert "drift_detected" in _run(det, _windows(drifted))
    assert det.active


def test_v2_digests_the_injected_condition(tmp_path):
    tr, te = _feat_df()
    # v2 training = normal train + the drifted operating condition folded in.
    tr_v2 = pd.concat([tr, dd.inject_sensor_drift_features(tr, gain=1.3)], ignore_index=True)
    v2_dir = tmp_path / "v2"
    v2_dir.mkdir()
    scaler, pca, baseline = dd.build_drift_baseline(tr_v2, "v2")
    dd.save_drift_baseline(v2_dir, scaler, pca, baseline)

    det = dd.DriftDetector(v2_dir, config=_CFG, alerts_dir=tmp_path)
    drifted = dd.inject_sensor_drift_features(te[te["ylabel"] == "HI"].head(20), gain=1.3)
    assert "drift_detected" not in _run(det, _windows(drifted))  # digested -> no drift
    assert not det.active
