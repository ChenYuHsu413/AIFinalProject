"""Module Servo — offline PyTorch DL baseline (smoke test).

Runs the offline DL (tiny CPU nets on the committed engineered features) and
checks the results JSON shape + the degradation signal (autoencoder
reconstruction error rises from healthy to most-degraded). Skips cleanly if
torch is not installed (it lives in requirements-dev.txt only).
"""
import json

import pytest

pytest.importorskip("torch")


def test_servo_dl_runs_and_writes_expected_json():
    from src.models import servo_dl

    out = json.load(open(servo_dl.run(), encoding="utf-8"))

    assert out["method"] == "servo_dl_torch"
    assert out["framework"].startswith("pytorch")
    assert 0.0 <= out["mlp_classification_macro_f1"] <= 1.0
    assert "r2" in out["mlp_regression"] and "mae" in out["mlp_regression"]
    assert out["mlp_regression"]["mae"] >= 0

    rec = out["reconstruction_error_by_class"]
    labels = out["labels"]
    assert set(rec) == set(labels)

    # Reconstruction error should be (weakly) monotonic from healthy -> degraded,
    # and the most-degraded class clearly above the healthy one.
    ordered = [rec[lab] for lab in labels]  # labels are HEALTH_LABELS order (LN..HI)
    assert all(b >= a - 1e-6 for a, b in zip(ordered, ordered[1:]))
    assert ordered[-1] > ordered[0]
