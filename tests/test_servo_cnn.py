"""Module Servo — Phase B 1D-CNN on raw-signal envelopes (smoke test).

The forward/shape checks run anywhere (synthetic tensors). The full run() check
needs the windowed .npz, which is an offline artefact built from the raw FMCRD zip
(gitignored, absent in CI) — so it skips cleanly when the data isn't present.
Skips entirely if torch is missing (it lives in requirements-dl.txt).
"""
import json

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from src.models import servo_cnn  # noqa: E402
from src.utils.paths import load_config, resolve  # noqa: E402


def test_cnn_and_ae_forward_shapes():
    x = torch.randn(6, 8, 256)
    assert servo_cnn._CNN(8, 4)(x).shape == (6, 4)
    assert servo_cnn._ConvAE(8)(x).shape == x.shape  # AE reconstructs same shape


def test_servo_cnn_run_when_data_present():
    npz = resolve(load_config()["servo"]["windows_path"])
    if not npz.exists():
        pytest.skip("windowed dataset absent (built offline from the raw FMCRD zip)")

    out = json.load(open(servo_cnn.run(), encoding="utf-8"))
    assert out["method"] == "servo_cnn_1d"

    clf = out["classifier"]
    assert 0.0 <= clf["macro_f1"] <= 1.0
    assert len(clf["confusion_matrix"]) == len(clf["labels"])

    rec = out["autoencoder"]["reconstruction_error_by_class"]
    ordered = [rec[lab] for lab in clf["labels"] if lab in rec]
    assert ordered[-1] > ordered[0]  # most-degraded reconstructs worse than healthy
