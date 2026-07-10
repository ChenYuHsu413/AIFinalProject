"""S2 — shared replay window pipeline (server-independent, offline)."""
from __future__ import annotations

from statistics import mean

import pytest

from src.utils.paths import resolve, load_config


def _manifest_present() -> bool:
    return resolve(load_config()["servo_replay"]["manifest"]).exists()


@pytest.mark.skipif(not _manifest_present(), reason="replay material not extracted")
def test_window_predictions_progress_ln_to_hi():
    from src.monitor.servo_replay_client import iter_window_predictions_from_rows
    from src.monitor.servo_replay_stream import iter_replay_rows

    preds = list(iter_window_predictions_from_rows(iter_replay_rows(), 6.0, 3.0))
    assert len(preds) >= 6

    dv = {}
    for p in preds:
        dv.setdefault(p["_true"], []).append(p["degradation_score"])
    # The degradation score must rise monotonically across the journey segments.
    assert mean(dv["LN"]) < mean(dv["LO"]) < mean(dv["HI"])
    # Every window carries the bookkeeping fields the alert engine/UI rely on.
    assert all({"_true", "_rows", "_stream_t", "risk_level"} <= p.keys() for p in preds)
