"""Tests for the Paderborn Module-C pipeline (loader parse + features + split).

All tests run on small synthetic arrays / frames — no Paderborn download or
``.mat`` parsing required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load_paderborn import parse_measurement_name
from src.data.build_paderborn_dataset import (
    bearing_label_map,
    extract_paderborn_features,
)
from src.features.vibration_features import time_domain_features
from src.models.train_paderborn import feature_columns, split_artificial_real


def test_parse_measurement_name():
    assert parse_measurement_name("N15_M07_F10_KA01_3.mat") == ("N15_M07_F10", "KA01", 3)
    assert parse_measurement_name("N09_M07_F10_KI04_20.mat") == ("N09_M07_F10", "KI04", 20)


def test_parse_measurement_name_rejects_bad():
    with pytest.raises(ValueError):
        parse_measurement_name("garbage.mat")


def test_bearing_label_map_classes_and_origins():
    bearings = {
        "healthy": ["K001", "K002"],
        "artificial_outer": ["KA01"],
        "artificial_inner": ["KI01"],
        "real_outer": ["KA04"],
        "real_inner": ["KI04"],
    }
    m = bearing_label_map(bearings)
    assert m["K001"] == ("healthy", "healthy")
    assert m["KA01"] == ("outer", "artificial")
    assert m["KI01"] == ("inner", "artificial")
    assert m["KA04"] == ("outer", "real")
    assert m["KI04"] == ("inner", "real")


def test_bearing_label_map_rejects_unknown_group():
    with pytest.raises(KeyError):
        bearing_label_map({"mystery_group": ["X1"]})


def test_extract_features_prefixes_each_channel():
    rng = np.random.default_rng(0)
    channels = {
        "vibration_1": rng.normal(0, 1, 4096),
        "phase_current_1": rng.normal(0, 0.5, 4096),
        "phase_current_2": rng.normal(0, 0.5, 4096),
    }
    row = extract_paderborn_features(channels, list(channels))
    # 3 channels x 10 time-domain features = 30 columns
    assert len(row) == 30
    for prefix in ("vib", "cur1", "cur2"):
        for feat in ("rms", "kurtosis", "crest_factor", "mean"):
            assert f"{prefix}_{feat}" in row
    assert all(np.isfinite(v) for v in row.values())


def test_extract_features_values_match_time_domain():
    sig = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    channels = {"vibration_1": sig}
    row = extract_paderborn_features(channels, ["vibration_1"])
    ref = time_domain_features(sig)
    assert row["vib_rms"] == pytest.approx(ref["rms"])
    assert row["vib_kurtosis"] == pytest.approx(ref["kurtosis"])


def _synth_feature_table() -> pd.DataFrame:
    return pd.DataFrame({
        "condition": ["N15_M07_F10"] * 6,
        "bearing_code": ["K001", "KA01", "KI01", "KA04", "KI04", "K002"],
        "fault_class": ["healthy", "outer", "inner", "outer", "inner", "healthy"],
        "damage_origin": ["healthy", "artificial", "artificial", "real", "real", "healthy"],
        "measurement": [1, 1, 1, 1, 1, 2],
        "vib_rms": [0.1, 0.5, 0.6, 0.55, 0.62, 0.11],
        "cur1_rms": [1.0, 1.2, 1.3, 1.25, 1.31, 1.01],
    })


def test_feature_columns_excludes_labels():
    feats = feature_columns(_synth_feature_table())
    assert set(feats) == {"vib_rms", "cur1_rms"}


def test_split_artificial_real_partitions_by_origin():
    train, real = split_artificial_real(_synth_feature_table())
    # train = healthy + artificial (4 rows); real = real-damage only (2 rows)
    assert set(train["damage_origin"]) == {"healthy", "artificial"}
    assert set(real["damage_origin"]) == {"real"}
    assert len(train) == 4 and len(real) == 2
    # no real-damage bearing leaks into training
    assert not set(train["bearing_code"]) & {"KA04", "KI04"}
