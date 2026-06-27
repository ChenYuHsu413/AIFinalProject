"""train_servo split-aware path: train on split==train, evaluate on held-out test."""
import copy
import json

import numpy as np
import pandas as pd

from src.features.servo_features import feature_set_columns
from src.utils.paths import load_config as _real_load


def _feature_table():
    rng = np.random.default_rng(1)
    cols = feature_set_columns("engineered")
    sev = {"LN": 0.05, "LO": 0.35, "MED": 0.65, "HI": 0.92}
    rows = []
    for split, n in (("train", 8), ("test", 4)):
        for lab, s in sev.items():
            for _ in range(n):
                feat = {c: float(rng.normal(s, 0.06)) for c in cols}
                feat["ylabel"] = lab
                feat["DV"] = float(np.clip(s + rng.normal(0, 0.03), 0, 1))
                feat["split"] = split
                rows.append(feat)
    df = pd.DataFrame(rows)
    df.insert(0, "run_index", np.arange(len(df)))
    return df


def test_train_servo_uses_holdout_split(tmp_path, monkeypatch):
    df = _feature_table()
    feat_path = tmp_path / "feat.parquet"
    df.to_parquet(feat_path)

    cfg = copy.deepcopy(_real_load())
    sv = cfg["servo"]
    sv["processed_features"] = str(feat_path)
    for key, fn in (("clf_model", "clf.joblib"), ("reg_model", "reg.joblib"),
                    ("feature_config", "fc.json"), ("clf_metrics", "clf_eval.json"),
                    ("reg_metrics", "reg_eval.json")):
        sv[key] = str(tmp_path / fn)

    import src.models.train_servo as T
    monkeypatch.setattr(T, "load_config", lambda *a, **k: cfg)
    T.run()

    clf = json.loads((tmp_path / "clf_eval.json").read_text(encoding="utf-8"))
    reg = json.loads((tmp_path / "reg_eval.json").read_text(encoding="utf-8"))
    # evaluated on the held-out test split (4 classes x 4 = 16), not CV on all
    assert clf["eval"] == "holdout_test"
    assert clf["n"] == 16 and reg["n"] == 16
    assert reg["eval"] == "holdout_test"
    assert (tmp_path / "clf.joblib").exists() and (tmp_path / "fc.json").exists()
