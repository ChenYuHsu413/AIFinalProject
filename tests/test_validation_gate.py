"""S3 — validation gate: each rule must actually block a bad candidate."""
from __future__ import annotations

import json
import shutil

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline

from src.models import servo_model_registry as registry
from src.pipeline.validation_gate import BLOCKED, FAIL, PASS, SKIP, run_gate
from src.utils.paths import load_config, resolve

_V1 = registry.version_dir("v1")
pytestmark = pytest.mark.skipif(not _V1.exists(),
                                reason="registry v1 not present (run --migrate)")


def _status(report, name):
    return next(c["status"] for c in report["checks"] if c["name"] == name)


def _detail(report, name):
    return next(c["detail"] for c in report["checks"] if c["name"] == name)


@pytest.fixture
def v1_copy(tmp_path):
    d = tmp_path / "candidate"
    shutil.copytree(_V1, d)
    return d


def test_gate_passes_on_active_copy(v1_copy):
    r = run_gate(v1_copy, active_version="v1")
    assert r["passed"]
    assert _status(r, "completeness") == PASS
    assert _status(r, "smoke_test") == PASS
    assert _status(r, "holdout_metrics") == PASS
    assert _status(r, "ae_monotonicity") == SKIP
    assert (v1_copy / "gate_report.json").exists()


def test_completeness_fails_on_missing_file(v1_copy):
    (v1_copy / "servo_reg.joblib").unlink()
    r = run_gate(v1_copy, active_version="v1")
    assert not r["passed"]
    assert _status(r, "completeness") == FAIL
    # downstream checks can't run -> blocked (counts as not-pass)
    assert _status(r, "holdout_metrics") == BLOCKED


def test_completeness_fails_on_bad_crc(v1_copy):
    m = json.loads((v1_copy / "metrics.json").read_text(encoding="utf-8"))
    m["model_crc32"]["servo_clf.joblib"] = "deadbeef"
    (v1_copy / "metrics.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    r = run_gate(v1_copy, active_version="v1")
    assert not r["passed"] and _status(r, "completeness") == FAIL
    assert "CRC32" in _detail(r, "completeness")


def test_completeness_fails_on_feature_mismatch(v1_copy):
    fc = json.loads((v1_copy / "servo_feature_config.json").read_text(encoding="utf-8"))
    fc["feature_columns"] = fc["feature_columns"][:-1]  # inconsistent with feature_set
    (v1_copy / "servo_feature_config.json").write_text(
        json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    r = run_gate(v1_copy, active_version="v1")
    assert not r["passed"] and _status(r, "completeness") == FAIL


def test_holdout_fails_on_degraded_model(tmp_path):
    """A model that predicts a constant class passes completeness+smoke (valid
    files/schema) but must FAIL the holdout metric gate."""
    d = tmp_path / "candidate"
    d.mkdir()
    fc = json.loads((_V1 / "servo_feature_config.json").read_text(encoding="utf-8"))
    cols = fc["feature_columns"]
    feat = pd.read_parquet(resolve(load_config()["servo"]["processed_features"]))
    tr = feat[feat["split"] == "train"] if "split" in feat.columns else feat
    X, y, dv = tr[cols], tr["ylabel"], tr["DV"]

    clf = Pipeline([("clf", DummyClassifier(strategy="constant", constant="LN"))]).fit(X, y)
    reg = DummyRegressor(strategy="constant", constant=0.0).fit(X, dv)
    joblib.dump({"pipeline": clf}, d / "servo_clf.joblib")
    joblib.dump({"pipeline": reg}, d / "servo_reg.joblib")
    (d / "servo_feature_config.json").write_text(
        json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    metrics = {"version": "candidate", "feature_set": fc["feature_set"],
               "feature_columns": cols, "model_crc32": registry.crc32_for_dir(d),
               "macro_f1": 0.9, "dv_r2": 0.9}  # self-reported (gate ignores; recomputes)
    (d / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    r = run_gate(d, active_version="v1")
    assert _status(r, "completeness") == PASS   # valid files + CRC + feature set
    assert _status(r, "smoke_test") == PASS      # constant model still yields valid schema
    assert _status(r, "holdout_metrics") == FAIL  # constant-LN -> macro-F1 far below active
    assert not r["passed"]
