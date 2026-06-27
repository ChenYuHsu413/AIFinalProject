"""Module C — live inference on one Paderborn measurement."""
import pytest

from app.backend import services
from src.models.predict_paderborn import load_paderborn_model


def _model_available() -> bool:
    try:
        load_paderborn_model(force=True)
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _model_available(), reason="paderborn_clf.joblib not trained"
)


def test_samples_have_features_and_labels():
    rows = services.paderborn_samples()
    assert rows, "expected representative samples"
    r = rows[0]
    assert {"bearing_code", "fault_class", "damage_origin", "features"} <= set(r)
    assert len(r["features"]) == 30  # 10 time-domain stats x 3 channels


def test_predict_returns_class_and_proba():
    rows = services.paderborn_samples()
    out = services.paderborn_predict_one(rows[0]["features"])
    assert out["predicted_class"] in out["labels"]
    assert abs(sum(out["proba"].values()) - 1.0) < 0.01  # 4-dp rounded probs
    assert 0.0 <= out["confidence"] <= 1.0


def test_missing_feature_raises():
    rows = services.paderborn_samples()
    feats = dict(rows[0]["features"])
    feats.pop(next(iter(feats)))
    with pytest.raises(ValueError, match="缺少必要特徵欄位"):
        services.paderborn_predict_one(feats)
