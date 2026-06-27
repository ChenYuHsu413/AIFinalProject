"""Live inference for Module C (Paderborn fault classification).

Loads the reference classifier bundle persisted by ``train_paderborn`` and turns
one feature row into a structured prediction (class + per-class probabilities).
The model was trained on healthy + ARTIFICIAL faults, so predicting on a REAL
accelerated-lifetime measurement shows the artificial->real domain gap live.

This module loads the bundle lazily and caches it (the cloud runtime never
trains; it only loads the committed ``paderborn_clf.joblib``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.utils.paths import load_config, resolve

_BUNDLE: Optional[Dict[str, Any]] = None


def load_paderborn_model(force: bool = False) -> Dict[str, Any]:
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE
    p = resolve(load_config()["paderborn"]["best_model"])
    if not p.exists():
        raise FileNotFoundError(
            "找不到 Paderborn 參考模型。請先執行：\n"
            "  python -m src.data.build_paderborn_dataset\n"
            "  python -m src.models.train_paderborn"
        )
    _BUNDLE = joblib.load(p)
    return _BUNDLE


def predict_paderborn(features: Dict[str, Any]) -> Dict[str, Any]:
    """Structured prediction for one Paderborn feature row."""
    b = load_paderborn_model()
    cols = list(b["feature_columns"])
    missing = [c for c in cols if c not in features]
    if missing:
        raise ValueError(f"缺少必要特徵欄位：{missing}")

    X = pd.DataFrame([{c: float(features[c]) for c in cols}])
    pipe = b["pipeline"]
    predicted = str(pipe.predict(X)[0])

    out: Dict[str, Any] = {"predicted_class": predicted, "labels": list(b["labels"])}
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
        classes = list(clf.classes_)
        out["proba"] = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
        out["confidence"] = round(float(max(proba)), 4)
    return out
