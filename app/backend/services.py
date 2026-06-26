"""Thin service layer between the FastAPI handlers and the inference code.

Kept deliberately small so unit tests can target it directly.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.predict import (
    REQUIRED_INPUT_COLUMNS,
    load_failure_type_models,
    load_model,
    predict_full,
    predict_records,
    predict_single,
)
from src.utils.paths import load_config, resolve


def health() -> Dict[str, Any]:
    try:
        load_model()
        return {"status": "ok", "model_loaded": True, "message": None}
    except FileNotFoundError as e:
        return {"status": "model_missing", "model_loaded": False, "message": str(e)}


def model_info() -> Dict[str, Any]:
    bundle = load_model()
    return {
        "model_name": bundle.model_name,
        "feature_set": bundle.feature_set,
        "feature_columns": bundle.feature_columns,
        "metrics": bundle.metrics,
    }


def predict_one(record: Dict[str, Any]) -> Dict[str, Any]:
    return predict_single(record)


def predict_one_full(record: Dict[str, Any]) -> Dict[str, Any]:
    return predict_full(record)


def failure_type_metrics() -> List[Dict[str, Any]]:
    cfg = load_config()
    path = resolve(cfg["paths"]["failure_type_metrics"])
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def predict_batch_from_csv(content: bytes) -> List[Dict[str, Any]]:
    df = pd.read_csv(io.BytesIO(content))
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"上傳的 CSV 缺少欄位：{missing}。"
            f"必要欄位：{REQUIRED_INPUT_COLUMNS}"
        )
    return predict_records(df[REQUIRED_INPUT_COLUMNS])


def comparison_metrics() -> List[Dict[str, Any]]:
    cfg = load_config()
    path = resolve(cfg["paths"]["metrics_csv"])
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _read_json_or_empty(rel_path: str | Path) -> Dict[str, Any]:
    """Read a small project-relative JSON; empty dict if the file is missing."""
    path = resolve(rel_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# Per-module KPI metric sources: module key -> {response key: (config section, path key)}.
# Module A's KPI strip is already served by /metrics + /model_info, so it is not listed here.
_METRICS_SUMMARY_SOURCES: Dict[str, Dict[str, tuple]] = {
    "servo": {"clf": ("servo", "clf_metrics"), "reg": ("servo", "reg_metrics")},
    "B": {"ims_rul": ("ims", "rul_metrics")},
    "Bplus": {
        "generalization": ("xjtu", "gen_metrics"),
        "lobo": ("xjtu", "lobo_metrics"),
        "loco": ("xjtu", "loco_metrics"),
    },
    "C": {"paderborn": ("paderborn", "metrics")},
}


def metrics_summary(module: str) -> Dict[str, Any]:
    """Return the raw KPI metric JSON(s) for a module's overview strip.

    Missing files map to an empty dict (same as the Streamlit reader) so the
    frontend can render placeholders before the offline jobs have run.
    """
    sources = _METRICS_SUMMARY_SOURCES.get(module)
    if sources is None:
        allowed = ", ".join(_METRICS_SUMMARY_SOURCES)
        raise ValueError(f"未知的 module：{module!r}。可用：{allowed}")
    cfg = load_config()
    out: Dict[str, Any] = {}
    for key, (section, path_key) in sources.items():
        out[key] = _read_json_or_empty(cfg[section][path_key])
    return out


def paderborn_eval() -> Dict[str, Any]:
    """Full Paderborn (Module C) evaluation: method / features / results / summary.

    Single source for the whole Module C page (baseline CV vs artificial→real
    generalization + confusion matrices). Empty dict if the job has not run.
    """
    return _read_json_or_empty(load_config()["paderborn"]["metrics"])


# --- Module B+ (XJTU multi-trajectory generalization) ------------------------
def xjtu_generalization() -> Dict[str, Any]:
    """Fixed-param per-bearing degradation detection: method / per_bearing / aggregate."""
    return _read_json_or_empty(load_config()["xjtu"]["gen_metrics"])


def xjtu_lobo_loco() -> Dict[str, Any]:
    """Supervised RUL generalization: leave-one-bearing-out + leave-one-condition-out."""
    cfg = load_config()
    return {
        "lobo": _read_json_or_empty(cfg["xjtu"]["lobo_metrics"]),
        "loco": _read_json_or_empty(cfg["xjtu"]["loco_metrics"]),
    }


def xjtu_domain_adapt() -> Dict[str, Any]:
    """E1 cross-condition domain adaptation ablation: results / summary."""
    return _read_json_or_empty(load_config()["xjtu"]["domain_adapt"]["da_metrics"])


# --- Module B (IMS single-trajectory health curve) ---------------------------
def ims_metrics() -> Dict[str, Any]:
    """IMS RUL metrics/meta: indicator, FPT index/time, lead time, near-failure errors."""
    return _read_json_or_empty(load_config()["ims"]["rul_metrics"])


def ims_health_curve() -> List[Dict[str, Any]]:
    """IMS health/RUL timeseries: timestamp, rul_true, rul_pred, health, is_degrading.

    Meta (FPT index, lead time) is served separately by /ims/metrics. ``to_json``
    turns the pre-degradation NaN ``rul_pred`` into valid JSON ``null``. Empty
    list if the RUL extrapolation job has not run.
    """
    path = resolve(load_config()["ims"]["rul_predictions"])
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


# --- Maintenance knowledge base (TF-IDF RAG) ---------------------------------
def knowledge_documents() -> List[Dict[str, Any]]:
    """List local KB documents (source, title, preview, chars)."""
    from src.knowledge.maintenance_rag import list_documents

    return list_documents()


def knowledge_search(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """TF-IDF keyword search over the KB; each hit has text/score/source/title/topic."""
    from src.knowledge.maintenance_rag import search

    return search(query, top_k=top_k)


# --- Module Servo (main line) -------------------------------------------------
def servo_model_info() -> Dict[str, Any]:
    from src.models.servo_predict import load_servo_models

    b = load_servo_models()
    return {
        "feature_set": b.config.get("feature_set"),
        "feature_columns": b.feature_columns,
        "labels": b.config.get("labels"),
        "clf_model": b.config.get("clf_model"),
        "reg_model": b.config.get("reg_model"),
        "clf_macro_f1": b.config.get("clf_macro_f1"),
        "reg_r2": b.config.get("reg_r2"),
        "placeholder": b.config.get("placeholder"),
    }


def servo_predict_one(features: Dict[str, Any]) -> Dict[str, Any]:
    from src.models.servo_predict import load_servo_models, predict_servo

    b = load_servo_models()
    missing = [c for c in b.feature_columns if c not in features]
    if missing:
        raise ValueError(f"缺少必要特徵欄位：{missing}")
    return predict_servo(features)


def servo_glossary() -> List[Dict[str, Any]]:
    """Motor field glossary: name / zh / desc / meaning / anomaly."""
    from src.servo.field_glossary import FIELD_DOCS

    return FIELD_DOCS


def servo_feature_sets() -> Dict[str, Any]:
    """Available feature sets: key -> label / desc / columns."""
    from src.features.servo_features import FEATURE_SETS

    return FEATURE_SETS


def servo_samples() -> List[Dict[str, Any]]:
    """Demo sample rows the dashboard can select from (features + ylabel / DV)."""
    path = resolve(load_config()["servo"]["sample_predictions"])
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


def servo_reference_metrics() -> Dict[str, Any]:
    """Reference baselines for the training simulator: clf / reg / dl."""
    cfg = load_config()["servo"]
    return {
        "clf": _read_json_or_empty(cfg["clf_metrics"]),
        "reg": _read_json_or_empty(cfg["reg_metrics"]),
        "dl": _read_json_or_empty(cfg["dl_metrics"]),
    }
