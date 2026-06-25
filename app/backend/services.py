"""Thin service layer between the FastAPI handlers and the inference code.

Kept deliberately small so unit tests can target it directly.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

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
