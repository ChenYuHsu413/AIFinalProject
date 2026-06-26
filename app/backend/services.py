"""Thin service layer between the FastAPI handlers and the inference code.

Kept deliberately small so unit tests can target it directly.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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


def predict_explain(record: Dict[str, Any]) -> Dict[str, Any]:
    """SHAP explanation for a single record. ``supported`` is False when the
    best model is not a tree the TreeExplainer can handle (mirrors the UI note)."""
    from src.models.explain import explain_record, is_supported

    if not is_supported():
        return {"supported": False}
    e = explain_record(record)
    return {
        "supported": True,
        "feature_names": e.feature_names,
        "feature_values": e.feature_values,
        "shap_values": e.shap_values,
        "base_value": e.base_value,
        "model_output": e.model_output,
    }


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


def predict_batch_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Predict on a JSON list of raw records (What-if 1D/2D sweep, grids)."""
    if not records:
        return []
    return predict_records(records)


def comparison_metrics() -> List[Dict[str, Any]]:
    cfg = load_config()
    path = resolve(cfg["paths"]["metrics_csv"])
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def test_predictions() -> Dict[str, List]:
    """Test-set ground truth + scores for the interactive threshold tuner.

    Returns ``y_true`` / ``y_proba`` arrays; the confusion-matrix math runs on
    the client. Empty arrays if evaluation has not been run."""
    path = resolve(load_config()["paths"]["outputs_metrics"]) / "test_predictions.csv"
    if not path.exists():
        return {"y_true": [], "y_proba": []}
    df = pd.read_csv(path)
    return {
        "y_true": df["y_true"].astype(int).tolist(),
        "y_proba": df["y_proba"].astype(float).tolist(),
    }


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


_IMS_INDICATOR_CANDIDATES = ("b1_rms", "b1_kurtosis", "b1_band_BPFO", "b1_crest_factor")


def ims_health_indicator(indicator: Optional[str] = None) -> Dict[str, Any]:
    """Recompute the health curve + degradation onset (FPT) for a chosen indicator.

    Mirrors the Streamlit 'switch the health indicator, recompute FPT live'
    interaction. ``available`` is False if the IMS feature table is not built."""
    from src.models.rul_extrapolation import build_health_indicator, detect_fpt

    cfg = load_config()["ims"]
    path = resolve(cfg["processed_features"])
    if not path.exists():
        return {"available": False, "reason": "IMS 特徵表尚未建立。"}
    df = pd.read_parquet(path).sort_index()
    candidates = [c for c in _IMS_INDICATOR_CANDIDATES if c in df.columns]
    ind = indicator or cfg["health_indicator"]
    if ind not in df.columns:
        raise ValueError(f"未知的 indicator：{ind!r}。可用：{candidates}")
    hi, health, hi_base, hi_fail = build_health_indicator(
        df[ind], cfg["hi_smooth_window"], cfg["baseline_n"], cfg["fail_percentile"]
    )
    fpt_idx = detect_fpt(hi, cfg["baseline_n"], cfg["fpt_n_sigma"], cfg["fpt_consecutive"])
    ts = df.index
    fpt_t = ts[fpt_idx]
    return {
        "available": True,
        "indicator": ind,
        "candidates": candidates,
        "fpt_index": int(fpt_idx),
        "fpt_time": str(fpt_t),
        "lead_time_days": (ts[-1] - fpt_t).total_seconds() / 86400.0,
        "hi_base": hi_base,
        "hi_fail": hi_fail,
        "alarm_health": float(cfg["alarm_health"]),
        "points": [
            {"timestamp": str(t), "health": float(h)}
            for t, h in zip(ts, health.to_numpy())
        ],
    }


def _downsample(arr: np.ndarray, n: int) -> np.ndarray:
    """Evenly subsample ``arr`` to at most ``n`` points (keeps endpoints)."""
    if arr.size <= n:
        return arr
    return arr[np.linspace(0, arr.size - 1, n).astype(int)]


def ims_snapshot(index: int) -> Dict[str, Any]:
    """Raw vibration waveform + FFT spectrum for one IMS snapshot.

    Needs the 1.5 GB raw IMS dataset (not shipped); returns ``available: False``
    when absent (e.g. the cloud demo). Waveform is downsampled and the spectrum
    capped to 2 kHz to keep the payload small."""
    cfg = load_config()["ims"]
    raw_dir = resolve(cfg["raw_dir"])
    if not raw_dir.exists():
        return {
            "available": False,
            "reason": "原始波形需 1.5 GB IMS 原始資料（未隨專案上傳）；雲端 demo 不提供。",
        }
    from src.data.load_ims import list_ims_files, load_ims_file

    files = list_ims_files(raw_dir)
    if not 0 <= index < len(files):
        raise ValueError(f"快照序號超出範圍 0..{len(files) - 1}")
    ts_sel, path_sel = files[index]
    ch = load_ims_file(path_sel)[:, cfg["target_bearing"] - 1]
    fs = float(cfg["sampling_rate_hz"])
    mag = np.abs(np.fft.rfft(ch))
    freqs = np.fft.rfftfreq(ch.size, d=1.0 / fs)
    keep = freqs <= 2000.0  # defect frequencies (BPFO/BPFI/BSF/FTF) are all < 300 Hz
    return {
        "available": True,
        "index": index,
        "n_snapshots": len(files),
        "timestamp": str(ts_sel),
        "target_bearing": int(cfg["target_bearing"]),
        "sampling_rate_hz": fs,
        "waveform": _downsample(ch, 2048).tolist(),
        "spectrum": {"freqs": freqs[keep].tolist(), "mags": mag[keep].tolist()},
        "defect_freqs": cfg["defect_freqs"],
    }


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
