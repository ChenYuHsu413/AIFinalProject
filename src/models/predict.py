"""Inference utilities: single-record and batch prediction.

The same helpers power the CLI (``python -m src.models.predict``), the
Streamlit dashboard and the FastAPI service.

Output schema (see README "Output schema" section)::

    {
      "failure_probability": 0.83,
      "predicted_class": 1,
      "health_score": 17.0,
      "risk_level": "High",
      "maintenance_advice": ["..."]
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import add_engineered_features
from src.utils.paths import load_config, resolve

# Columns the user must supply (the five raw numeric + Type)
REQUIRED_INPUT_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    pipeline: Any
    feature_columns: List[str]
    model_name: str
    feature_set: str
    metrics: Dict[str, float]


_BUNDLE: Optional[ModelBundle] = None
_FAILURE_BUNDLE: Optional[Dict[str, Any]] = None


# Short human-readable notes per failure type (used in predict_full output)
FAILURE_TYPE_NOTES = {
    "TWF": "刀具磨耗故障 (Tool Wear Failure)：建議優先評估換刀時機與刀具狀態。",
    "HDF": "散熱故障 (Heat Dissipation Failure)：建議檢查冷卻迴路、風扇與散熱片是否異常。",
    "PWF": "電源／功率故障 (Power Failure)：建議檢查供電穩定度與扭矩×轉速的瞬時功率設定。",
    "OSF": "過載故障 (Overstrain Failure)：建議檢查負載量、夾具夾持力與切削參數是否過嚴。",
    "RNF": "隨機故障 (Random Failure)：依資料集設計屬隨機事件，較難由感測訊號預測。",
}


def load_model(path: Optional[str | Path] = None, force: bool = False) -> ModelBundle:
    """Load (and cache) the persisted best-model bundle."""
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE
    cfg = load_config()
    p = resolve(path or cfg["paths"]["best_model"])
    if not p.exists():
        raise FileNotFoundError(
            f"找不到已訓練的模型：{p}\n"
            "請先執行：`python -m src.models.train`"
        )
    raw = joblib.load(p)
    _BUNDLE = ModelBundle(
        pipeline=raw["pipeline"],
        feature_columns=list(raw["feature_columns"]),
        model_name=raw["model_name"],
        feature_set=raw["feature_set"],
        metrics=dict(raw["metrics"]),
    )
    return _BUNDLE


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def prepare_input(records: pd.DataFrame | Dict[str, Any] | List[Dict[str, Any]]) -> pd.DataFrame:
    """Coerce caller input into a DataFrame with engineered features attached."""
    if isinstance(records, dict):
        df = pd.DataFrame([records])
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    else:
        df = records.copy()

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要輸入欄位：{missing}")
    df = add_engineered_features(df)
    return df


# ---------------------------------------------------------------------------
# Risk / advice rules
# ---------------------------------------------------------------------------
def _risk_level(prob: float) -> str:
    cfg = load_config()["risk"]
    if prob < cfg["low_max"]:
        return "Low"
    if prob < cfg["medium_max"]:
        return "Medium"
    return "High"


def _maintenance_advice(row: pd.Series, prob: float) -> List[str]:
    cfg = load_config()["advice_thresholds"]
    advice: List[str] = []
    if row["temp_diff"] >= cfg["high_temp_diff_K"]:
        advice.append(
            f"製程與環境溫差過大（{row['temp_diff']:.1f} K）：建議檢查散熱迴路與通風路徑。"
        )
    if row["Torque [Nm]"] >= cfg["high_torque_Nm"]:
        advice.append(
            f"扭矩偏高（{row['Torque [Nm]']:.1f} Nm）：請檢查機械負載並確認主軸是否有卡阻。"
        )
    if row["Tool wear [min]"] >= cfg["high_tool_wear_min"]:
        advice.append(
            f"刀具磨耗偏高（{row['Tool wear [min]']:.0f} 分鐘）：建議安排換刀或預防性保養時段。"
        )
    if row["Rotational speed [rpm]"] <= cfg["low_rotational_speed_rpm"]:
        advice.append(
            f"轉速偏低（{row['Rotational speed [rpm]']:.0f} rpm）：請確認驅動命令與阻抗扭矩是否異常。"
        )
    if prob >= load_config()["risk"]["medium_max"]:
        advice.append(
            "故障機率偏高：建議立即通報維護單位，並評估是否在下一批次前先停機檢查。"
        )
    elif prob >= load_config()["risk"]["low_max"]:
        advice.append(
            "故障機率屬中度：建議提高巡檢頻率，並持續監看接下來幾個班次的扭矩與溫度趨勢。"
        )
    if not advice:
        advice.append(
            "目前運轉條件看起來健康，請持續例行監控即可。"
        )
    return advice


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _result_for_row(row: pd.Series, prob: float) -> Dict[str, Any]:
    return {
        "failure_probability": round(float(prob), 4),
        "predicted_class": int(prob >= 0.5),
        "health_score": round((1.0 - float(prob)) * 100.0, 2),
        "risk_level": _risk_level(float(prob)),
        "maintenance_advice": _maintenance_advice(row, float(prob)),
    }


def load_failure_type_models(force: bool = False) -> Dict[str, Any]:
    """Load (and cache) the second-stage per-failure-type model bundle."""
    global _FAILURE_BUNDLE
    if _FAILURE_BUNDLE is not None and not force:
        return _FAILURE_BUNDLE
    cfg = load_config()
    p = resolve(cfg["paths"]["failure_type_model"])
    if not p.exists():
        raise FileNotFoundError(
            f"找不到第二階段故障類型模型：{p}\n"
            "請先執行：`python -m src.models.train_failure_types`"
        )
    _FAILURE_BUNDLE = joblib.load(p)
    return _FAILURE_BUNDLE


def predict_failure_types(record: Dict[str, Any]) -> Dict[str, float]:
    """Per-failure-type probability for a single raw record."""
    bundle = load_failure_type_models()
    df = prepare_input(record)
    out: Dict[str, float] = {}
    for ft, info in bundle.items():
        X = df[info["feature_columns"]]
        out[ft] = float(info["pipeline"].predict_proba(X)[0, 1])
    return out


def predict_full(record: Dict[str, Any], type_threshold: float = 0.3) -> Dict[str, Any]:
    """First-stage + second-stage prediction in one payload."""
    main = predict_single(record)
    types = predict_failure_types(record)
    ranked = sorted(types.items(), key=lambda kv: kv[1], reverse=True)
    likely = [ft for ft, p in ranked if p >= type_threshold]
    if not likely:
        likely = [ranked[0][0]] if ranked else []
    main["failure_type_probabilities"] = {k: round(v, 4) for k, v in types.items()}
    main["likely_failure_types"] = likely
    main["failure_type_notes"] = [
        FAILURE_TYPE_NOTES[ft] for ft in likely if ft in FAILURE_TYPE_NOTES
    ]
    return main


def predict_records(records) -> List[Dict[str, Any]]:
    """Predict on one or more records and return enriched result dicts."""
    bundle = load_model()
    df = prepare_input(records)
    X = df[bundle.feature_columns].copy()
    pipe = bundle.pipeline
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[:, 1]
    else:
        d = pipe.decision_function(X)
        proba = (d - d.min()) / (d.max() - d.min() + 1e-9)
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(proba):
        out.append(_result_for_row(df.iloc[i], float(p)))
    return out


def predict_single(record: Dict[str, Any]) -> Dict[str, Any]:
    return predict_records([record])[0]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def _demo_record() -> Dict[str, Any]:
    return {
        "Type": "L",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 108,
    }


def main() -> None:
    import json

    bundle = load_model()
    print(f"使用最佳模型：{bundle.model_name} / {bundle.feature_set}")
    print(f"測試集指標：{bundle.metrics}")
    demo = _demo_record()
    print("\n示範輸入：")
    print(json.dumps(demo, indent=2, ensure_ascii=False))
    res = predict_single(demo)
    print("\n預測結果：")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
