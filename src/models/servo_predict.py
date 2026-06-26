"""Inference for Module Servo: structured health-state + DV output.

Loads the reference classifier / regressor / feature-config and turns one
aggregated feature row into the STRUCTURED payload consumed by the dashboard
and the LLM maintenance assistant::

    {
      "predicted_health_state": "MED",
      "health_state_zh": "中度退化",
      "health_state_proba": {"LN": .05, "LO": .18, "MED": .61, "HI": .16},
      "model_confidence": 0.61,
      "degradation_score": 0.64,      # DV regressor output (0..1)
      "health_score": 36.0,           # (1 - DV) * 100
      "risk_level": "Medium",
      "consistency_warning": null,    # set when clf state and DV risk disagree
      "top_features": [{"feature": "...", "z": 3.1, "hint": "..."}],
      "maintenance_advice": ["..."],
      "placeholder": true
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.servo.field_glossary import HEALTH_LABEL_ZH, HEALTH_RISK_LEVEL, feature_hint
from src.utils.paths import load_config, resolve


@dataclass
class ServoBundle:
    clf: Any
    reg: Any
    feature_columns: List[str]
    config: Dict[str, Any]


_BUNDLE: Optional[ServoBundle] = None


def load_servo_models(force: bool = False) -> ServoBundle:
    global _BUNDLE
    if _BUNDLE is not None and not force:
        return _BUNDLE
    cfg = load_config()["servo"]
    clf_p, reg_p, fc_p = (resolve(cfg["clf_model"]), resolve(cfg["reg_model"]),
                          resolve(cfg["feature_config"]))
    if not (clf_p.exists() and reg_p.exists() and fc_p.exists()):
        raise FileNotFoundError(
            "找不到 Servo 參考模型。請先執行：\n"
            "  python -m src.data.build_servo_dataset\n"
            "  python -m src.models.train_servo"
        )
    import json
    clf_b = joblib.load(clf_p)
    reg_b = joblib.load(reg_p)
    config = json.loads(fc_p.read_text(encoding="utf-8"))
    _BUNDLE = ServoBundle(
        clf=clf_b["pipeline"], reg=reg_b["pipeline"],
        feature_columns=list(config["feature_columns"]), config=config)
    return _BUNDLE


def _risk_from_dv(dv: float, bands: Dict[str, float]) -> str:
    if dv < bands.get("low_max", 0.33):
        return "Low"
    if dv < bands.get("medium_max", 0.66):
        return "Medium"
    return "High"


_RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def _consistency_warning(state: str, dv_risk: str) -> Optional[str]:
    """Flag when the classifier's state and the DV regressor's risk strongly
    disagree (>=2 tiers apart), e.g. state=HI but DV implies Low risk.

    The state and the DV come from two INDEPENDENT models, so a large gap means
    the structured output is internally contradictory. A 1-tier gap is expected
    (4 health states map onto 3 risk tiers) and is not flagged.
    """
    expected = HEALTH_RISK_LEVEL.get(state)
    if expected not in _RISK_ORDER or dv_risk not in _RISK_ORDER:
        return None
    if abs(_RISK_ORDER[expected] - _RISK_ORDER[dv_risk]) >= 2:
        return (f"分類器判為「{HEALTH_LABEL_ZH.get(state, state)}」（對應風險 {expected}），"
                f"但退化值推得風險為 {dv_risk}，兩模型結果明顯不一致，請以實測複核為準。")
    return None


def _top_features(row: pd.Series, cols: List[str], baseline: Dict[str, Dict[str, float]],
                  k: int = 3) -> List[Dict[str, Any]]:
    """Features deviating most (by |z| vs healthy baseline)."""
    scored = []
    for c in cols:
        b = baseline.get(c, {"mean": 0.0, "std": 1.0})
        z = (float(row[c]) - b["mean"]) / (b["std"] or 1.0)
        scored.append((c, z))
    scored.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return [{"feature": c, "z": round(z, 2), "hint": feature_hint(c)}
            for c, z in scored[:k]]


def _advice(state: str, top: List[Dict[str, Any]]) -> List[str]:
    tips: List[str] = []
    if state in ("MED", "HI"):
        drivers = "、".join(t["feature"] for t in top)
        tips.append(
            f"模型判斷為「{HEALTH_LABEL_ZH.get(state, state)}」，主要異常特徵為 {drivers}。"
            "建議檢查滾珠螺桿潤滑、機構是否卡滯，以及負載是否異常（需由現場人員確認）。"
        )
    if state == "HI":
        tips.append("退化程度偏高：建議提高巡檢頻率並評估安排維護時間窗，避免非計畫停機。")
    for t in top:
        tips.append(f"可能徵兆 — {t['hint']}")
    if state in ("LN", "LO") and not any(t for t in top if abs(t["z"]) > 2):
        tips.append("目前狀態接近健康，維持例行監控即可。")
    # de-duplicate while keeping order
    seen, out = set(), []
    for t in tips:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def predict_servo(record: Dict[str, Any] | pd.Series) -> Dict[str, Any]:
    """Structured prediction for one aggregated feature row."""
    b = load_servo_models()
    row = pd.Series(record) if not isinstance(record, pd.Series) else record
    X = pd.DataFrame([{c: float(row[c]) for c in b.feature_columns}])

    proba_arr = b.clf.predict_proba(X)[0]
    classes = list(b.clf.named_steps["clf"].classes_)
    proba = {cls: round(float(p), 4) for cls, p in zip(classes, proba_arr)}
    state = classes[int(np.argmax(proba_arr))]
    confidence = float(np.max(proba_arr))

    dv = float(np.clip(b.reg.predict(X)[0], 0.0, 1.0))
    bands = b.config.get("dv_risk", {"low_max": 0.33, "medium_max": 0.66})
    risk = _risk_from_dv(dv, bands)
    top = _top_features(X.iloc[0], b.feature_columns,
                        b.config.get("healthy_baseline", {}))

    return {
        "predicted_health_state": state,
        "health_state_zh": HEALTH_LABEL_ZH.get(state, state),
        "health_state_proba": proba,
        "model_confidence": round(confidence, 4),
        "degradation_score": round(dv, 4),
        "health_score": round((1.0 - dv) * 100.0, 2),
        "risk_level": risk,
        "consistency_warning": _consistency_warning(state, risk),
        "top_features": top,
        "maintenance_advice": _advice(state, top),
        "placeholder": bool(b.config.get("placeholder", True)),
    }


def main() -> None:
    import json
    cfg = load_config()["servo"]
    demo = pd.read_csv(resolve(cfg["sample_predictions"]))
    b = load_servo_models()
    rec = demo.iloc[len(demo) // 2]
    print(f"參考模型：分類 {b.config['clf_model']} / 回歸 {b.config['reg_model']}")
    print("示範輸入（真實 ylabel = %s）：" % rec.get("ylabel"))
    print(json.dumps(predict_servo(rec), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
