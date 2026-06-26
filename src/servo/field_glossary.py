"""Human-readable glossary for the servomotor dataset.

Powers the "馬達欄位解釋 / 資料教學" page and the plain-language explanations
shown next to predictions.  Kept as plain data so it has no UI dependency.
"""
from __future__ import annotations

from typing import Dict, List, TypedDict


class FieldDoc(TypedDict):
    name: str        # raw column name
    zh: str          # Chinese name
    desc: str        # plain explanation
    meaning: str     # what it means for a servomotor system
    anomaly: str     # what an abnormal value might indicate


# Health-state label -> Chinese display + tone (matches src.ui.style tones).
HEALTH_LABEL_ZH: Dict[str, str] = {
    "LN": "健康",
    "LO": "輕度退化",
    "MED": "中度退化",
    "HI": "高度退化",
}
HEALTH_LABEL_TONE: Dict[str, str] = {
    "LN": "good",
    "LO": "primary",
    "MED": "warn",
    "HI": "danger",
}
# Map a health state to the coarse risk level used elsewhere in the project.
HEALTH_RISK_LEVEL: Dict[str, str] = {
    "LN": "Low",
    "LO": "Low",
    "MED": "Medium",
    "HI": "High",
}


FIELD_DOCS: List[FieldDoc] = [
    {"name": "time", "zh": "時間",
     "desc": "量測時間戳（取樣序）。",
     "meaning": "用來還原訊號的時間順序與運轉週期。",
     "anomaly": "時間不連續可能代表資料遺失或取樣異常。"},
    {"name": "DV", "zh": "退化值",
     "desc": "資料集標註的退化程度數值（degradation value）。",
     "meaning": "數值越大代表機構退化越嚴重，是回歸模型的目標。",
     "anomaly": "DV 持續上升代表退化累積，需安排檢查。"},
    {"name": "rod_demand_pos", "zh": "目標位置",
     "desc": "控制系統希望滾珠螺桿到達的位置命令。",
     "meaning": "代表上層控制的期望軌跡。",
     "anomaly": "與實際位置長期偏離代表追隨能力下降。"},
    {"name": "rod_actual_pos", "zh": "實際位置",
     "desc": "機構實際到達的位置。",
     "meaning": "反映機械端真實響應。",
     "anomaly": "落後目標位置可能表示阻力上升或卡滯。"},
    {"name": "position_error", "zh": "位置誤差",
     "desc": "實際位置與目標位置的差（rod_actual_pos − rod_demand_pos）。",
     "meaning": "伺服追隨能力的直接指標。",
     "anomaly": "誤差變大常見於潤滑不足、卡滯、負載異常或控制增益失準。"},
    {"name": "torque", "zh": "馬達扭矩",
     "desc": "馬達輸出的力量。",
     "meaning": "反映目前負載大小。",
     "anomaly": "異常升高可能表示負載增加、摩擦變大或機構卡滯。"},
    {"name": "rotor_speed", "zh": "轉子速度",
     "desc": "馬達轉子的轉速。",
     "meaning": "反映運動命令的執行情形。",
     "anomaly": "速度不穩可能表示控制響應異常或負載干擾。"},
    {"name": "i_3p_a", "zh": "A 相電流",
     "desc": "三相電流之一。",
     "meaning": "與負載 / 扭矩需求相關。",
     "anomaly": "異常可能代表負載、摩擦或控制異常。"},
    {"name": "i_3p_b", "zh": "B 相電流",
     "desc": "三相電流之一。",
     "meaning": "與負載 / 扭矩需求相關。",
     "anomaly": "三相不平衡可能代表繞組或驅動異常。"},
    {"name": "i_3p_c", "zh": "C 相電流",
     "desc": "三相電流之一。",
     "meaning": "與負載 / 扭矩需求相關。",
     "anomaly": "三相不平衡可能代表繞組或驅動異常。"},
    {"name": "direct", "zh": "D 軸電流",
     "desc": "dq 座標下的直軸電流。",
     "meaning": "與磁場（勵磁）控制有關。",
     "anomaly": "異常可能代表磁場控制或弱磁策略異常。"},
    {"name": "quadrature", "zh": "Q 軸電流",
     "desc": "dq 座標下的交軸電流。",
     "meaning": "與扭矩輸出高度相關。",
     "anomaly": "升高常伴隨負載 / 摩擦上升，是退化的敏感訊號。"},
    {"name": "del_pos", "zh": "位移增量",
     "desc": "相鄰取樣間的位置變化量。",
     "meaning": "反映瞬時運動速率。",
     "anomaly": "抖動變大可能表示運動不平順或機構磨耗。"},
    {"name": "run_index", "zh": "運轉段索引",
     "desc": "一段運轉週期的編號（特徵聚合的單位）。",
     "meaning": "用來把連續訊號切成可建模的段落。",
     "anomaly": "（索引欄，非健康指標；不代表剩餘壽命 RUL。）"},
    {"name": "transitions", "zh": "運動轉態",
     "desc": "運動方向 / 階段切換的標記。",
     "meaning": "協助界定每段運轉的邊界。",
     "anomaly": "（輔助欄位，用於切段。）"},
    {"name": "ylabel", "zh": "健康狀態",
     "desc": "資料集標註的健康等級：LN / LO / MED / HI。",
     "meaning": "分類模型的目標：健康 / 輕度 / 中度 / 高度退化。",
     "anomaly": "等級越高代表退化越嚴重，需提高維護優先級。"},
]


def field_docs_table() -> List[Dict[str, str]]:
    """Glossary as a list of plain dicts (for DataFrame display)."""
    return [dict(d) for d in FIELD_DOCS]


# Plain-language hints keyed by aggregated feature name -> shown when a feature
# ranks among the top abnormal drivers of a prediction.
FEATURE_HINTS: Dict[str, str] = {
    "current_rms": "三相電流 RMS 偏高，常伴隨負載或摩擦上升。",
    "torque_std": "扭矩波動偏大，可能機構阻力不均或卡滯。",
    "rotor_speed_std": "轉速波動偏大，可能控制響應或負載干擾。",
    "position_error_mean": "位置誤差平均偏高，追隨能力下降。",
    "position_error_max": "位置誤差峰值偏高，可能瞬間卡滯或過衝。",
    "quadrature_rms": "Q 軸電流 RMS 偏高，扭矩需求上升（負載 / 摩擦）。",
    "direct_rms": "D 軸電流 RMS 異常，與磁場控制相關。",
}


def feature_hint(col: str) -> str:
    """Best-effort plain-language hint for an aggregated feature column."""
    if col in FEATURE_HINTS:
        return FEATURE_HINTS[col]
    base = col.rsplit("_", 1)[0]
    for d in FIELD_DOCS:
        if d["name"] == base:
            return f"{d['zh']}（{d['name']}）：{d['anomaly']}"
    return f"{col} 偏離正常範圍。"
