"""Feature engineering for Module Servo (PHM servomotor-driven ballscrew).

The raw PHM dataset is a long per-timestep time series.  We never feed the raw
CSV to the server; instead each ``run_index`` segment is aggregated into ONE
feature row (time-domain statistics).  ``ylabel`` is the health-state target
(classification) and ``DV`` the degradation-value target (regression).

This module is the single source of truth for:
  * which raw columns exist (``RAW_COLUMNS``),
  * how a run segment is aggregated (``aggregate_run`` / ``build_feature_table``),
  * the selectable feature sets used by the reference model + training simulator
    (``FEATURE_SETS``).

It deliberately has no Streamlit / model dependencies so it stays unit-testable.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Raw per-timestep columns as documented for the PHM servomotor dataset.
RAW_COLUMNS = [
    "time", "DV", "rod_demand_pos", "rod_actual_pos", "torque", "rotor_speed",
    "i_3p_a", "i_3p_b", "i_3p_c", "direct", "quadrature",
    "run_index", "transitions", "del_pos", "ylabel",
]

# Signals aggregated into per-run statistics. ``position_error`` is derived.
BASE_SIGNALS = [
    "rotor_speed", "torque", "del_pos",
    "i_3p_a", "i_3p_b", "i_3p_c", "direct", "quadrature",
    "rod_demand_pos", "rod_actual_pos", "position_error",
]
STATS = ["mean", "std", "min", "max", "rms"]

HEALTH_LABELS = ["LN", "LO", "MED", "HI"]  # fixed order, healthy -> most degraded


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def add_position_error(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``position_error = rod_actual_pos - rod_demand_pos`` if missing."""
    if "position_error" not in df.columns:
        df = df.copy()
        df["position_error"] = df["rod_actual_pos"] - df["rod_demand_pos"]
    return df


def aggregate_run(run: pd.DataFrame) -> Dict[str, float]:
    """Aggregate one ``run_index`` segment into a flat feature dict."""
    run = add_position_error(run)
    feats: Dict[str, float] = {}
    for sig in BASE_SIGNALS:
        x = pd.to_numeric(run[sig], errors="coerce").to_numpy()
        x = x[~np.isnan(x)]
        feats[f"{sig}_mean"] = float(np.mean(x)) if len(x) else 0.0
        feats[f"{sig}_std"] = float(np.std(x)) if len(x) else 0.0
        feats[f"{sig}_min"] = float(np.min(x)) if len(x) else 0.0
        feats[f"{sig}_max"] = float(np.max(x)) if len(x) else 0.0
        feats[f"{sig}_rms"] = _rms(x)
    # Combined three-phase current RMS (a single, interpretable current feature).
    ia = pd.to_numeric(run["i_3p_a"], errors="coerce").to_numpy()
    ib = pd.to_numeric(run["i_3p_b"], errors="coerce").to_numpy()
    ic = pd.to_numeric(run["i_3p_c"], errors="coerce").to_numpy()
    stacked = np.concatenate([ia, ib, ic])
    stacked = stacked[~np.isnan(stacked)]
    feats["current_rms"] = _rms(stacked)
    return feats


def build_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a raw per-timestep frame into one row per ``run_index``.

    Returns a feature table with the aggregated columns plus the labels
    ``ylabel`` (mode of the segment) and ``DV`` (mean of the segment).
    """
    raw = add_position_error(raw)
    rows: List[Dict[str, float]] = []
    for run_index, seg in raw.groupby("run_index"):
        feats = aggregate_run(seg)
        feats["run_index"] = int(run_index)
        # ylabel is (near) constant within a run -> take the mode.
        feats["ylabel"] = str(seg["ylabel"].mode().iloc[0])
        feats["DV"] = float(pd.to_numeric(seg["DV"], errors="coerce").mean())
        rows.append(feats)
    out = pd.DataFrame(rows).sort_values("run_index").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Selectable feature sets (reference model + training simulator)
# ---------------------------------------------------------------------------
def _agg(sig: str, stats: List[str]) -> List[str]:
    return [f"{sig}_{s}" for s in stats]


FEATURE_SETS: Dict[str, Dict[str, object]] = {
    "basic_motion": {
        "label": "基本運動特徵",
        "desc": "轉速 / 扭矩 / 位移增量的均值、標準差、RMS。",
        "columns": (
            _agg("rotor_speed", ["mean", "std", "rms"])
            + _agg("torque", ["mean", "std", "rms"])
            + _agg("del_pos", ["mean", "std", "rms"])
        ),
    },
    "current": {
        "label": "電流特徵",
        "desc": "三相電流與 D/Q 軸電流的 RMS / 標準差（負載與磁場/扭矩控制）。",
        "columns": [
            "i_3p_a_rms", "i_3p_b_rms", "i_3p_c_rms",
            "direct_rms", "direct_std", "quadrature_rms", "quadrature_std",
        ],
    },
    "position_tracking": {
        "label": "位置追隨特徵",
        "desc": "目標 / 實際位置與位置誤差（追隨能力 / 卡滯徵兆）。",
        "columns": [
            "rod_demand_pos_mean", "rod_actual_pos_mean",
            "position_error_mean", "position_error_max", "position_error_std",
        ],
    },
    "full": {
        "label": "全特徵",
        "desc": "運動 + 電流 + 位置追隨的完整聚合特徵。",
        "columns": [],  # filled below as the union of the three groups
    },
    "engineered": {
        "label": "工程精選特徵",
        "desc": "退化最敏感的精選：三相電流 RMS、扭矩 / 轉速波動、位置誤差、Q/D 軸 RMS。",
        "columns": [
            "current_rms", "torque_std", "rotor_speed_std",
            "position_error_mean", "position_error_max",
            "quadrature_rms", "direct_rms",
        ],
    },
}

# "full" = union of motion + current + position (order-preserving, de-duplicated).
_full: List[str] = []
for _name in ("basic_motion", "current", "position_tracking"):
    for _c in FEATURE_SETS[_name]["columns"]:  # type: ignore[index]
        if _c not in _full:
            _full.append(_c)
FEATURE_SETS["full"]["columns"] = _full


def feature_set_columns(name: str) -> List[str]:
    if name not in FEATURE_SETS:
        raise KeyError(f"未知的特徵組：{name}。可用：{list(FEATURE_SETS)}")
    return list(FEATURE_SETS[name]["columns"])  # type: ignore[arg-type]


def all_feature_columns() -> List[str]:
    """Every aggregated column that ``build_feature_table`` can produce."""
    cols = [f"{s}_{st}" for s in BASE_SIGNALS for st in STATS]
    cols.append("current_rms")
    return cols
