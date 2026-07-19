"""情境 18 特徵計算 —— BL / STIFF / FR 三族的 per-run 聚合。

**與 `dsp_analytics.py` 的關係(方法章節須引用):**
原函數 `dead_zone_width` / `reversal_error` 的簽名為 `(demand, actual, velocity)`,
既無法傳入取樣率換算後的搜尋窗長,也無法指定只在特定事件型別(指令換向)上取樣。
因此本檔採**參數化重實作**,並以逐位元等價測試錨定:當
`search_samples=20, move_threshold=0.05, events=全速度過零, fallback=原預設值` 時,
輸出與 `dsp_analytics` 原函數 **bit-identical**(見 `tests/test_s18_features.py`)。
`dsp_analytics.py` 原檔保持原樣不動。

**零事件policy:** 事件數為 0 時輸出 `NaN`,**絕不**落回原函數的預設常數
(`dead_zone_width` 的 0.05 / `reversal_error` 的 `mean|FE|*0.1`)——那會讓
「無法計算」偽裝成「量到一個小值」。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.s18.dsp_analytics import AdvancedMechanicalDiagnostics as _M

# ---------------------------------------------------------------- 事件偵測


def zero_crossing_events(velocity: np.ndarray) -> np.ndarray:
    """dsp_analytics 的原始事件定義:``v[i] * v[i-1] < 0`` 的所有 i。"""
    v = np.asarray(velocity)
    return np.where(v[1:] * v[:-1] < 0)[0] + 1


def commanded_reversal_events(demand: np.ndarray) -> np.ndarray:
    """指令換向:demand 階梯的 delta **變號**處(不是每個階梯)。

    Phase 0 實測每 run 有 4 次階梯但只有 0–3 次換向,且 0 次確實會發生。
    """
    d = np.asarray(demand)
    step_idx = np.where(np.abs(np.diff(d)) > 1e-9)[0] + 1
    if len(step_idx) < 2:
        return np.array([], dtype=int)
    deltas = d[step_idx] - d[step_idx - 1]
    flip = np.where(deltas[1:] * deltas[:-1] < 0)[0] + 1
    return step_idx[flip]


# ------------------------------------------------- 參數化版本(等價測試錨定)


def dead_zone_width_param(pos_demand, pos_actual, events: Sequence[int],
                          search_samples: int = 20,
                          move_threshold: float = 0.05,
                          fallback: Optional[float] = None) -> float:
    """死區寬度。演算法同 `dsp_analytics.dead_zone_width`,但事件點與搜尋窗長外部指定。

    ``fallback=None`` -> 無事件時回傳 NaN(本專案用法)。
    ``fallback=0.05`` -> 重現原函數行為(等價測試用)。
    """
    d = np.asarray(pos_demand)
    a = np.asarray(pos_actual)
    n = len(a)
    dead_zones = []
    for i in events:
        for j in range(i, min(i + search_samples, n)):
            if abs(a[j] - a[i]) > move_threshold:
                dead_zones.append(abs(d[j] - d[i]))
                break
    if not dead_zones:
        return float("nan") if fallback is None else float(fallback)
    return float(np.mean(dead_zones))


def reversal_error_param(pos_demand, pos_actual, events: Sequence[int],
                         fallback_on_empty: bool = False) -> float:
    """反轉誤差。演算法同 `dsp_analytics.reversal_error`,事件點外部指定。

    ``fallback_on_empty=True`` 重現原函數的 ``mean|demand-actual| * 0.1`` 退化公式。
    """
    d = np.asarray(pos_demand)
    a = np.asarray(pos_actual)
    if len(events) == 0:
        if fallback_on_empty:
            return float(np.mean(np.abs(d - a)) * 0.1)
        return float("nan")
    return float(np.mean([abs(d[i] - a[i]) for i in events]))


def stiff_torque_slope(following_error, torque) -> float:
    """`STIFF_TorqueSlope` —— 扭矩對追隨誤差的斜率(等效剛性)。

    依 0715-Sup 規格本意實作:`polyfit(FE, torque)[0]`。
    `dsp_analytics.force_displacement_slope` **與其規格不符**——該函數算的是
    `polyfit(fe, pos)[0]`(位置對 FE),完全未用到 torque,故本專案將其輸出
    改名為 `PosFE_Slope` 並降級為探索性特徵,不做剛性宣稱。
    """
    fe = np.asarray(following_error)
    tq = np.asarray(torque)
    if len(fe) < 3 or np.var(fe) < 1e-8:
        return float("nan")
    return float(np.polyfit(fe, tq, 1)[0])


def viscous_from_complement(velocity, torque, speed_min: float = 100.0) -> float:
    """黏滯摩擦係數,高速帶取**庫倫帶的補集** ``|v| >= speed_min``。

    dsp_analytics 原本用 ``|v| > 500``,但本資料 |v| 最大僅 183 -> 命中 0 筆 ->
    恆回傳預設常數 0.001。改用庫倫帶(|v| < 100)的補集,零新增自由參數。
    """
    v = np.abs(np.asarray(velocity))
    t = np.abs(np.asarray(torque))
    m = v >= speed_min
    if int(np.sum(m)) <= 5:
        return float("nan")
    return float(np.polyfit(v[m], t[m], 1)[0])


# ------------------------------------------------------------ per-run 聚合

FEATURE_COLUMNS = [
    "BL_DeadZone_cmd", "BL_DeadZone_zc",
    "BL_ReversalErr_cmd", "BL_ReversalErr_zc",
    "BL_HystArea", "BL_DirFE_Asym",
    "STIFF_TorqueSlope", "STIFF_ComplStd", "PosFE_Slope",
    "FR_Coulomb", "FR_Viscous",
    "FE_RMS", "FE_Max",
]
META_COLUMNS = ["n_cmd_reversals", "n_zero_crossings"]

# 標「不適用」的特徵:仍計算存欄供稽核,但排除於單調性檢定與複合分數之外。
# BL_DeadZone —— 結構性不適用,理由見 config/s18_params.yaml dead_zone.status。
# FR_Viscous —— 預註冊穩定性判準未通過(同號條件),見 friction.viscous_status。
NOT_APPLICABLE = ["BL_DeadZone_cmd", "BL_DeadZone_zc", "FR_Viscous"]
ANALYSIS_FEATURES = [c for c in FEATURE_COLUMNS if c not in NOT_APPLICABLE]


def compute_run_features(run, params: Dict) -> Dict[str, float]:
    """一個 run -> 一列特徵。輸入僅限訊號欄,不含 DV / ylabel。"""
    d = run["rod_demand_pos"].to_numpy()
    a = run["rod_actual_pos"].to_numpy()
    v = run["rotor_speed"].to_numpy()
    tq = run["torque"].to_numpy()
    fe = a - d                      # repo 慣例:position_error = actual - demand

    ev_cmd = commanded_reversal_events(d)
    ev_zc = zero_crossing_events(v)
    dz = params["dead_zone"]
    fr = params["friction"]
    ss, mt = int(dz["search_samples"]), float(dz["move_threshold"])

    out: Dict[str, float] = {
        "BL_DeadZone_cmd": dead_zone_width_param(d, a, ev_cmd, ss, mt),
        "BL_DeadZone_zc": dead_zone_width_param(d, a, ev_zc, ss, mt),
        "BL_ReversalErr_cmd": reversal_error_param(d, a, ev_cmd),
        "BL_ReversalErr_zc": reversal_error_param(d, a, ev_zc),
        "BL_HystArea": _M.hysteresis_area(d, a),
        "BL_DirFE_Asym": _M.direction_dependent_following_error(d, a, v),
        "STIFF_TorqueSlope": stiff_torque_slope(fe, tq),
        "STIFF_ComplStd": _M.compliance_std(a, fe),
        "PosFE_Slope": _M.force_displacement_slope(a, fe, np.abs(tq)),
        "FR_Coulomb": _M.stribeck_friction_parameters(v, tq)[0],
        "FR_Viscous": viscous_from_complement(v, tq, float(fr["viscous_speed_min"])),
        "FE_RMS": float(np.sqrt(np.mean(fe ** 2))),
        "FE_Max": float(np.max(np.abs(fe))),
        "n_cmd_reversals": int(len(ev_cmd)),
        "n_zero_crossings": int(len(ev_zc)),
    }
    return out


def dead_zone_sensitivity(run, windows_ms: Sequence[float], fs_hz: float,
                          move_threshold: float = 0.05) -> Dict[str, float]:
    """搜尋窗長敏感度:同一 run 在數個窗長下的死區值(全過零事件)。"""
    d = run["rod_demand_pos"].to_numpy()
    a = run["rod_actual_pos"].to_numpy()
    ev = zero_crossing_events(run["rotor_speed"].to_numpy())
    out = {}
    for w in windows_ms:
        ss = int(round(w * fs_hz / 1000.0))
        out[f"{w:g}ms"] = dead_zone_width_param(d, a, ev, ss, move_threshold)
    return out
