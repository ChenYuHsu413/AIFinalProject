"""Maintenance-advice decision layer (Module B / B+ extension E2).

Turns a single trajectory's *current* state — health score, whether degradation
onset (FPT) has been detected, and the current RUL estimate — into an actionable
recommendation:

  * a **risk level** (green / yellow / red),
  * a **suggested maintenance window** (maintain within this many hours from now),
  * a human-readable **rationale**,
  * an optional **cost-aware note** when illustrative cost parameters are supplied.

This is decision *support* heuristic, not control.  The cost parameters are
illustrative and the advice has **not** been validated against real maintenance
outcomes.  See ``docs/MODULE_B_PLUS_EXTENSIONS_PLAN.md`` (E2) and the project
honesty red lines in ``CLAUDE.md``.

The function is pure (no config / IO): callers pass the thresholds explicitly so
it can be reused by the Streamlit dashboard, a CLI, or the streaming replay (E3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

RISK_GREEN = "green"
RISK_YELLOW = "yellow"
RISK_RED = "red"

_RISK_LABEL_ZH = {
    RISK_GREEN: "綠 · 健康",
    RISK_YELLOW: "黃 · 退化中",
    RISK_RED: "紅 · 迫近失效",
}


@dataclass(frozen=True)
class Advice:
    """Maintenance recommendation for one trajectory at one point in time."""

    risk: str                                  # green / yellow / red
    risk_label_zh: str                         # 綠 · 健康 / 黃 · 退化中 / 紅 · 迫近失效
    suggested_window_hours: Optional[float]    # maintain within this many hours; None if N/A
    rationale: List[str] = field(default_factory=list)
    cost_note: Optional[str] = None            # only when cost params are supplied


def maintenance_advice(
    health: float,
    rul_hours: Optional[float],
    past_fpt: bool,
    *,
    alarm_health: float = 30.0,
    safety_margin: float = 0.3,
    cost_unplanned: Optional[float] = None,
    cost_planned: Optional[float] = None,
) -> Advice:
    """Map current health / FPT / RUL to a risk level, window and rationale.

    Parameters
    ----------
    health:
        Current health score on the 0..100 scale (100 = baseline-healthy, 0 = at
        the failure threshold).
    rul_hours:
        Current remaining-useful-life estimate in hours, or ``None`` when RUL is
        not estimable yet (e.g. pre-FPT, or too few points after onset to fit).
    past_fpt:
        Whether degradation onset (First Predicting Time) has been detected.
    alarm_health:
        Health at or below this level is treated as imminent failure (red).
    safety_margin:
        Fraction of the RUL held back as a safety buffer.  The suggested window
        is ``rul_hours * (1 - safety_margin)``.
    cost_unplanned, cost_planned:
        Illustrative cost of an unexpected stoppage vs a planned maintenance.
        When both are supplied (and a window exists) a cost-aware note is added.

    Returns
    -------
    Advice
        Frozen dataclass with the risk level, suggested window and rationale.
    """
    if not 0.0 <= safety_margin < 1.0:
        raise ValueError(f"safety_margin must be in [0, 1), got {safety_margin}")

    # --- risk level: anchored on health vs the alarm threshold, then FPT -----
    if health <= alarm_health:
        risk = RISK_RED
    elif past_fpt:
        risk = RISK_YELLOW
    else:
        risk = RISK_GREEN

    # --- suggested maintenance window: now + RUL * (1 - safety_margin) -------
    if rul_hours is not None and rul_hours > 0:
        window = float(rul_hours) * (1.0 - safety_margin)
    else:
        window = None

    # --- rationale ----------------------------------------------------------
    rationale: List[str] = []
    if risk == RISK_RED:
        rationale.append(
            f"健康度 {health:.0f} 已跌破告警門檻 {alarm_health:.0f}，視為迫近失效，"
            "建議立即安排停機檢修。"
        )
    elif risk == RISK_YELLOW:
        rationale.append(
            f"已偵測到退化起點（FPT），健康度 {health:.0f} 仍高於告警門檻 {alarm_health:.0f}，"
            "屬可規劃的退化中狀態。"
        )
    else:
        rationale.append(
            f"尚未偵測到退化起點，健康度 {health:.0f}，運轉健康，維持例行監控即可。"
        )

    if window is not None:
        rationale.append(
            f"剩餘壽命估計 ≈ {rul_hours:.1f} h，建議於 {window:.1f} h 內"
            f"（保留 {safety_margin:.0%} 安全裕度）安排維護。"
        )
    elif past_fpt:
        rationale.append("已過退化起點但剩餘壽命尚不可靠估計，建議提高巡檢頻率並持續監看趨勢。")

    # --- optional cost-aware note (illustrative) ----------------------------
    cost_note = None
    if cost_unplanned and cost_planned and window is not None:
        ratio = cost_planned / cost_unplanned
        cost_note = (
            f"成本對照（示意）：計畫維護成本約為非預期停機的 {ratio:.0%}。"
            f"在估計失效前約 {window:.1f} h（保留 {safety_margin:.0%} 安全裕度）安排計畫維護，"
            "期望成本低於拖到失效才被動停機。"
        )

    return Advice(
        risk=risk,
        risk_label_zh=_RISK_LABEL_ZH[risk],
        suggested_window_hours=window,
        rationale=rationale,
        cost_note=cost_note,
    )
