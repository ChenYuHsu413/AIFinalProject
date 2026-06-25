"""LLM maintenance assistant.

Consumes the STRUCTURED model output from ``servo_predict`` (never replaces the
model's judgement) plus optionally-retrieved knowledge chunks, and produces a
conservative, human-readable maintenance write-up: plain-language explanation,
possible causes, recommended checks, priority, a work-order draft and a report
summary.

Uses the Anthropic SDK when ``ANTHROPIC_API_KEY`` is set; otherwise falls back
to a deterministic template so the feature works fully offline.  All output is
phrased conservatively ("可能 / 建議檢查 / 需由現場人員確認") — never "一定故障".
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.servo.field_glossary import HEALTH_LABEL_ZH
from src.utils.paths import load_config

_SYSTEM_PROMPT = (
    "你是一位伺服馬達與滾珠螺桿機構的維護助理。你會收到一個機器學習模型的「結構化輸出」"
    "（健康狀態、退化分數、風險等級、主要異常特徵、模型信心）以及可能的維修知識片段。\n"
    "你的工作是把這些結果翻成現場人員看得懂的話，並給出保守的維護建議。嚴格遵守：\n"
    "1. 不要說「一定故障」「確定壞了」。一律使用「可能」「建議檢查」「需由現場人員確認」等措辭。\n"
    "2. 不要捏造模型沒有提供的數據；以提供的特徵與知識片段為依據。\n"
    "3. 你輔助判斷，不取代模型，也不對馬達下達控制命令。\n"
    "請用繁體中文，輸出以下小節：模型結果說明、可能原因、建議檢查項目、維修優先級、工單草稿、維修報告摘要。"
)


def _client():
    """Return an Anthropic client, or None if unavailable (offline mode)."""
    cfg = load_config().get("llm", {})
    if not cfg.get("enabled", True):
        return None
    if not os.environ.get(cfg.get("api_key_env", "ANTHROPIC_API_KEY")):
        return None
    try:
        import anthropic  # type: ignore

        return anthropic.Anthropic()
    except Exception:
        return None


def _format_structured(prediction: Dict[str, Any],
                       chunks: Optional[List[Dict[str, Any]]]) -> str:
    state = prediction.get("predicted_health_state", "?")
    lines = [
        f"健康狀態：{state}（{prediction.get('health_state_zh', '')}）",
        f"退化分數 DV：{prediction.get('degradation_score', 0):.3f}（越高越退化）",
        f"健康分數：{prediction.get('health_score', 0):.1f}/100",
        f"風險等級：{prediction.get('risk_level', '?')}",
        f"模型信心：{prediction.get('model_confidence', 0):.2f}",
        "主要異常特徵（依偏離程度）：",
    ]
    for t in prediction.get("top_features", []):
        lines.append(f"  - {t['feature']}（z={t['z']}）：{t['hint']}")
    if prediction.get("placeholder"):
        lines.append("（註：目前為 placeholder 合成資料訓練的示範模型。）")
    if chunks:
        lines.append("\n檢索到的維修知識片段：")
        for c in chunks:
            src = c.get("title") or c.get("source", "")
            lines.append(f"  [{src}] {str(c.get('text', ''))[:300]}")
    return "\n".join(lines)


def _call_llm(system: str, user: str) -> str:
    client = _client()
    if client is None:
        raise RuntimeError("LLM 不可用")
    cfg = load_config().get("llm", {})
    # NOTE: temperature/top_p are rejected (400) on this model tier; do not send.
    resp = client.messages.create(
        model=cfg.get("model", "claude-opus-4-8"),
        max_tokens=int(cfg.get("max_tokens", 1200)),
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_report(prediction: Dict[str, Any],
                    chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """Full maintenance write-up. Returns {'text', 'source'} (source=llm|fallback)."""
    structured = _format_structured(prediction, chunks)
    try:
        text = _call_llm(_SYSTEM_PROMPT,
                         f"以下是模型的結構化輸出，請依系統指示產生完整維護建議：\n\n{structured}")
        return {"text": text, "source": "llm"}
    except Exception:
        return {"text": _fallback_report(prediction, chunks), "source": "fallback"}


def answer_question(question: str, prediction: Dict[str, Any],
                    chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """Conservative Q&A grounded in the prediction + knowledge chunks."""
    structured = _format_structured(prediction, chunks)
    try:
        text = _call_llm(
            _SYSTEM_PROMPT + "\n現在請只回答使用者的問題，保持保守措辭。",
            f"模型結構化輸出：\n{structured}\n\n使用者問題：{question}")
        return {"text": text, "source": "llm"}
    except Exception:
        return {"text": _fallback_answer(question, prediction), "source": "fallback"}


# ---------------------------------------------------------------------------
# Deterministic fallback templates (no API key needed)
# ---------------------------------------------------------------------------
def _fallback_report(prediction: Dict[str, Any],
                     chunks: Optional[List[Dict[str, Any]]]) -> str:
    state = prediction.get("predicted_health_state", "?")
    zh = HEALTH_LABEL_ZH.get(state, state)
    dv = prediction.get("degradation_score", 0)
    risk = prediction.get("risk_level", "?")
    tops = prediction.get("top_features", [])
    drivers = "、".join(t["feature"] for t in tops) or "（無明顯異常特徵）"
    causes = "\n".join(f"- {t['hint']}" for t in tops) or "- 目前無明顯異常徵兆。"
    checks = {
        "LN": ["維持例行監控即可。"],
        "LO": ["留意位置誤差與電流是否持續上升。", "確認潤滑狀態正常。"],
        "MED": ["檢查滾珠螺桿潤滑與是否卡滯。", "檢視負載是否異常。",
                "比對扭矩 / 電流趨勢。"],
        "HI": ["盡快安排檢查，評估維護時間窗。", "檢查潤滑、卡滯與負載。",
               "必要時降載或停機檢查（需由現場人員確認）。"],
    }.get(state, ["建議由現場人員進一步確認。"])
    priority = {"Low": "低（例行監控）", "Medium": "中（提高巡檢頻率）",
                "High": "高（盡快安排檢查）"}.get(risk, "—")
    check_lines = "\n".join(f"- {c}" for c in checks)
    kb = ""
    if chunks:
        kb = "\n## 參考知識\n" + "\n".join(
            f"- [{c.get('title') or c.get('source','')}] {str(c.get('text',''))[:160]}…"
            for c in chunks)
    return f"""## 模型結果說明
模型判斷目前為「**{zh}（{state}）**」，退化分數 DV 約 {dv:.2f}，風險等級為 **{risk}**。
主要異常特徵為 {drivers}。以下為保守的維護建議，實際情況**需由現場人員確認**。

## 可能原因
{causes}

## 建議檢查項目
{check_lines}

## 維修優先級
{priority}

## 工單草稿
- 設備：伺服馬達 / 滾珠螺桿機構
- 現象：模型判斷為 {zh}（{state}），主要異常 {drivers}
- 建議處置：{checks[0]}
- 風險等級：{risk}
- 備註：本工單由系統依模型輸出自動草擬，需現場人員複核。

## 維修報告摘要
系統依感測特徵推估目前健康狀態為 {zh}，風險 {risk}。建議依上述項目檢查並記錄結果；
本結果為決策輔助，不代表確定故障。{kb}

> （此內容由離線 fallback 範本產生；設定 ANTHROPIC_API_KEY 後可改用 LLM 生成更自然的敘述。）"""


def _fallback_answer(question: str, prediction: Dict[str, Any]) -> str:
    zh = prediction.get("health_state_zh", "")
    risk = prediction.get("risk_level", "?")
    tops = "、".join(t["feature"] for t in prediction.get("top_features", []))
    return (
        f"針對你的問題「{question}」：\n\n"
        f"目前模型判斷為 **{zh}**，風險等級 **{risk}**，主要異常特徵為 {tops}。\n"
        "建議檢查潤滑、是否卡滯與負載是否異常；實際狀況**需由現場人員確認**。\n\n"
        "> （離線 fallback 回答；設定 ANTHROPIC_API_KEY 後可獲得更完整的問答。）"
    )
