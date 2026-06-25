"""LLM maintenance assistant.

Consumes the STRUCTURED model output from ``servo_predict`` (never replaces the
model's judgement) plus optionally-retrieved knowledge chunks, and produces a
conservative, human-readable maintenance write-up: plain-language explanation,
possible causes, recommended checks, priority, a work-order draft and a report
summary.

Multi-provider with graceful degradation: the providers listed in
``llm.providers`` are tried in order (Groq / OpenRouter / Gemini all expose
OpenAI-compatible endpoints with free tiers; Anthropic via its SDK). If none is
configured or reachable, it falls back to a deterministic offline template.
OpenAI-compatible calls use only the standard library, so no extra runtime
dependency is required. Output is phrased conservatively.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from src.servo.field_glossary import HEALTH_LABEL_ZH
from src.utils.paths import load_config

# Providers exposing an OpenAI-compatible /chat/completions endpoint.
_OPENAI_COMPAT_URL = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}
_PROVIDER_LABEL = {
    "groq": "Groq", "openrouter": "OpenRouter",
    "gemini": "Gemini", "anthropic": "Anthropic",
}

_SYSTEM_PROMPT = (
    "你是一位伺服馬達與滾珠螺桿機構的維護助理。你會收到一個機器學習模型的「結構化輸出」"
    "（健康狀態、退化分數、風險等級、主要異常特徵、模型信心）以及可能的維修知識片段。\n"
    "你的工作是把這些結果翻成現場人員看得懂的話，並給出保守的維護建議。嚴格遵守：\n"
    "1. 不要說「一定故障」「確定壞了」。一律使用「可能」「建議檢查」「需由現場人員確認」等措辭。\n"
    "2. 不要捏造模型沒有提供的數據；以提供的特徵與知識片段為依據。\n"
    "3. 你輔助判斷，不取代模型，也不對馬達下達控制命令。\n"
    "請用繁體中文，輸出以下小節：模型結果說明、可能原因、建議檢查項目、維修優先級、工單草稿、維修報告摘要。"
)


def available_providers() -> List[str]:
    """Configured providers whose API key env var is currently set, in order."""
    cfg = load_config().get("llm", {})
    if not cfg.get("enabled", True):
        return []
    out = []
    for prov in cfg.get("providers", []):
        env = cfg.get(prov, {}).get("api_key_env", "")
        if env and os.environ.get(env):
            out.append(prov)
    return out


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


def _call_openai_compat(url: str, api_key: str, model: str,
                        system: str, user: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_anthropic(pcfg: dict, system: str, user: str, max_tokens: int) -> str:
    import anthropic  # type: ignore

    client = anthropic.Anthropic()
    # NOTE: temperature/top_p are rejected (400) on this model tier; do not send.
    resp = client.messages.create(
        model=pcfg.get("model", "claude-opus-4-8"),
        max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", None) == "text")


def _call_llm(system: str, user: str) -> Tuple[str, str]:
    """Try each configured provider in order. Returns (text, provider).

    Raises if no provider produced text (caller then uses the offline template).
    """
    cfg = load_config().get("llm", {})
    if not cfg.get("enabled", True):
        raise RuntimeError("LLM 已停用")
    max_tokens = int(cfg.get("max_tokens", 1200))
    for prov in cfg.get("providers", []):
        pcfg = cfg.get(prov, {})
        key = os.environ.get(pcfg.get("api_key_env", ""))
        if not key:
            continue
        try:
            if prov in _OPENAI_COMPAT_URL:
                text = _call_openai_compat(_OPENAI_COMPAT_URL[prov], key,
                                           pcfg.get("model", ""), system, user, max_tokens)
            elif prov == "anthropic":
                text = _call_anthropic(pcfg, system, user, max_tokens)
            else:
                continue
            if text and text.strip():
                return text, prov
        except Exception:
            continue  # try the next provider
    raise RuntimeError("沒有可用的 LLM 供應商")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_report(prediction: Dict[str, Any],
                    chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """Full maintenance write-up. Returns {'text','source'} (source=provider|fallback)."""
    structured = _format_structured(prediction, chunks)
    try:
        text, prov = _call_llm(
            _SYSTEM_PROMPT,
            f"以下是模型的結構化輸出，請依系統指示產生完整維護建議：\n\n{structured}")
        return {"text": text, "source": prov}
    except Exception:
        return {"text": _fallback_report(prediction, chunks), "source": "fallback"}


def answer_question(question: str, prediction: Dict[str, Any],
                    chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """Conservative Q&A grounded in the prediction + knowledge chunks."""
    structured = _format_structured(prediction, chunks)
    try:
        text, prov = _call_llm(
            _SYSTEM_PROMPT + "\n現在請只回答使用者的問題，保持保守措辭。",
            f"模型結構化輸出：\n{structured}\n\n使用者問題：{question}")
        return {"text": text, "source": prov}
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
