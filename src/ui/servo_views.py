"""Streamlit page renderers for Module Servo (the project main line).

Kept out of ``app/streamlit_app.py`` so the main file only wires navigation +
dispatch.  Each ``render_*`` draws one page body (the hero / action bar / KPI
strip are drawn by the main file before dispatch).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.servo_features import FEATURE_SETS
from src.models import servo_simulator as sim
from src.models.servo_predict import predict_servo
from src.knowledge.maintenance_rag import list_documents, search
from src.llm.maintenance_assistant import answer_question, generate_report
from src.servo.field_glossary import (
    FIELD_DOCS,
    HEALTH_LABEL_TONE,
    HEALTH_LABEL_ZH,
)
from src.ui import style
from src.utils.paths import load_config, resolve

_RISK_LABEL = {"Low": "低", "Medium": "中", "High": "高"}
_RISK_TONE = {"Low": "good", "Medium": "warn", "High": "danger"}
_PRIMARY = "#6366f1"


# ---------------------------------------------------------------------------
# data helpers (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _features() -> pd.DataFrame:
    return pd.read_parquet(resolve(load_config()["servo"]["processed_features"]))


@st.cache_data(show_spinner=False)
def _demo() -> pd.DataFrame:
    return pd.read_csv(resolve(load_config()["servo"]["feature_demo"]))


@st.cache_data(show_spinner=False)
def _samples() -> pd.DataFrame:
    return pd.read_csv(resolve(load_config()["servo"]["sample_predictions"]))


@st.cache_data(show_spinner=False)
def _metric(name_key: str) -> dict:
    p = resolve(load_config()["servo"][name_key])
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _placeholder_note() -> None:
    if load_config()["servo"].get("placeholder", True):
        style.note(
            "目前模型以 <b>placeholder 合成資料</b> 訓練，僅供流程展示；"
            "下載真實 PHM 資料並重訓後即為正式結果。", kind="warn")


def _proba_bar(proba: Dict[str, float]) -> go.Figure:
    order = ["LN", "LO", "MED", "HI"]
    items = [(k, proba.get(k, 0.0)) for k in order if k in proba]
    colors = {"LN": "#22c55e", "LO": "#84cc16", "MED": "#f59e0b", "HI": "#ef4444"}
    fig = go.Figure(go.Bar(
        x=[v for _, v in items],
        y=[f"{HEALTH_LABEL_ZH[k]} ({k})" for k, _ in items],
        orientation="h",
        marker_color=[colors[k] for k, _ in items],
        text=[f"{v*100:.1f}%" for _, v in items], textposition="auto"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(range=[0, 1], tickformat=".0%"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------------------------------------------------------------------------
# Page: Servo health dashboard (single prediction + structured output)
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    _placeholder_note()
    samples = _samples()
    labels = list(samples["ylabel"]) if "ylabel" in samples else []

    with style.zone("sky", key="servo-pick"):
        style.section("選擇一筆運轉段")
        c1, c2 = st.columns([2, 1])
        with c1:
            opts = [f"#{i} · 真實標籤 {HEALTH_LABEL_ZH.get(l, l)} ({l})"
                    for i, l in enumerate(labels)]
            idx = st.selectbox("樣本（來自 demo 樣本筆）", range(len(opts)),
                               format_func=lambda i: opts[i]) if opts else None
        with c2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            go_pred = st.button("🔍 估測健康狀態", width="stretch", type="primary")

    if idx is None:
        style.note("找不到樣本筆，請先執行 <code>python -m src.data.build_servo_dataset</code>。",
                   kind="danger")
        return

    if go_pred or "servo_pred" not in st.session_state:
        pred = predict_servo(samples.iloc[idx])
        pred["_true_label"] = labels[idx] if idx < len(labels) else None
        st.session_state.servo_pred = pred
        # new prediction invalidates any previously generated LLM outputs
        st.session_state.pop("servo_llm_report", None)
        st.session_state.pop("servo_llm_answer", None)
    pred = st.session_state.servo_pred

    state = pred["predicted_health_state"]
    tone = HEALTH_LABEL_TONE.get(state, "primary")

    style.section("健康狀態估測結果")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        style.big_stat("健康狀態", f"{pred['health_state_zh']}",
                       sub=f"分類：{state}", tone=tone)
    with k2:
        style.big_stat("退化分數 DV", f"{pred['degradation_score']:.2f}",
                       sub="0=健康 · 1=高度退化", tone=tone)
    with k3:
        style.big_stat("健康分數", f"{pred['health_score']:.0f}",
                       sub="(1−DV)×100", tone=tone)
    with k4:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        style.risk_pill(pred["risk_level"], _RISK_LABEL.get(pred["risk_level"]))
        st.markdown(
            f"<div style='text-align:center;margin-top:10px;color:#64748b;font-size:.85rem;'>"
            f"模型信心 <b>{pred['model_confidence']*100:.0f}%</b></div>",
            unsafe_allow_html=True)

    if pred.get("consistency_warning"):
        style.note("⚠ " + pred["consistency_warning"], kind="danger")

    cL, cR = st.columns([1, 1])
    with cL:
        style.section("各健康狀態機率")
        st.plotly_chart(_proba_bar(pred["health_state_proba"]), width="stretch",
                        key="servo-proba")
        if pred.get("_true_label"):
            ok = pred["_true_label"] == state
            st.caption(("✅ 與真實標籤一致：" if ok else "⚠ 與真實標籤不同：")
                       + f"真實 {HEALTH_LABEL_ZH.get(pred['_true_label'])}（{pred['_true_label']}）")
    with cR:
        style.section("主要異常特徵")
        for t in pred["top_features"]:
            style.metric_with_bar(
                t["feature"], f"z = {t['z']}",
                min(1.0, abs(t["z"]) / 6.0),
                sub=t["hint"],
                tone="danger" if abs(t["z"]) > 3 else "warn" if abs(t["z"]) > 1.5 else "good")

    style.section("建議處置")
    for tip in pred["maintenance_advice"]:
        style.advice_card(tip)

    st.info("想要更完整的人話解釋與工單草稿？到側邊欄「LLM 維護助理」頁，"
            "它會接收這筆結果並產生維修建議。", icon="🤖")


# ---------------------------------------------------------------------------
# Page: AI training simulator
# ---------------------------------------------------------------------------
def render_simulator() -> None:
    _placeholder_note()
    df = _features()

    with style.zone("mint", key="sim-controls"):
        style.section("選擇訓練設定")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n = st.selectbox("資料量", [100, 500, 1000, 5000],
                             index=1, help="抽樣多少筆參與訓練")
        with c2:
            task = st.selectbox("任務", ["分類（健康狀態）", "回歸（退化值 DV）"])
        with c3:
            fs_keys = list(FEATURE_SETS.keys())
            fs = st.selectbox("特徵組", fs_keys,
                              index=fs_keys.index("engineered"),
                              format_func=lambda k: FEATURE_SETS[k]["label"])
        with c4:
            is_clf = task.startswith("分類")
            algos = sim.CLASSIFIER_NAMES if is_clf else sim.REGRESSOR_NAMES
            algo = st.selectbox("演算法", algos,
                                format_func=lambda a: sim.ALGO_LABELS.get(a, a))
        st.caption(f"特徵組「{FEATURE_SETS[fs]['label']}」：{FEATURE_SETS[fs]['desc']}")
        run = st.button("🚀 開始訓練（瀏覽器端小模型）", type="primary", width="stretch")

    _render_dl_expander()

    if not run:
        style.note("選好資料量 / 特徵組 / 演算法後按「開始訓練」，"
                   "比較小模型與離線 Reference Model 的差異。")
        return

    n = min(n, len(df))
    if is_clf:
        res = sim.run_classification(df, fs, algo, n)
        ref = _metric("clf_metrics")
        _show_classification(res, ref)
    else:
        res = sim.run_regression(df, fs, algo, n)
        ref = _metric("reg_metrics")
        _show_regression(res, ref)

    style.section("為什麼會這樣？")
    for note in sim.explain_result("clf" if is_clf else "reg",
                                   FEATURE_SETS[fs]["label"], res["n_samples"]):
        st.markdown(f"- {note}")


def _render_dl_expander() -> None:
    """Read-only offline DL results (deep learning never trains on the server)."""
    dl = _metric("dl_metrics")
    if not dl:
        return
    with st.expander("🧠 深度學習離線結果（唯讀，第二部分）"):
        st.caption(dl.get("note", ""))
        d1, d2, d3 = st.columns(3)
        d1.metric("MLP 分類 macro-F1", f"{dl.get('mlp_classification_macro_f1', 0):.3f}")
        d2.metric("MLP 回歸 R²", f"{dl.get('mlp_regression', {}).get('r2', 0):.3f}")
        d3.metric("MLP 回歸 MAE", f"{dl.get('mlp_regression', {}).get('mae', 0):.3f}")
        rec = dl.get("reconstruction_error_by_class", {})
        if rec:
            st.caption("PCA 重建誤差（以健康資料擬合）— 退化越嚴重，重建誤差應越大：")
            order = [l for l in ["LN", "LO", "MED", "HI"] if l in rec]
            fig = go.Figure(go.Bar(
                x=[HEALTH_LABEL_ZH.get(l, l) for l in order],
                y=[rec[l] for l in order],
                marker_color=["#22c55e", "#84cc16", "#f59e0b", "#ef4444"][:len(order)],
                text=[f"{rec[l]:.2f}" for l in order], textposition="auto"))
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch", key="servo-dl-recon")


def _show_classification(res: dict, ref: dict) -> None:
    style.section("訓練結果（分類）")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("資料量", f"{res['n_samples']}")
    k2.metric("特徵數", f"{res['n_features']}")
    k3.metric("Accuracy", f"{res['accuracy']:.3f}")
    k4.metric("Macro-F1", f"{res['macro_f1']:.3f}")
    st.caption(f"訓練時間：{res['train_time_s']:.3f} 秒　·　演算法："
               f"{sim.ALGO_LABELS.get(res['algo'], res['algo'])}")

    cL, cR = st.columns([1, 1])
    with cL:
        style.section("混淆矩陣（測試集 vs 真實標籤）")
        cm = np.array(res["confusion_matrix"])
        labels = res["labels"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues",
            text=cm, texttemplate="%{text}", showscale=False))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="預測", yaxis_title="真實",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch", key="sim-cm")
    with cR:
        style.section("小模型 vs Reference Model")
        ref_f1 = ref.get("macro_f1")
        rows = [{"模型": "你的小模型", "Macro-F1": res["macro_f1"]}]
        if ref_f1 is not None:
            rows.append({"模型": f"Reference（{ref.get('model','?')}）",
                         "Macro-F1": ref_f1})
        cmp = pd.DataFrame(rows)
        bar = go.Figure(go.Bar(x=cmp["模型"], y=cmp["Macro-F1"],
                               marker_color=[_PRIMARY, "#94a3b8"][:len(cmp)],
                               text=[f"{v:.3f}" for v in cmp["Macro-F1"]],
                               textposition="auto"))
        bar.update_layout(height=320, yaxis=dict(range=[0, 1]),
                          margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, width="stretch", key="sim-cmp")
        if ref_f1 is not None:
            gap = ref_f1 - res["macro_f1"]
            st.caption(f"Reference Model 以完整資料離線訓練（macro-F1 {ref_f1:.3f}）；"
                       f"你的小模型差距約 {gap:+.3f}。資料越多、特徵越貼切，差距越小。")


def _show_regression(res: dict, ref: dict) -> None:
    style.section("訓練結果（回歸 DV）")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("資料量", f"{res['n_samples']}")
    k2.metric("MAE", f"{res['mae']:.3f}")
    k3.metric("RMSE", f"{res['rmse']:.3f}")
    k4.metric("R²", f"{res['r2']:.3f}")
    st.caption(f"訓練時間：{res['train_time_s']:.3f} 秒　·　演算法："
               f"{sim.ALGO_LABELS.get(res['algo'], res['algo'])}")

    style.section("小模型 vs Reference Model")
    ref_r2 = ref.get("r2")
    rows = [{"模型": "你的小模型", "R²": res["r2"], "MAE": res["mae"]}]
    if ref_r2 is not None:
        rows.append({"模型": f"Reference（{ref.get('model','?')}）",
                     "R²": ref_r2, "MAE": ref.get("mae", float('nan'))})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if ref_r2 is not None:
        st.caption(f"Reference Model R²={ref_r2:.3f}、MAE={ref.get('mae',0):.3f}（離線、完整資料）。"
                   "小模型資料量小，R² 通常較低、MAE 較高。")


# ---------------------------------------------------------------------------
# Page: motor field glossary / data tutorial
# ---------------------------------------------------------------------------
def render_glossary() -> None:
    style.section("馬達訊號欄位解釋")
    st.caption("把伺服馬達常見訊號用白話說明，降低非專業觀眾的理解門檻。")
    df = pd.DataFrame(FIELD_DOCS).rename(columns={
        "name": "欄位", "zh": "中文", "desc": "說明",
        "meaning": "對伺服馬達的意義", "anomaly": "異常時可能代表"})
    st.dataframe(df, width="stretch", hide_index=True, height=560)

    style.section("特徵組說明")
    for key, spec in FEATURE_SETS.items():
        with st.expander(f"{spec['label']}（{key}）— {len(spec['columns'])} 個特徵"):
            st.write(spec["desc"])
            st.code(", ".join(spec["columns"]) or "（運動+電流+位置追隨的聯集）")


# ---------------------------------------------------------------------------
# Page: LLM maintenance assistant
# ---------------------------------------------------------------------------
def render_assistant() -> None:
    pred = st.session_state.get("servo_pred")
    if pred is None:
        # default to a MED sample so the page is usable directly
        s = _samples()
        row = s[s["ylabel"] == "MED"].iloc[0] if "MED" in set(s.get("ylabel", [])) \
            else s.iloc[len(s) // 2]
        pred = predict_servo(row)
        st.session_state.servo_pred = pred

    with style.zone("stone", key="llm-ctx"):
        style.section("目前的模型結果（助理輸入）")
        c1, c2, c3 = st.columns(3)
        c1.metric("健康狀態", f"{pred['health_state_zh']} ({pred['predicted_health_state']})")
        c2.metric("風險等級", _RISK_LABEL.get(pred["risk_level"], pred["risk_level"]))
        c3.metric("退化分數", f"{pred['degradation_score']:.2f}")
        st.caption("主要異常特徵：" + "、".join(t["feature"] for t in pred["top_features"]))
        if pred.get("consistency_warning"):
            style.note("⚠ " + pred["consistency_warning"], kind="danger")

    from src.llm.maintenance_assistant import _PROVIDER_LABEL, available_providers

    provs = available_providers()
    if provs:
        names = "、".join(_PROVIDER_LABEL.get(p, p) for p in provs)
        style.note(f"已偵測到 LLM 供應商：<b>{names}</b>（依序嘗試，失敗才退回離線範本）。",
                   kind="info")
    else:
        style.note(
            "未偵測到任何 LLM 供應商金鑰，將使用<b>離線 fallback 範本</b>。"
            "可設定免費供應商任一：<code>GROQ_API_KEY</code> / "
            "<code>OPENROUTER_API_KEY</code> / <code>GEMINI_API_KEY</code>"
            "（或 <code>ANTHROPIC_API_KEY</code>）後改用 LLM 生成。", kind="info")

    def _badge(src: str) -> str:
        return ("⚪ 離線範本" if src == "fallback"
                else f"🟢 {_PROVIDER_LABEL.get(src, src)}")

    cgen, cqa = st.columns(2)
    with cgen:
        style.section("生成維護建議")
        if st.button("🤖 產生維護報告（含工單草稿）", type="primary", width="stretch"):
            with st.spinner("生成中…"):
                chunks = search(
                    " ".join(t["feature"] for t in pred["top_features"]) + " 伺服馬達 滾珠螺桿",
                    top_k=3)
                st.session_state.servo_llm_report = generate_report(pred, chunks)
        rep = st.session_state.get("servo_llm_report")
        if rep:
            st.caption(f"來源：{_badge(rep['source'])}")
            st.markdown(rep["text"])
    with cqa:
        style.section("維修問答")
        q = st.text_input("輸入問題", value="目前狀況要先檢查什麼？")
        if st.button("詢問助理", width="stretch"):
            with st.spinner("回答中…"):
                chunks = search(q + " 伺服馬達 滾珠螺桿", top_k=3)
                st.session_state.servo_llm_answer = answer_question(q, pred, chunks)
        ans = st.session_state.get("servo_llm_answer")
        if ans:
            st.caption(f"來源：{_badge(ans['source'])}")
            st.markdown(ans["text"])


# ---------------------------------------------------------------------------
# Page: maintenance knowledge base
# ---------------------------------------------------------------------------
def render_knowledge() -> None:
    style.section("維修知識庫")
    docs = list_documents()
    st.caption(f"目前收錄 {len(docs)} 份離線知識文件（可由白名單爬蟲或手動整理擴充）。")

    cols = st.columns(2)
    for i, d in enumerate(docs):
        with cols[i % 2]:
            style.dash_tile("📄", d["title"], f"{d['source']} · {d['chars']} 字")

    style.section("關鍵字檢索（TF-IDF）")
    q = st.text_input("輸入症狀或關鍵字", value="位置誤差 變大 卡滯")
    if st.button("🔎 檢索", type="primary"):
        hits = search(q, top_k=5)
        if not hits:
            style.note("沒有檢索到相關片段，換個關鍵字試試。", kind="warn")
        for h in hits:
            with st.expander(f"[{h.get('title') or h.get('source')}] · 相關度 {h['score']}"):
                st.write(h["text"])
