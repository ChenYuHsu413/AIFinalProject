"""Streamlit dashboard for the predictive-maintenance prototype.

Run::

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script and only puts its own directory on
# ``sys.path``.  Prepend the project root so the ``src.*`` imports resolve.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import numpy as np
import pandas as pd
import streamlit as st

try:
    import streamlit_shadcn_ui as ui
    HAS_SHADCN = True
except Exception:  # pragma: no cover
    HAS_SHADCN = False

from src.models.predict import (
    REQUIRED_INPUT_COLUMNS,
    load_model,
    predict_full,
    predict_records,
    predict_single,
)
from src.models.explain import explain_record, is_supported as shap_supported
from src.data.load_ims import list_ims_files, load_ims_file
from src.models.rul_extrapolation import build_health_indicator, detect_fpt
from src.ui import style
from src.ui.charts import (
    confusion_heatmap,
    failure_probability_gauge,
    failure_type_bar,
    health_curve,
    rul_forecast,
    input_radar,
    leaderboard_bar,
    one_d_sweep,
    probability_histogram,
    risk_donut,
    risk_landscape,
    shap_bar,
    sparkline,
    vibration_spectrum,
    vibration_waveform,
    xjtu_health_overlay,
)
from src.utils.paths import load_config, resolve


# ---------------------------------------------------------------------------
# Page config + global styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="伺服馬達預測性維護原型系統",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)
style.inject()


@st.cache_data(show_spinner=False)
def _metric_json(name: str) -> dict:
    """Read a small JSON under ``outputs/metrics/`` (empty dict if missing)."""
    p = _ROOT / "outputs" / "metrics" / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Model status & sidebar
# ---------------------------------------------------------------------------
try:
    bundle = load_model()
except FileNotFoundError:
    style.note(
        "找不到已訓練的模型。請先在終端執行 "
        "<code>python -m src.models.train</code> 產生 "
        "<code>outputs/models/best_model.joblib</code>。",
        kind="danger",
    )
    st.stop()


# Navigation = sidebar buttons grouped by module.  st.button fires on every
# click (unlike option_menu, which ignores a click on the already-selected
# item), so single-item groups stay reachable and the highlight is driven
# entirely by ``active_page`` (active = primary button, others = secondary).
NAV_GROUPS = [
    {"title": None,
     "items": [("首頁總覽", "🏠")]},
    {"title": "模組 A · 靜態風險 (AI4I)",
     "items": [("手動單筆預測", "🎯"), ("What-if 敏感度分析", "💡"),
               ("批次 CSV 上傳", "📥"), ("模型評估結果", "📊")]},
    {"title": "模組 B · 動態健康度 (IMS)",
     "items": [("健康度總覽", "💓"), ("RUL 預測", "📉"), ("互動探索", "🔍")]},
    {"title": "模組 B+ · 多軌跡泛化 (XJTU)",
     "items": [("多軌跡泛化", "🧬")]},
    {"title": None,
     "items": [("關於本專案", "ℹ️")]},
]

style.sidebar_brand(
    emoji="🔧",
    title="Predictive Maintenance",
    subtitle="伺服馬達故障風險預測原型",
)

if "active_page" not in st.session_state:
    st.session_state.active_page = "首頁總覽"
active = st.session_state.active_page

with st.sidebar:
    for g in NAV_GROUPS:
        if g["title"]:
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;color:#94a3b8;"
                f"letter-spacing:.04em;white-space:nowrap;margin:14px 4px 2px;'>"
                f"{g['title']}</div>",
                unsafe_allow_html=True,
            )
        for name, icon in g["items"]:
            if st.button(
                f"{icon}  {name}",
                key=f"nav::{name}",
                width="stretch",
                type="primary" if name == active else "secondary",
            ):
                st.session_state.active_page = name
                st.rerun()

page = st.session_state.active_page

style.sidebar_model_card(
    bundle.model_name, bundle.feature_set,
    bundle.metrics["f1"], bundle.metrics["recall"],
)
style.sidebar_dataset_card(
    "UCI AI4I 2020", "Synthetic · 10,000 筆 · 故障率 3.39%"
)
style.sidebar_footer(
    '<span class="sf-pill">DECISION SUPPORT</span>'
    '<span class="sf-pill">NOT CONTROL</span>'
    '<div style="margin-top:8px;">本系統提供維護建議，<b>不</b>直接控制馬達。</div>'
    '<div style="margin-top:8px;">'
    'Repo · <a href="https://github.com/ChenYuHsu413/AIFinalProject" target="_blank">'
    'ChenYuHsu413/AIFinalProject</a></div>'
)


# ---------------------------------------------------------------------------
# Page-specific hero + status strip
# ---------------------------------------------------------------------------
HEROES = {
    "首頁總覽": (
        "Dashboard · One-page view",
        "預測性維護原型系統 · 總覽",
        "一頁看完模型、資料、排行榜與最近活動。"
        "可從這裡跳到任何工作流：單筆預測、What-if、批次上傳、評估。",
    ),
    "手動單筆預測": (
        "Predict · Explain · Advise",
        "單筆故障風險預測",
        "輸入目前運轉條件，立即得到故障機率、健康分數、可能的故障類型，"
        "以及基於 SHAP 的可解釋分析。",
    ),
    "What-if 敏感度分析": (
        "Sensitivity · 1D / 2D",
        "What-if 敏感度分析",
        "拖動運轉參數滑桿，即時觀察故障機率的變化，並可掃描單一特徵或產生 "
        "2D 風險地景，找到「最安全的工作點」。",
    ),
    "批次 CSV 上傳": (
        "Batch · CSV",
        "批次 CSV 上傳預測",
        "一次處理多筆運轉條件，產出每筆的機率、健康分數、風險等級，"
        "並可下載結果 CSV。",
    ),
    "模型評估結果": (
        "Evaluation · Threshold tuner",
        "模型評估與互動式門檻",
        "比較 50 組（10 模型 × 5 特徵）訓練結果，並透過 slider 即時調整決策"
        "門檻、觀察 Precision / Recall / F1 的取捨。",
    ),
    "健康度總覽": (
        "Module B · Dynamic Health",
        "模組 B · 健康度總覽",
        "以 IMS 軸承 run-to-failure 數據，由振動特徵推導健康分數、偵測退化起點 "
        "(FPT)，並可調整告警門檻、沿時間軸回放整個運轉過程。",
    ),
    "RUL 預測": (
        "Module B · RUL Forecast",
        "模組 B · 剩餘壽命預測",
        "以退化趨勢外推估計剩餘壽命 (RUL)，並對照監督式回歸為何在單一退化軌跡上失效。",
    ),
    "互動探索": (
        "Module B · Explore",
        "模組 B · 互動探索",
        "切換健康指標即時重算退化起點，並檢視任一快照的原始振動波形與 FFT 頻譜"
        "（標出 BPFO 等軸承故障頻率）。",
    ),
    "多軌跡泛化": (
        "Module B+ · Cross-condition",
        "模組 B+ · 多軌跡泛化 (XJTU)",
        "以 XJTU-SY 15 顆軸承 × 3 種工況，用同一組固定參數驗證健康監測的跨軸承、"
        "跨工況泛化，並以 leave-one-bearing-out / leave-one-condition-out 檢視監督式 "
        "RUL 的條件遷移能力 —— 補上 IMS 單軌跡缺乏的泛化證據。",
    ),
    "關於本專案": (
        "About · Tech stack",
        "關於本專案",
        "以 UCI AI4I 2020 建立的端到端預測性維護原型系統。"
        "覆蓋 CRISP-DM、SHAP、Optuna、Streamlit、FastAPI、Docker、GitHub Actions。",
    ),
}
_eyebrow, _title, _subtitle = HEROES[page]

# ---- page-aware top header: chips / action bar / KPI strip switch by module ----
_REPO = "https://github.com/ChenYuHsu413/AIFinalProject"
_MODULE_B_PAGES = {"健康度總覽", "RUL 預測", "互動探索"}
_MODULE_BPLUS_PAGES = {"多軌跡泛化"}


def _page_module(p: str) -> str:
    if p in _MODULE_B_PAGES:
        return "B"
    if p in _MODULE_BPLUS_PAGES:
        return "Bplus"
    if p == "關於本專案":
        return "about"
    return "A"  # 首頁總覽 + 模組 A


_module = _page_module(page)

_HERO_CHIPS = {
    "A": ["CRISP-DM", "10 模型 × 5 特徵組合", "SHAP", "Optuna", "Streamlit",
          "FastAPI", "Docker"],
    "B": ["IMS 軸承 run-to-failure", "20 kHz 振動", "時域 / 頻域特徵",
          "FPT 退化起點", "趨勢外推 RUL"],
    "Bplus": ["XJTU-SY", "15 軸承 × 3 工況", "固定參數泛化",
              "LOBO / LOCO", "Domain shift"],
    "about": ["CRISP-DM", "SHAP", "Optuna", "Streamlit", "FastAPI",
              "Docker", "GitHub Actions"],
}
style.hero(
    eyebrow=_eyebrow, title=_title, subtitle=_subtitle,
    chips=_HERO_CHIPS[_module],
)


# Action bar — links switch with the active module.
_ACTIONS = {
    "A": [
        {"label": "FastAPI /docs", "icon": "📚",
         "url": "http://127.0.0.1:8000/docs", "primary": True},
        {"label": "GitHub Repo", "icon": "📁", "url": _REPO},
        {"label": "Model Card", "icon": "📜",
         "url": f"{_REPO}/blob/main/outputs/models/MODEL_CARD.md"},
        {"label": "Dataset (UCI)", "icon": "📊",
         "url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"},
    ],
    "B": [
        {"label": "GitHub Repo", "icon": "📁", "url": _REPO, "primary": True},
        {"label": "模組 B 成果", "icon": "📈",
         "url": f"{_REPO}/blob/main/docs/MODULE_B_RESULTS.md"},
        {"label": "IMS Dataset (NASA PCoE)", "icon": "📊",
         "url": "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"},
    ],
    "Bplus": [
        {"label": "GitHub Repo", "icon": "📁", "url": _REPO, "primary": True},
        {"label": "模組 B+ 規劃", "icon": "🧬",
         "url": f"{_REPO}/blob/main/docs/MODULE_B_PLUS_XJTU_PLAN.md"},
        {"label": "XJTU-SY Dataset", "icon": "📊",
         "url": "https://biaowang.tech/xjtu-sy-bearing-datasets/"},
    ],
    "about": [
        {"label": "GitHub Repo", "icon": "📁", "url": _REPO, "primary": True},
        {"label": "FastAPI /docs", "icon": "📚",
         "url": "http://127.0.0.1:8000/docs"},
        {"label": "Model Card", "icon": "📜",
         "url": f"{_REPO}/blob/main/outputs/models/MODEL_CARD.md"},
    ],
}
style.action_bar(_ACTIONS[_module])

# Top KPI strip — switches with the active module.
m = bundle.metrics


def _tone_for(value: float) -> str:
    if value >= 0.85:
        return "good"
    if value >= 0.65:
        return "primary"
    if value >= 0.45:
        return "warn"
    return "danger"


if _module == "A":
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        style.metric_with_bar(
            "Recall", f"{m['recall']:.3f}", m['recall'],
            sub="故障命中率", tone=_tone_for(m['recall']),
        )
    with kpi_cols[1]:
        style.metric_with_bar(
            "F1", f"{m['f1']:.3f}", m['f1'],
            sub="P/R 調和平均", tone=_tone_for(m['f1']),
        )
    with kpi_cols[2]:
        style.metric_with_bar(
            "ROC-AUC", f"{m['roc_auc']:.3f}", m['roc_auc'],
            sub="整體鑑別力", tone=_tone_for(m['roc_auc']),
        )
    with kpi_cols[3]:
        style.metric_with_bar(
            "PR-AUC", f"{m['pr_auc']:.3f}", m['pr_auc'],
            sub="不平衡首選", tone=_tone_for(m['pr_auc']),
        )
    with kpi_cols[4]:
        # FN ratio: 13 / 68 = ~19% miss rate. Convert to a 0–1 "ok" score.
        style.metric_with_bar(
            "Miss rate", f"{13/68:.1%}", 1 - 13/68,
            sub="FN 13 / 68 故障", tone="warn",
        )
elif _module == "B":
    _ims = _metric_json("ims_rul.json")
    _im = _ims.get("metrics", {})
    style.kpi_strip([
        {"label": "退化提前量",
         "value": f"{_ims.get('lead_time_days', 0):.1f} 天",
         "sub": "FPT → 失效預警"},
        {"label": "RUL MAE",
         "value": f"{_im.get('mae_hours', 0):.1f} h",
         "sub": "退化區趨勢外推"},
        {"label": "RUL RMSE",
         "value": f"{_im.get('rmse_hours', 0):.1f} h",
         "sub": "退化區"},
        {"label": "資料軌跡", "value": "IMS Set 2 · 1 軸承",
         "sub": "單軌跡 run-to-failure"},
    ])
elif _module == "Bplus":
    _agg = _metric_json("xjtu_generalization.json").get("aggregate", {})
    _lobo = _metric_json("xjtu_lobo.json").get("pooled", {})
    _loco = _metric_json("xjtu_loco.json").get("pooled", {})
    _n = _agg.get("n_bearings", 0)
    style.kpi_strip([
        {"label": "退化偵測", "value": f"{_n}/{_n} 軸承",
         "sub": "3 工況全數偵測"},
        {"label": "平均提前量",
         "value": f"{_agg.get('mean_lead_time_hours', 0):.1f} h",
         "sub": "固定參數 FPT"},
        {"label": "LOBO R²", "value": f"{_lobo.get('r2', 0):+.2f}",
         "sub": "工況內留一軸承"},
        {"label": "LOCO R²", "value": f"{_loco.get('r2', 0):+.2f}",
         "sub": "留一工況 · domain shift"},
    ])
# about: hero + action bar only (no metric strip)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
RISK_LABEL = {"Low": "低", "Medium": "中", "High": "高"}
RISK_TONE = {"Low": "good", "Medium": "warn", "High": "danger"}


# --- Module B (IMS) shared loaders -----------------------------------------
@st.cache_data(show_spinner=False)
def _ims_feature_table(path_str: str) -> pd.DataFrame:
    """Cached load of the IMS feature table (keyed by path string)."""
    return pd.read_parquet(path_str).sort_index()


def _ims_guard() -> tuple[dict, bool]:
    """Render the 'data not ready' notes; return (ims_cfg, ready)."""
    ims = load_config()["ims"]
    if not resolve(ims["processed_features"]).exists():
        style.note(
            "尚未建立 IMS 特徵表。請先依 <code>data/README.md</code> 放好 Set 2，"
            "再執行 <code>python -m src.data.build_ims_dataset</code>。",
            kind="warn",
        )
        return ims, False
    return ims, True


def _ims_predictions(ims: dict):
    """Return (preds_df, meta_dict) or (None, None) if RUL not computed yet."""
    pred_path = resolve(ims["rul_predictions"])
    if not pred_path.exists():
        style.note(
            "尚未計算健康曲線 / RUL。執行 "
            "<code>python -m src.models.rul_extrapolation</code> 後即會顯示。",
            kind="info",
        )
        return None, None
    preds = pd.read_csv(pred_path, parse_dates=["timestamp"])
    meta = json.loads(resolve(ims["rul_metrics"]).read_text(encoding="utf-8"))
    return preds, meta


@st.cache_data(show_spinner=False)
def _xjtu_feature_table(path_str: str) -> pd.DataFrame:
    """Cached load of the XJTU combined feature table (keyed by path string)."""
    return pd.read_parquet(path_str)


def _xjtu_guard() -> tuple[dict, bool]:
    """Render the 'data not ready' note; return (xjtu_cfg, ready)."""
    xj = load_config().get("xjtu")
    if xj is None or not resolve(xj["gen_summary"]).exists():
        style.note(
            "尚未產生 XJTU 多軌跡泛化結果。請放好 XJTU-SY（Condition 1）後依序執行 "
            "<code>python -m src.data.build_xjtu_dataset</code> 與 "
            "<code>python -m src.models.eval_xjtu_generalization</code>。",
            kind="warn",
        )
        return xj or {}, False
    return xj, True


def _xjtu_artifacts(xj: dict):
    """Load committed XJTU artefacts: (summary_df, lobo_dict|None, loco_dict|None)."""
    summary = pd.read_csv(resolve(xj["gen_summary"]))

    def _json(key: str):
        p = resolve(xj[key])
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    return summary, _json("lobo_metrics"), _json("loco_metrics")


def render_result_block(result: dict) -> None:
    """Headline: gauge + big health stat + risk pill, in a 3-column layout."""
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.plotly_chart(
            failure_probability_gauge(result["failure_probability"]),
            width='stretch',
            key=f"gauge-{id(result)}",
        )
    with c2:
        style.big_stat(
            "健康分數",
            f"{result['health_score']:.1f}",
            sub="(1 − 故障機率) × 100",
            tone=RISK_TONE.get(result["risk_level"], "primary"),
        )
    with c3:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        style.risk_pill(result["risk_level"],
                        RISK_LABEL.get(result["risk_level"], result["risk_level"]))
        if "predicted_class" in result:
            st.markdown(
                "<div style='text-align:center;margin-top:14px;color:#64748b;font-size:0.85rem;'>"
                f"預測類別：<b>{result['predicted_class']}</b>"
                f"（thr = 0.5）</div>",
                unsafe_allow_html=True,
            )


def render_advice(result: dict) -> None:
    style.section("維護建議")
    for tip in result.get("maintenance_advice", []):
        style.advice_card(tip)


# ---------------------------------------------------------------------------
# Page 0: dashboard / one-page overview
# ---------------------------------------------------------------------------
if page == "首頁總覽":
    cfg = load_config()
    metrics_csv = resolve(cfg["paths"]["metrics_csv"])

    # ---- 3 module entry tiles (A / B / B+) ----
    style.section("快速入口")
    tile_targets = [
        ("🅰", "模組 A · 單筆風險預測",
         "AI4I 靜態特徵 → 故障機率 + SHAP 解釋 + 維護建議",
         "手動單筆預測", "go-a"),
        ("🅱", "模組 B · 健康度總覽",
         "IMS 振動 → 健康指標、退化起點 (FPT)、RUL 外推",
         "健康度總覽", "go-b"),
        ("🅱➕", "模組 B+ · 多軌跡泛化",
         "XJTU 15 軸承 × 3 工況 → 跨軸承 / 跨工況泛化驗證",
         "多軌跡泛化", "go-bplus"),
    ]
    tile_cols = st.columns(3)
    for col, (icon, title, sub, target_page, key) in zip(tile_cols, tile_targets):
        with col:
            if style.dash_button_tile(icon, title, sub, key=key):
                st.session_state.active_page = target_page
                st.rerun()

    st.divider()

    # ---- main 2-col grid ----
    cL, cR = st.columns([1.2, 1])

    with cL:
        style.section("目前部署的模型")
        with style.zone("sky", key="home-model"):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                style.big_stat("Recall", f"{m['recall']:.3f}",
                               "故障命中率",
                               tone=_tone_for(m['recall']))
            with mc2:
                style.big_stat("F1", f"{m['f1']:.3f}",
                               "整體分數",
                               tone=_tone_for(m['f1']))
            with mc3:
                style.big_stat("PR-AUC", f"{m['pr_auc']:.3f}",
                               "不平衡資料首選",
                               tone=_tone_for(m['pr_auc']))
            st.markdown(
                f"<div style='color:#64748b;font-size:0.9rem;margin-top:10px;'>"
                f"<b>{bundle.model_name}</b>　·　特徵組合 "
                f"<code>{bundle.feature_set}</code>　·　共 "
                f"{len(bundle.feature_columns)} 個特徵</div>",
                unsafe_allow_html=True,
            )

        if metrics_csv.exists():
            style.section("Top 5 模型排行榜（依 F1）")
            comp = pd.read_csv(metrics_csv)
            st.plotly_chart(
                leaderboard_bar(comp, metric="f1", top_n=5),
                width='stretch',
            )

    with cR:
        style.section("資料集")
        with style.zone("mint", key="home-dataset"):
            style.kpi_strip([
                {"label": "Rows", "value": "10,000",
                 "sub": "AI4I 2020 全部"},
                {"label": "Failure rate", "value": "3.39%",
                 "sub": "嚴重不平衡"},
            ])
            style.kpi_strip([
                {"label": "Type L", "value": "60.0%",
                 "sub": "低成本變體"},
                {"label": "Type M", "value": "30.0%",
                 "sub": "中規格"},
                {"label": "Type H", "value": "10.0%",
                 "sub": "高規格"},
            ])
            st.caption(
                "資料來源：UCI Machine Learning Repository。"
                "由參數化過程模型產生，**不是**真實工廠紀錄。"
            )

        style.section("最近 What-if 活動")
        if "wif_history" in st.session_state and st.session_state.wif_history:
            history = st.session_state.wif_history
            mini = sparkline(history[-30:], color=style.PRIMARY, height=120)
            st.plotly_chart(mini, width='stretch')
            st.caption(
                f"已試 {len(history)} 步；"
                f"目前 {history[-1]*100:.1f}%　·　"
                f"歷史最高 {max(history)*100:.1f}%　·　"
                f"歷史最低 {min(history)*100:.1f}%"
            )
        else:
            style.note(
                "去 <b>What-if 敏感度分析</b> 頁拖動滑桿，"
                "之後這裡就會出現你最近的探索軌跡。",
            )

    st.divider()

    # ---- footer chips ----
    style.section("技術棧")
    style.kpi_strip([
        {"label": "ML",     "value": "scikit-learn", "sub": "+ XGBoost / LightGBM"},
        {"label": "Explain", "value": "SHAP",        "sub": "TreeExplainer"},
        {"label": "Tune",   "value": "Optuna",      "sub": "TPE sampler"},
        {"label": "UI",     "value": "Streamlit",   "sub": "+ Plotly + shadcn"},
        {"label": "API",    "value": "FastAPI",     "sub": "8 endpoints"},
        {"label": "Deploy", "value": "Docker",      "sub": "+ GitHub Actions CI"},
    ])


# ---------------------------------------------------------------------------
# Page 1: manual single prediction
# ---------------------------------------------------------------------------
elif page == "手動單筆預測":
    with style.zone("sky", key="manual-input"):
        style.section("輸入運轉條件")
        with st.form("manual"):
            c1, c2, c3 = st.columns(3)
            with c1:
                type_ = st.selectbox("產品類別 Type", ["L", "M", "H"], index=0)
                wear = st.number_input(
                    "Tool wear [min]", min_value=0.0, value=108.0, step=1.0,
                )
            with c2:
                air_t = st.number_input(
                    "Air temperature [K]", 270.0, 320.0, 298.1, step=0.1,
                )
                torque = st.number_input(
                    "Torque [Nm]", 0.0, 100.0, 42.8, step=0.1,
                )
            with c3:
                proc_t = st.number_input(
                    "Process temperature [K]", 270.0, 330.0, 308.6, step=0.1,
                )
                rpm = st.number_input(
                    "Rotational speed [rpm]", 0.0, 5000.0, 1551.0, step=10.0,
                )
            submitted = st.form_submit_button(
                "🚀 執行預測", type="primary", width='stretch',
            )

    if submitted:
        record = {
            "Type": type_,
            "Air temperature [K]": air_t,
            "Process temperature [K]": proc_t,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }
        try:
            result = predict_full(record)
            has_ft = True
        except FileNotFoundError:
            result = predict_single(record)
            has_ft = False

        # ----- big headline block -----
        render_result_block(result)

        # ----- tabs for the rest -----
        tabs = st.tabs([
            "🎯 結果與建議",
            "🔍 SHAP 解釋",
            "🔧 故障類型",
            "📊 運轉指紋",
        ])

        with tabs[0]:
            render_advice(result)

            with st.expander("📜 API 回傳的 JSON（可直接複製到 curl / Postman）"):
                cjson_l, cjson_r = st.columns([2, 1])
                with cjson_l:
                    import json as _json
                    st.code(
                        _json.dumps(result, indent=2, ensure_ascii=False),
                        language="json",
                    )
                with cjson_r:
                    st.markdown(
                        "**對應 FastAPI 端點**\n\n"
                        "`POST /predict_full`\n\n"
                        "把上方 JSON 對應的輸入欄位 send 過去，"
                        "就會得到同樣的回傳結構。",
                    )

        with tabs[1]:
            if not shap_supported():
                style.note("目前的最佳模型不支援 TreeExplainer，因此略過 SHAP 解釋。",
                           kind="warn")
            else:
                try:
                    with st.spinner("計算 SHAP 貢獻..."):
                        exp = explain_record(record)
                    df_sv = (
                        pd.DataFrame({
                            "feature": exp.feature_names,
                            "value": exp.feature_values,
                            "shap": exp.shap_values,
                        })
                        .assign(abs_shap=lambda d: d["shap"].abs())
                        .sort_values("abs_shap", ascending=False)
                        .head(10)
                        .iloc[::-1]
                        .reset_index(drop=True)
                    )
                    st.plotly_chart(shap_bar(df_sv), width='stretch')
                    style.note(
                        f"基準 log-odds = <b>{exp.base_value:.3f}</b>；"
                        f"本筆 log-odds = <b>{exp.model_output:.3f}</b>"
                        "（sigmoid 後 ≈ 機率）。紅色長條把預測推向「故障」，"
                        "綠色長條拉回「健康」。",
                    )
                except Exception as e:
                    style.note(f"無法產生 SHAP 解釋：{e}", kind="warn")

        with tabs[2]:
            if has_ft:
                st.plotly_chart(
                    failure_type_bar(
                        result["failure_type_probabilities"],
                        result["likely_failure_types"],
                    ),
                    width='stretch',
                )
                for note in result.get("failure_type_notes", []):
                    style.note(note)
            else:
                style.note(
                    "尚未訓練第二階段故障類型模型。請執行 "
                    "<code>python -m src.models.train_failure_types</code>",
                    kind="warn",
                )

        with tabs[3]:
            c_l, c_r = st.columns([1.1, 1])
            with c_l:
                st.plotly_chart(
                    input_radar(record, prob=result["failure_probability"]),
                    width='stretch',
                )
            with c_r:
                style.section("衍生特徵")
                style.kpi_strip([
                    {"label": "temp_diff", "value": f"{proc_t - air_t:.2f}",
                     "sub": "製程 − 環境溫度"},
                    {"label": "power_proxy", "value": f"{torque * rpm:.0f}",
                     "sub": "扭矩 × 轉速"},
                    {"label": "wear_torque", "value": f"{wear * torque:.0f}",
                     "sub": "磨耗 × 扭矩"},
                    {"label": "wear_speed", "value": f"{wear * rpm:.0f}",
                     "sub": "磨耗 × 轉速"},
                    {"label": "temp_wear", "value": f"{proc_t * wear:.0f}",
                     "sub": "製程溫度 × 磨耗"},
                ])


# ---------------------------------------------------------------------------
# Page 2: What-if sensitivity analysis
# ---------------------------------------------------------------------------
elif page == "What-if 敏感度分析":
    with style.zone("mint", key="whatif-input"):
        style.section("即時操控運轉條件")
        st.caption(
            "拖動下方滑桿，三個分頁會即時更新：預測結果、1D 掃描曲線、2D 風險地景。"
        )
        c1, c2 = st.columns(2)
        with c1:
            wif_type = st.selectbox("Type", ["L", "M", "H"], key="wif_type")
            wif_air = st.slider(
                "Air temperature [K]", 295.0, 305.0, 298.1, step=0.1, key="wif_air"
            )
            wif_proc = st.slider(
                "Process temperature [K]", 305.0, 315.0, 308.6, step=0.1, key="wif_proc"
            )
        with c2:
            wif_rpm = st.slider(
                "Rotational speed [rpm]", 1100.0, 2900.0, 1551.0, step=10.0, key="wif_rpm"
            )
            wif_torque = st.slider(
                "Torque [Nm]", 0.0, 80.0, 42.8, step=0.5, key="wif_torque"
            )
            wif_wear = st.slider(
                "Tool wear [min]", 0.0, 260.0, 108.0, step=1.0, key="wif_wear"
            )

    base_record = {
        "Type": wif_type,
        "Air temperature [K]": wif_air,
        "Process temperature [K]": wif_proc,
        "Rotational speed [rpm]": wif_rpm,
        "Torque [Nm]": wif_torque,
        "Tool wear [min]": wif_wear,
    }
    try:
        wif_result = predict_full(base_record)
        has_ft = True
    except FileNotFoundError:
        wif_result = predict_single(base_record)
        has_ft = False

    # ---- session-state probability history (for the sparkline) ----
    if "wif_history" not in st.session_state:
        st.session_state.wif_history = []
    st.session_state.wif_history.append(
        float(wif_result["failure_probability"])
    )
    st.session_state.wif_history = st.session_state.wif_history[-40:]
    history = st.session_state.wif_history

    tabs = st.tabs([
        "🎯 即時預測 + 指紋",
        "➡️ 1D 掃描",
        "🗺️ 2D 風險地景",
    ])

    with tabs[0]:
        render_result_block(wif_result)

        # ---- recent change history (sparkline + delta) ----
        if len(history) >= 2:
            delta = history[-1] - history[-2]
            delta_pct = delta * 100
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
            colour = "#dc2626" if delta > 0 else ("#16a34a" if delta < 0 else "#64748b")
            sp_c1, sp_c2 = st.columns([3, 1])
            with sp_c1:
                style.section("最近 40 步的故障機率走勢")
                st.plotly_chart(
                    sparkline(history, color=style.PRIMARY, height=80),
                    width='stretch',
                )
            with sp_c2:
                st.markdown(
                    f"<div style='padding-top:38px;text-align:center;"
                    f"font-size:1.6rem;font-weight:700;color:{colour};'>"
                    f"{arrow} {abs(delta_pct):.1f}%</div>"
                    f"<div style='text-align:center;color:#64748b;font-size:0.78rem;'>"
                    f"相對上一步</div>",
                    unsafe_allow_html=True,
                )

        c_l, c_r = st.columns([1, 1])
        with c_l:
            st.plotly_chart(
                input_radar(base_record, prob=wif_result["failure_probability"]),
                width='stretch',
            )
        with c_r:
            if has_ft:
                st.plotly_chart(
                    failure_type_bar(
                        wif_result["failure_type_probabilities"],
                        wif_result["likely_failure_types"],
                    ),
                    width='stretch',
                )
            else:
                style.note("尚未訓練二階段模型。", kind="warn")

    with tabs[1]:
        feature_ranges = {
            "Air temperature [K]": (295.0, 305.0, 41),
            "Process temperature [K]": (305.0, 315.0, 41),
            "Rotational speed [rpm]": (1100.0, 2900.0, 41),
            "Torque [Nm]": (0.0, 80.0, 41),
            "Tool wear [min]": (0.0, 260.0, 41),
        }
        sweep_feature = st.selectbox(
            "選擇要掃描的特徵",
            list(feature_ranges.keys()),
            index=3,
            key="sweep_feature",
        )
        lo, hi, n = feature_ranges[sweep_feature]
        sweep_values = np.linspace(lo, hi, n)
        sweep_probs = []
        for v in sweep_values:
            rec = dict(base_record)
            rec[sweep_feature] = float(v)
            sweep_probs.append(predict_single(rec)["failure_probability"])
        st.plotly_chart(
            one_d_sweep(sweep_values, sweep_probs, sweep_feature,
                        base_record[sweep_feature]),
            width='stretch',
        )

    with tabs[2]:
        feature_ranges = {
            "Air temperature [K]": (295.0, 305.0, 25),
            "Process temperature [K]": (305.0, 315.0, 25),
            "Rotational speed [rpm]": (1100.0, 2900.0, 25),
            "Torque [Nm]": (0.0, 80.0, 25),
            "Tool wear [min]": (0.0, 260.0, 25),
        }
        c3, c4 = st.columns(2)
        with c3:
            feat_x = st.selectbox("X 軸特徵", list(feature_ranges.keys()),
                                  index=3, key="heat_x")
        with c4:
            feat_y = st.selectbox("Y 軸特徵", list(feature_ranges.keys()),
                                  index=4, key="heat_y")
        if feat_x == feat_y:
            style.note("請選擇兩個不同的特徵。", kind="warn")
        else:
            with st.spinner("計算風險地景..."):
                xlo, xhi, _ = feature_ranges[feat_x]
                ylo, yhi, _ = feature_ranges[feat_y]
                grid_n = 25
                xs = np.linspace(xlo, xhi, grid_n)
                ys = np.linspace(ylo, yhi, grid_n)
                grid_records = []
                for yv in ys:
                    for xv in xs:
                        rec = dict(base_record)
                        rec[feat_x] = float(xv)
                        rec[feat_y] = float(yv)
                        grid_records.append(rec)
                grid_probs = [r["failure_probability"]
                              for r in predict_records(grid_records)]
                Z = np.array(grid_probs).reshape(grid_n, grid_n)
            st.plotly_chart(
                risk_landscape(xs, ys, Z, feat_x, feat_y,
                               base_record[feat_x], base_record[feat_y]),
                width='stretch',
            )
            style.note(
                "紅 = 高故障風險、綠 = 健康。白點是目前運轉設定。"
                "把白點移向綠色區域就是「最安全的工作點」。"
            )


# ---------------------------------------------------------------------------
# Page 3: batch CSV upload
# ---------------------------------------------------------------------------
elif page == "批次 CSV 上傳":
    with style.zone("sand", key="batch-input"):
        style.section("批次預測（CSV 上傳）")
        style.note(
            "上傳至少包含這些欄位的 CSV："
            f"<code>{', '.join(REQUIRED_INPUT_COLUMNS)}</code>"
            "；系統會自動補上衍生特徵並執行批次推論。"
        )
        uploaded = st.file_uploader("選擇 CSV 檔案", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
            if missing:
                style.note(f"CSV 缺少欄位：{missing}", kind="danger")
            else:
                results = predict_records(df[REQUIRED_INPUT_COLUMNS])
                out = df.copy()
                out["failure_probability"] = [r["failure_probability"] for r in results]
                out["predicted_class"] = [r["predicted_class"] for r in results]
                out["health_score"] = [r["health_score"] for r in results]
                out["risk_level"] = [r["risk_level"] for r in results]

                n_high = sum(r["risk_level"] == "High" for r in results)
                n_med = sum(r["risk_level"] == "Medium" for r in results)
                n_low = sum(r["risk_level"] == "Low" for r in results)
                probs = [r["failure_probability"] for r in results]
                mean_prob = float(np.mean(probs)) if probs else 0.0
                max_prob = float(np.max(probs)) if probs else 0.0

                # --- headline summary tiles ---
                style.section(f"批次摘要（共 {len(results)} 筆）")
                style.kpi_strip([
                    {"label": "Low / Medium / High",
                     "value": f"{n_low} / {n_med} / {n_high}",
                     "sub": "風險等級分布"},
                    {"label": "平均故障機率",
                     "value": f"{mean_prob*100:.1f}%",
                     "sub": "所有筆數的算術平均"},
                    {"label": "最高故障機率",
                     "value": f"{max_prob*100:.1f}%",
                     "sub": "整批最危險的單筆"},
                    {"label": "需立即處理",
                     "value": str(n_high),
                     "sub": "風險 ≥ 0.7 的筆數"},
                ])

                # --- visual summary: donut + histogram ---
                c_l, c_r = st.columns([1, 1.3])
                with c_l:
                    st.plotly_chart(
                        risk_donut(n_low, n_med, n_high),
                        width='stretch',
                    )
                with c_r:
                    st.plotly_chart(
                        probability_histogram(probs),
                        width='stretch',
                    )

                # --- top-5 highest-risk rows for quick action ---
                if n_high or n_med:
                    style.section("最高風險的 5 筆")
                    top5 = (
                        out.sort_values("failure_probability", ascending=False)
                        .head(5)[
                            REQUIRED_INPUT_COLUMNS
                            + ["failure_probability", "risk_level"]
                        ]
                        .reset_index(drop=True)
                    )
                    st.dataframe(
                        top5.style.format({"failure_probability": "{:.1%}"})
                        .background_gradient(
                            cmap="OrRd", subset=["failure_probability"],
                            vmin=0, vmax=1,
                        ),
                        width='stretch',
                    )

                # --- full result table ---
                with st.expander(f"完整結果表（{len(results)} 筆）", expanded=False):
                    st.dataframe(out, width='stretch')

                st.download_button(
                    "📥 下載預測結果 CSV",
                    out.to_csv(index=False).encode("utf-8-sig"),
                    file_name="predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as e:
            style.note(f"處理 CSV 時發生錯誤：{e}", kind="danger")


# ---------------------------------------------------------------------------
# Page 4: model evaluation
# ---------------------------------------------------------------------------
elif page == "模型評估結果":
    cfg = load_config()
    metrics_csv = resolve(cfg["paths"]["metrics_csv"])
    if not metrics_csv.exists():
        style.note("尚未產生比較表，請先執行 "
                   "<code>python -m src.models.train</code>。", kind="warn")
    else:
        comp = pd.read_csv(metrics_csv)
        tabs = st.tabs([
            "📊 跨模型比較",
            "🎚️ 互動式門檻",
            "🖼️ 訓練 / 評估圖表",
        ])

        with tabs[0]:
            top3 = comp.sort_values("f1", ascending=False).head(3).reset_index(drop=True)
            style.section("頒獎台 · Top 3 by F1")
            style.kpi_strip([
                {"label": "🥇 第一名",
                 "value": top3.iloc[0]["model_name"],
                 "sub": (f"F1 {top3.iloc[0]['f1']:.3f} · "
                         f"Recall {top3.iloc[0]['recall']:.3f} · "
                         f"{top3.iloc[0]['feature_set']}")},
                {"label": "🥈 第二名",
                 "value": top3.iloc[1]["model_name"],
                 "sub": (f"F1 {top3.iloc[1]['f1']:.3f} · "
                         f"Recall {top3.iloc[1]['recall']:.3f} · "
                         f"{top3.iloc[1]['feature_set']}")},
                {"label": "🥉 第三名",
                 "value": top3.iloc[2]["model_name"],
                 "sub": (f"F1 {top3.iloc[2]['f1']:.3f} · "
                         f"Recall {top3.iloc[2]['recall']:.3f} · "
                         f"{top3.iloc[2]['feature_set']}")},
            ])

            # ----- pick metric & view leaderboard chart -----
            style.section("依指標排序的視覺化排行榜")
            metric_choice = st.radio(
                "排序依據",
                ["f1", "recall", "precision", "roc_auc", "pr_auc"],
                horizontal=True, index=0, key="lb-metric",
            )
            top_n = st.slider(
                "顯示前 N 名", min_value=5, max_value=30,
                value=12, step=1, key="lb-topn",
            )
            st.plotly_chart(
                leaderboard_bar(comp, metric=metric_choice, top_n=top_n),
                width='stretch',
            )

            # ----- full sortable table tucked behind an expander -----
            with st.expander("完整 50 筆比較表（可排序、可複製）",
                             expanded=False):
                styled = (
                    comp[
                        ["model_name", "feature_set", "feature_count",
                         "accuracy", "precision", "recall",
                         "f1", "roc_auc", "pr_auc"]
                    ]
                    .sort_values(metric_choice, ascending=False)
                    .style.format({
                        "accuracy": "{:.3f}", "precision": "{:.3f}",
                        "recall": "{:.3f}", "f1": "{:.3f}",
                        "roc_auc": "{:.3f}", "pr_auc": "{:.3f}",
                    })
                    .background_gradient(
                        cmap="Greens",
                        subset=["recall", "f1", "pr_auc"],
                        vmin=0, vmax=1,
                    )
                )
                st.dataframe(styled, width='stretch')

        with tabs[1]:
            preds_path = resolve(cfg["paths"]["outputs_metrics"]) / "test_predictions.csv"
            if not preds_path.exists():
                style.note(
                    "尚未產生測試集機率檔。請先執行 "
                    "<code>python -m src.models.evaluate</code>。",
                    kind="warn",
                )
            else:
                tp = pd.read_csv(preds_path)
                n_total = len(tp)
                n_pos = int(tp["y_true"].sum())
                style.note(
                    f"測試集共 <b>{n_total}</b> 筆；其中故障樣本 <b>{n_pos}</b> 筆"
                    f"（比例 {n_pos / n_total * 100:.2f}%）。"
                )
                thr = st.slider(
                    "決策門檻 threshold（機率 ≥ threshold ⇒ 預測為故障）",
                    0.0, 1.0, 0.5, step=0.01,
                )
                y_true = tp["y_true"].astype(int).values
                y_pred = (tp["y_proba"].values >= thr).astype(int)
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                tp_ = int(((y_true == 1) & (y_pred == 1)).sum())
                precision = tp_ / (tp_ + fp) if (tp_ + fp) else 0.0
                recall = tp_ / (tp_ + fn) if (tp_ + fn) else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) else 0.0)

                style.kpi_strip([
                    {"label": "Precision", "value": f"{precision:.3f}",
                     "sub": "誤報越少越高"},
                    {"label": "Recall",    "value": f"{recall:.3f}",
                     "sub": "漏報越少越高"},
                    {"label": "F1",        "value": f"{f1:.3f}",
                     "sub": "兩者調和"},
                    {"label": "漏報 FN",   "value": str(fn),
                     "sub": f"誤報 FP = {fp}"},
                ])
                cm_mat = np.array([[tn, fp], [fn, tp_]])
                st.plotly_chart(confusion_heatmap(cm_mat, thr),
                                width='stretch')
                style.note(
                    "<b>怎麼讀</b>：往左拉門檻 → 預測為故障的樣本變多 → "
                    "Recall 上升、FN（漏報）下降，但 FP（誤報）通常會上升、"
                    "Precision 下降。預測性維護情境通常偏好較低的門檻以提高 Recall。"
                )

        with tabs[2]:
            fig_dir = resolve(cfg["paths"]["outputs_figures"])
            captions = {
                "confusion_matrix.png": "混淆矩陣",
                "roc_curve.png": "ROC 曲線",
                "pr_curve.png": "Precision-Recall 曲線",
                "compare_recall.png": "各模型 Recall 比較",
                "compare_f1.png": "各模型 F1 比較",
                "compare_pr_auc.png": "各模型 PR-AUC 比較",
                "compare_roc_auc.png": "各模型 ROC-AUC 比較",
                "feature_importance_native.png": "原生特徵重要性",
                "feature_importance_permutation.png": "Permutation 特徵重要性",
                "feature_count_vs_f1.png": "特徵數 vs F1",
                "feature_count_vs_recall.png": "特徵數 vs Recall",
                "feature_count_vs_pr_auc.png": "特徵數 vs PR-AUC",
            }
            cols = st.columns(2)
            for i, (img, cap) in enumerate(captions.items()):
                p = fig_dir / img
                if p.exists():
                    with cols[i % 2]:
                        st.image(str(p), caption=cap, width='stretch')


# ---------------------------------------------------------------------------
# Module B · 健康度總覽 (health overview: alarm slider + time scrubber)
# ---------------------------------------------------------------------------
elif page == "健康度總覽":
    ims, ready = _ims_guard()
    if ready:
        preds, meta = _ims_predictions(ims)
        if preds is not None:
            fpt_t = pd.to_datetime(meta["fpt_time"])
            health = preds["health"].to_numpy()
            ts = preds["timestamp"]

            # --- interaction 1: adjustable alarm threshold (live lead time) ---
            alarm = st.slider("維護告警門檻（健康分數）", 5, 60,
                              int(ims["alarm_health"]), step=1)
            below = np.where(health <= alarm)[0]
            if below.size:
                lead_days = (ts.iloc[-1] - ts.iloc[below[0]]).total_seconds() / 86400.0
                lead_txt, lead_sub = f"{lead_days:.1f} 天", f"門檻 {alarm} @ {ts.iloc[below[0]]:%m-%d %H:%M}"
            else:
                lead_txt, lead_sub = "未觸發", f"健康未跌破 {alarm}"

            m = meta["metrics"]
            style.kpi_strip([
                {"label": "告警提前量", "value": lead_txt, "sub": lead_sub},
                {"label": "退化起點 FPT", "value": f"{fpt_t:%m-%d %H:%M}",
                 "sub": f"提前 {meta['lead_time_days']:.1f} 天"},
                {"label": "RUL MAE（退化區）", "value": f"{m['mae_hours']:.1f} h",
                 "sub": f"RMSE {m['rmse_hours']:.1f} h"},
                {"label": "健康指標", "value": meta["indicator"], "sub": "趨勢外推法"},
            ])

            st.plotly_chart(
                health_curve(ts, health, alarm_health=float(alarm), fpt_t=fpt_t),
                width='stretch',
            )

            # --- interaction 2: time-axis scrubber (replay the run) ---
            style.section("時間軸回放")
            i = st.slider("運轉時間點（快照序號）", 0, len(preds) - 1,
                          len(preds) - 1, step=1)
            row = preds.iloc[i]
            rul_txt = f"{row['rul_true']:.1f} h" if pd.notna(row["rul_true"]) else "—"
            pred_txt = f"{row['rul_pred']:.1f} h" if pd.notna(row["rul_pred"]) else "（退化前）"
            cc = st.columns(4)
            with cc[0]:
                style.big_stat("此刻時間", f"{row['timestamp']:%m-%d %H:%M}", "")
            with cc[1]:
                style.big_stat("健康分數", f"{row['health']:.0f}", "100 健康 → 0 失效",
                               tone="danger" if row["health"] <= alarm else "primary")
            with cc[2]:
                style.big_stat("實際 RUL", rul_txt, "距離失效")
            with cc[3]:
                style.big_stat("預測 RUL", pred_txt, "趨勢外推")
            style.note(
                "拖動上方滑桿沿時間回放整個運轉過程：健康分數如何從 100 漸滑到 0，"
                "以及該時刻的實際 / 預測剩餘壽命。"
            )


# ---------------------------------------------------------------------------
# Module B · RUL 預測 (forecast vs actual + method contrast)
# ---------------------------------------------------------------------------
elif page == "RUL 預測":
    ims, ready = _ims_guard()
    if ready:
        preds, meta = _ims_predictions(ims)
        if preds is not None:
            m, near = meta["metrics"], meta["near_failure_metrics"]
            style.kpi_strip([
                {"label": "RUL MAE（退化區）", "value": f"{m['mae_hours']:.1f} h",
                 "sub": f"{m['n_eval']} 點"},
                {"label": "RUL RMSE（退化區）", "value": f"{m['rmse_hours']:.1f} h",
                 "sub": "均方根誤差"},
                {"label": "RUL MAE（近失效）", "value": f"{near['mae_hours']:.1f} h",
                 "sub": f"最後 {near['n_eval']} 筆"},
                {"label": "方法", "value": "趨勢外推", "sub": "指數 + 滾動視窗"},
            ])

            deg = preds[preds["is_degrading"] & preds["rul_pred"].notna()]
            if not deg.empty:
                st.plotly_chart(
                    rul_forecast(deg["timestamp"], deg["rul_true"], deg["rul_pred"]),
                    width='stretch',
                )
            style.note(
                "RUL 以退化趨勢外推估計。本軸承屬<b>突發型失效</b>（指標在最後 ~2% 壽命"
                "才暴衝），因此 RUL 早期偏粗、越接近失效越收斂 —— 這是該失效模式的固有限制，"
                "如實呈現而非過度配適。", kind="warn",
            )
            style.section("方法學對照")
            style.note(
                "初期曾嘗試<b>監督式回歸</b>（RandomForest / GradientBoosting + 時間切分）"
                "直接預測 RUL，結果嚴重失敗（MAE≈120 h、R²≈−76）：單一退化軌跡下，測試段的 "
                "RUL 區間完全落在訓練段之外，而樹模型無法外推單調目標。"
                "故改採趨勢外推法。詳見 <code>docs/MODULE_B_RESULTS.md</code>。"
            )


# ---------------------------------------------------------------------------
# Module B · 互動探索 (switch indicator + raw waveform / spectrum)
# ---------------------------------------------------------------------------
elif page == "互動探索":
    ims, ready = _ims_guard()
    if ready:
        df = _ims_feature_table(str(resolve(ims["processed_features"])))

        # --- interaction 3: switch the health indicator, recompute FPT live ---
        style.section("健康指標切換（即時重算退化起點）")
        candidates = [c for c in ("b1_rms", "b1_kurtosis", "b1_band_BPFO",
                                   "b1_crest_factor") if c in df.columns]
        indicator = st.selectbox("健康指標", candidates,
                                 index=candidates.index(ims["health_indicator"])
                                 if ims["health_indicator"] in candidates else 0)
        hi, health, hi_base, hi_fail = build_health_indicator(
            df[indicator], ims["hi_smooth_window"], ims["baseline_n"],
            ims["fail_percentile"],
        )
        fpt_idx = detect_fpt(hi, ims["baseline_n"], ims["fpt_n_sigma"],
                             ims["fpt_consecutive"])
        fpt_t = df.index[fpt_idx]
        lead_days = (df.index[-1] - fpt_t).total_seconds() / 86400.0
        style.kpi_strip([
            {"label": "指標", "value": indicator, "sub": "切換看 FPT 變化"},
            {"label": "退化起點 FPT", "value": f"{fpt_t:%m-%d %H:%M}",
             "sub": f"第 {fpt_idx} 個快照"},
            {"label": "提前量", "value": f"{lead_days:.1f} 天", "sub": "FPT → 失效"},
        ])
        hdf = pd.DataFrame({"health": health.to_numpy()}, index=df.index)
        st.plotly_chart(
            health_curve(hdf.index, hdf["health"],
                         alarm_health=float(ims["alarm_health"]), fpt_t=fpt_t),
            width='stretch',
        )
        style.note(
            "切換指標即時重算健康曲線與退化起點。可看到 <code>b1_rms</code> 最單調穩定，"
            "峭度 / BPFO 雜訊大或更接近末期才暴衝 —— 這就是選 RMS 當主指標的原因。"
        )

        # --- interaction 4: raw waveform & FFT spectrum of one snapshot ---
        style.section("原始波形與頻譜檢視")
        if not resolve(ims["raw_dir"]).exists():
            style.note(
                "原始波形 / 頻譜需要 1.5 GB 的 IMS 原始資料（未隨專案上傳）。"
                "此功能僅在本地放置 Set 2 後可用；雲端 demo 不提供此項。",
                kind="info",
            )
        else:
            files = list_ims_files(ims["raw_dir"])
            snap = st.slider("快照序號", 0, len(files) - 1, fpt_idx, step=1)
            ts_sel, path_sel = files[snap]
            ch = load_ims_file(path_sel)[:, ims["target_bearing"] - 1]
            st.caption(f"快照時間 {ts_sel:%Y-%m-%d %H:%M:%S}　·　軸承 "
                       f"B{ims['target_bearing']}（{ims['health_indicator']} 退化通道）")
            wc, sc = st.columns(2)
            with wc:
                st.plotly_chart(
                    vibration_waveform(ch, ims["sampling_rate_hz"]),
                    width='stretch',
                )
            with sc:
                st.plotly_chart(
                    vibration_spectrum(ch, ims["sampling_rate_hz"], ims["defect_freqs"]),
                    width='stretch',
                )
            style.note(
                "拖動快照序號比較健康初期 vs 退化末期：退化後 FFT 在 "
                "<b>BPFO（外圈，約 236 Hz）</b>附近的能量明顯增強，這是外圈剝落的物理指紋。"
            )


# ---------------------------------------------------------------------------
# Module B+ · 多軌跡泛化 (XJTU cross-bearing generalization)
# ---------------------------------------------------------------------------
elif page == "多軌跡泛化":
    xj, ready = _xjtu_guard()
    if ready:
        summary, lobo, loco = _xjtu_artifacts(xj)

        n_cond = summary["condition"].nunique()
        style.kpi_strip([
            {"label": "軸承數（獨立軌跡）", "value": str(len(summary)),
             "sub": f"{n_cond} 種工況"},
            {"label": "平均退化提前量", "value": f"{summary['lead_time_hours'].mean():.2f} h",
             "sub": "固定參數"},
            {"label": "平均退化區 MAE", "value": f"{summary['mae_hours'].mean():.2f} h",
             "sub": "趨勢外推"},
            {"label": "LOCO 合併 R²",
             "value": (f"{loco['pooled']['r2']:+.2f}" if loco else "—"),
             "sub": "跨工況監督式"},
        ])

        style.section("固定參數跨工況泛化（步驟 3）")
        per_cond = (summary.groupby("condition", sort=False)
                    .agg(軸承數=("bearing", "count"),
                         平均提前量_h=("lead_time_hours", "mean"),
                         平均MAE_h=("mae_hours", "mean")).reset_index()
                    .rename(columns={"condition": "工況"}))
        per_cond["平均提前量_h"] = per_cond["平均提前量_h"].round(2)
        per_cond["平均MAE_h"] = per_cond["平均MAE_h"].round(2)
        st.dataframe(per_cond, hide_index=True, width="stretch")
        style.note(
            "同一組固定參數（未逐顆、未逐工況調）套到 <b>3 種工況、15 顆獨立軸承</b>，"
            "全數偵測到退化起點。下方為各軸承明細與健康指標疊圖。"
        )
        with st.expander("各軸承明細（15 顆）"):
            disp = summary.rename(columns={
                "condition": "工況", "bearing": "軸承", "n_snapshots": "壽命(快照)",
                "lead_time_hours": "提前量(h)", "mae_hours": "退化區MAE(h)",
            })[["工況", "軸承", "壽命(快照)", "提前量(h)", "退化區MAE(h)"]].copy()
            disp["提前量(h)"] = disp["提前量(h)"].round(2)
            disp["退化區MAE(h)"] = disp["退化區MAE(h)"].round(2)
            st.dataframe(disp, hide_index=True, width="stretch")

        # health-indicator overlay (smoothed h_rms vs % of life, coloured by condition)
        feat_path = resolve(xj["processed_features"])
        if feat_path.exists():
            feat = _xjtu_feature_table(str(feat_path))
            ind, sw = xj["health_indicator"], xj["hi_smooth_window"]
            fpt_by = {(r["condition"], r["bearing"]): r["fpt_index"]
                      for _, r in summary.iterrows()}
            long_rows, fpt_rows = [], []
            for (cond, bearing), g in feat.groupby(["condition", "bearing"], sort=False):
                g = g.sort_values("minute")
                hi = g[ind].rolling(sw, min_periods=1).median().to_numpy()
                n = len(g)
                life_pct = np.linspace(0, 100, n)
                long_rows.append(pd.DataFrame(
                    {"condition": cond, "bearing": bearing, "life_pct": life_pct, "hi": hi}))
                fi = min(int(fpt_by.get((cond, bearing), 0)), n - 1)
                fpt_rows.append({"condition": cond, "bearing": bearing,
                                 "life_pct": life_pct[fi], "hi": hi[fi]})
            st.plotly_chart(
                xjtu_health_overlay(pd.concat(long_rows, ignore_index=True),
                                    pd.DataFrame(fpt_rows)),
                width="stretch",
            )
            style.note(
                "15 顆軸承（3 工況以顏色區分）、<b>同一組固定參數</b>，健康指標 h_rms 末期都"
                "明顯上升、且都偵測到退化起點（★）。這是跨軸承、跨工況的泛化證據。"
            )

        # supervised RUL: LOBO (within-pool) vs LOCO (cross-condition)
        if lobo or loco:
            style.section("監督式 RUL：LOBO vs LOCO（步驟 4）")
            cc = st.columns(2)
            with cc[0]:
                style.big_stat("LOBO 合併 R²",
                               f"{lobo['pooled']['r2']:+.2f}" if lobo else "—",
                               "留一軸承（同工況可入訓練）")
            with cc[1]:
                style.big_stat("LOCO 合併 R²",
                               f"{loco['pooled']['r2']:+.2f}" if loco else "—",
                               "留一工況（測試工況未見過）",
                               tone="danger")
            if loco:
                lr = pd.DataFrame(loco["per_condition"]).rename(columns={
                    "held_out_condition": "留出工況", "n_test_bearings": "測試軸承",
                    "mae_hours": "MAE(h)", "r2": "R²",
                })[["留出工況", "測試軸承", "MAE(h)", "R²"]].copy()
                lr["MAE(h)"] = lr["MAE(h)"].round(2)
                lr["R²"] = lr["R²"].round(2)
                st.dataframe(lr, hide_index=True, width="stretch")
            style.note(
                "<b>LOBO</b>（留一軸承、同工況樣本可進訓練）通常優於 <b>LOCO</b>（留一整個工況、"
                "測試工況的轉速/負載完全沒見過）—— 兩者落差正是<b>跨工況 domain shift</b> 的證據。"
                "誠實結論：監督式 RUL 對運轉條件敏感，跨工況泛化需更多工況或領域自適應；"
                "固定參數的趨勢外推健康監測則跨工況仍穩健（見上方）。",
                kind="warn",
            )


# ---------------------------------------------------------------------------
# Page 5: about
# ---------------------------------------------------------------------------
else:
    style.section("專案總覽")
    overview_img = _ROOT / "docs" / "0620.png"
    if overview_img.exists():
        c_l, c_r = st.columns([2, 1])
        with c_l:
            st.image(str(overview_img), width='stretch',
                     caption="專案總覽 infographic")
        with c_r:
            with style.zone("mint", key="about-positioning"):
                st.markdown(
                    """
                    ##### 系統定位
                    **預測性維護原型**：以目前運轉條件估計故障風險，提供決策輔助。

                    ##### 不是
                    - 即時控制器
                    - 精準 RUL 預測器
                    - 已驗證的工廠系統
                    """
                )
            with style.zone("sky", key="about-stack"):
                st.markdown(
                    """
                    ##### 技術棧
                    - **ML**：scikit-learn · XGBoost · LightGBM
                    - **可解釋**：SHAP · Permutation Importance
                    - **調參**：Optuna
                    - **UI**：Streamlit · Plotly · shadcn-ui
                    - **服務**：FastAPI · Docker · GitHub Actions
                    """
                )
    else:
        style.note("infographic 圖片 (`docs/0620.png`) 不存在。", kind="warn")

    st.divider()
    style.section("模組 A vs B vs B+ 對照")
    ca, cb, cc = st.columns(3)
    with ca:
        with style.zone("mint", key="about-mod-a"):
            st.markdown(
                """
                ##### 🅰 模組 A · 靜態風險評估
                **AI4I 2020** 單筆製程點資料
                → 分類模型（10 × 5）
                → 故障機率 · 健康分數 · 維護建議
                """
            )
    with cb:
        with style.zone("sky", key="about-mod-b"):
            st.markdown(
                """
                ##### 🅱 模組 B · 動態健康度預測
                **IMS 軸承** 20kHz 振動全壽命
                → 時頻特徵 · FPT · 趨勢外推
                → 健康退化曲線 · RUL 剩餘壽命
                """
            )
    with cc:
        with style.zone("sand", key="about-mod-bp"):
            st.markdown(
                """
                ##### 🅱➕ 模組 B+ · 多軌跡泛化
                **XJTU 軸承** 15 顆 / 3 工況
                → 固定參數 FPT · LOBO / LOCO
                → 跨軸承 / 跨工況泛化驗證
                """
            )
    compare_df = pd.DataFrame(
        {
            "🅰 模組 A · 靜態風險": [
                "UCI AI4I 2020（合成）",
                "單筆點資料（製程快照）",
                "溫度 / 扭矩 / 轉速 / 刀具磨耗",
                "故障 / 正常二元分類",
                "監督式分類（10 模型 × 5 特徵組）",
                "F1 / Recall / ROC-AUC / PR-AUC",
                "故障機率 · 健康分數 · 維護建議",
                "無（靜態快照）",
                "單筆 / What-if / 批次 / 評估",
                "合成資料、無 RUL 標籤",
            ],
            "🅱 模組 B · 動態健康度": [
                "NASA/IMS 軸承 Set 2（實測 run-to-failure）",
                "20 kHz 高頻振動時間序列",
                "加速度振動（時域 + 頻域）",
                "健康分數退化 + 剩餘壽命 RUL",
                "健康指標 + FPT + 指數趨勢外推",
                "RUL MAE / RMSE（小時）、退化提前量",
                "退化曲線（100→0）· RUL · 告警提前量",
                "有（全壽命動態演進）",
                "健康度總覽 / RUL 預測 / 互動探索",
                "單一退化軌跡、突發失效 → RUL 偏粗",
            ],
            "🅱➕ 模組 B+ · 多軌跡泛化": [
                "XJTU-SY（實測 run-to-failure）",
                "25.6 kHz 振動 × 15 顆 / 3 工況",
                "加速度振動（水平 / 垂直）",
                "跨軸承 / 跨工況泛化驗證",
                "固定參數 FPT/外推 + LOBO/LOCO 監督式",
                "FPT 提前量、R²（LOBO / LOCO）",
                "多軌跡健康疊圖、泛化 / 遷移結論",
                "有（15 條獨立軌跡）",
                "多軌跡泛化",
                "絕對 RUL 跨壽命尺度 / 工況受限",
            ],
        },
        index=["資料集", "資料型態", "感測量", "目標", "建模方法",
               "評估指標", "主要輸出", "時間維度", "對應頁面", "核心限制"],
    )
    st.table(compare_df)
    style.note(
        "三軌（A 靜態 / B 動態 / B+ 多軌跡泛化）<b>無法合併成同一個模型</b>"
        "（物理量、感測器、目標皆不同），在系統中以平行獨立軌道呈現。B+ 以多軌跡、多工況"
        "驗證「健康監測可泛化、絕對 RUL 受限」。細節見 <code>docs/MODULE_B_RESULTS.md</code> 與 "
        "<code>docs/MODULE_B_PLUS_XJTU_PLAN.md</code>。"
    )

    st.divider()
    style.section("數字一覽")
    style.kpi_strip([
        {"label": "git 追蹤檔案", "value": "123", "sub": "含 LICENSE / Docker / CI"},
        {"label": "比較模型組合", "value": "40", "sub": "model_comparison.csv"},
        {"label": "繪圖函式", "value": "17", "sub": "src/ui/charts.py"},
        {"label": "FastAPI 端點", "value": "7", "sub": "health / predict / metrics 等"},
        {"label": "Streamlit 頁面", "value": "10", "sub": "A 5 + B 3 + B+ 1 + 關於 1"},
        {"label": "單元測試", "value": "18 / 18", "sub": "全部通過"},
    ])
    style.section("外部連結")
    st.markdown(
        """
        - GitHub Repo · <https://github.com/ChenYuHsu413/AIFinalProject>
        - GitHub Actions · <https://github.com/ChenYuHsu413/AIFinalProject/actions>
        - 模組 A 資料集（AI4I 2020）· <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
        - 模組 B 資料集（NASA/IMS Bearing）· <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
        - 模組 B+ 資料集（XJTU-SY Bearing）· <https://biaowang.tech/xjtu-sy-bearing-datasets/>
        - 模型卡 · `outputs/models/MODEL_CARD.md`
        - 報告大綱 · `outputs/reports/REPORT_OUTLINE.md`
        """
    )
