"""Global CSS + small HTML helpers for the Streamlit dashboard.

The intent is to keep the polish layer in one place so the app file stays
focused on logic.  All CSS lives in ``GLOBAL_CSS`` below; callers inject it
once at the top of the script via ``inject()``.
"""
from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Palette — keep in sync with .streamlit/config.toml
# ---------------------------------------------------------------------------
PRIMARY = "#0d9488"        # teal
PRIMARY_DARK = "#0f766e"
ACCENT = "#0369a1"         # deep cyan-blue (used in gradients)
SUCCESS = "#16a34a"
WARNING = "#f59e0b"
DANGER = "#dc2626"
INK = "#0f172a"
MUTED = "#64748b"
PANEL = "#ffffff"
PAGE = "#f8fafc"
BORDER = "#e5e7eb"


GLOBAL_CSS = f"""
<style>
/* ---- Reset some default Streamlit chrome ---- */
header[data-testid="stHeader"] {{
    background: rgba(255, 255, 255, 0);
    height: 0;
}}
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-weight: 500;
}}

/* ---- Hero banner ---- */
.hero-banner {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(13, 148, 136, 0.18);
}}
.hero-banner .hero-eyebrow {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    opacity: 0.85;
    margin-bottom: 6px;
}}
.hero-banner h1 {{
    margin: 0 0 8px 0;
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.25;
}}
.hero-banner p {{
    margin: 0;
    opacity: 0.92;
    font-size: 0.96rem;
    max-width: 760px;
}}

/* ---- Section header (left accent bar) ---- */
.section-h {{
    border-left: 4px solid {PRIMARY};
    padding-left: 12px;
    margin: 22px 0 12px 0;
    font-size: 1.12rem;
    font-weight: 600;
    color: {INK};
}}

/* ---- Risk pill ---- */
.risk-pill {{
    display: inline-block;
    padding: 12px 22px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.05rem;
    text-align: center;
    color: white;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    min-width: 160px;
}}
.risk-low {{ background: linear-gradient(135deg, #10b981, {SUCCESS}); }}
.risk-medium {{ background: linear-gradient(135deg, {WARNING}, #d97706); }}
.risk-high {{ background: linear-gradient(135deg, #ef4444, {DANGER}); }}

/* ---- Soft note box ---- */
.note-box {{
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid {ACCENT};
    padding: 12px 16px;
    border-radius: 8px;
    color: #075985;
    font-size: 0.92rem;
    margin: 10px 0;
}}
.note-box.warn {{
    background: #fffbeb;
    border-color: #fde68a;
    border-left-color: {WARNING};
    color: #92400e;
}}
.note-box.danger {{
    background: #fef2f2;
    border-color: #fecaca;
    border-left-color: {DANGER};
    color: #991b1b;
}}

/* ---- Fallback metric card (when shadcn-ui not used) ---- */
.metric-card {{
    background: {PANEL};
    padding: 16px 18px;
    border-radius: 12px;
    border: 1px solid {BORDER};
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    height: 100%;
}}
.metric-card .label {{
    color: {MUTED};
    font-size: 0.82rem;
    margin-bottom: 6px;
}}
.metric-card .value {{
    color: {INK};
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.1;
}}
.metric-card .delta {{
    color: {MUTED};
    font-size: 0.78rem;
    margin-top: 4px;
}}

/* ---- Polish Streamlit's own widgets ---- */
.stButton > button {{
    border-radius: 8px;
    font-weight: 600;
    border: none;
    padding: 8px 18px;
    transition: transform 0.05s ease-in;
}}
.stButton > button:active {{ transform: translateY(1px); }}
div[data-testid="stExpander"] {{
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
}}
div[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
}}
</style>
"""


def inject() -> None:
    """Inject the global CSS once at the top of the Streamlit script."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable HTML widgets (server-rendered markdown blocks)
# ---------------------------------------------------------------------------
def hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-h">{title}</div>', unsafe_allow_html=True)


def risk_pill(risk: str, label_zh: str) -> None:
    cls = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(
        risk, "risk-low"
    )
    st.markdown(
        f'<div class="risk-pill {cls}">風險等級：{label_zh}（{risk}）</div>',
        unsafe_allow_html=True,
    )


def note(body: str, kind: str = "info") -> None:
    cls = {"info": "", "warn": "warn", "danger": "danger"}.get(kind, "")
    st.markdown(f'<div class="note-box {cls}">{body}</div>', unsafe_allow_html=True)


def fallback_metric_card(label: str, value: str, delta: str | None = None) -> None:
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
