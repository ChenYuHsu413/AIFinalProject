"""Global CSS + small HTML helpers for the Streamlit dashboard.

The intent is to keep the polish layer in one place so the app file stays
focused on logic.  All CSS lives in ``GLOBAL_CSS`` below; callers inject it
once at the top of the script via ``inject()``.
"""
from __future__ import annotations

import textwrap

import streamlit as st


def _render(html: str) -> None:
    """``st.markdown`` with safe handling of multi-line indented HTML.

    Streamlit's Markdown parser treats lines indented by 4+ spaces as code
    blocks, so triple-quoted HTML written for code readability gets escaped
    and shown to the user as literal text.  We dedent and strip first.
    """
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


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

/* ---- Keyframes ---- */
@keyframes heroShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulseRing {{
    0%   {{ box-shadow: 0 0 0 0 rgba(13, 148, 136, 0.45); }}
    70%  {{ box-shadow: 0 0 0 14px rgba(13, 148, 136, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(13, 148, 136, 0); }}
}}

/* ---- Hero banner (animated gradient) ---- */
.hero-banner {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 50%, #1e3a8a 100%);
    background-size: 200% 200%;
    animation: heroShift 14s ease infinite, fadeUp 0.5s ease-out;
    color: white;
    padding: 30px 34px;
    border-radius: 18px;
    margin-bottom: 22px;
    box-shadow: 0 14px 36px rgba(13, 148, 136, 0.22);
    position: relative;
    overflow: hidden;
}}
.hero-banner::after {{
    content: "";
    position: absolute;
    top: -50%; right: -10%;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 60%);
    pointer-events: none;
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
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.25;
}}
.hero-banner p {{
    margin: 0;
    opacity: 0.92;
    font-size: 0.96rem;
    max-width: 760px;
}}
.hero-banner .hero-chips {{
    margin-top: 14px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}}
.hero-banner .hero-chips span {{
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    color: white;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    backdrop-filter: blur(4px);
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
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 10px 22px;
    transition: transform 0.05s ease-in, box-shadow 0.15s ease-out;
}}
.stButton > button:hover {{
    box-shadow: 0 6px 16px rgba(13, 148, 136, 0.25);
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
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0;
    padding: 8px 18px;
    background: #f1f5f9;
    border: 1px solid transparent;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(180deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%) !important;
    color: white !important;
    border-color: transparent !important;
}}

/* ---- Coloured zone panels for visual rhythm ---- */
.zone {{
    padding: 18px 22px;
    border-radius: 14px;
    margin: 14px 0;
    animation: fadeUp 0.4s ease-out;
}}
.zone-mint   {{ background: linear-gradient(135deg, #ecfdf5, #f0fdfa); border: 1px solid #d1fae5; }}
.zone-sand   {{ background: linear-gradient(135deg, #fffbeb, #fef9c3); border: 1px solid #fde68a; }}
.zone-sky    {{ background: linear-gradient(135deg, #eff6ff, #f0f9ff); border: 1px solid #bfdbfe; }}
.zone-blush  {{ background: linear-gradient(135deg, #fef2f2, #fff1f2); border: 1px solid #fecaca; }}
.zone-stone  {{ background: #f1f5f9; border: 1px solid #e2e8f0; }}

/* ---- Big animated stat ---- */
.big-stat {{
    text-align: center;
    padding: 22px 14px;
    background: white;
    border-radius: 14px;
    border: 1px solid {BORDER};
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    animation: fadeUp 0.5s ease-out;
}}
.big-stat .label {{
    color: {MUTED};
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}}
.big-stat .value {{
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    margin: 8px 0 4px;
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}}
.big-stat .sub {{
    color: {MUTED};
    font-size: 0.78rem;
}}
.big-stat .value.danger {{ background: linear-gradient(135deg, #ef4444, {DANGER}); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.big-stat .value.warn   {{ background: linear-gradient(135deg, {WARNING}, #d97706); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.big-stat .value.good   {{ background: linear-gradient(135deg, #10b981, {SUCCESS}); -webkit-background-clip: text; background-clip: text; color: transparent; }}

/* ---- Result card with subtle pulse around the icon ---- */
.result-card {{
    background: white;
    border-radius: 14px;
    border: 1px solid {BORDER};
    padding: 20px 24px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    animation: fadeUp 0.4s ease-out;
}}

/* ---- Streamlit tabs panel: small fade-in ---- */
div[role="tabpanel"] {{
    animation: fadeUp 0.35s ease-out;
}}

/* ---- Metric card with progress bar ---- */
.metric-bar {{
    background: white;
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 16px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    animation: fadeUp 0.45s ease-out;
    position: relative;
    overflow: hidden;
}}
.metric-bar::before {{
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    width: 4px;
    background: linear-gradient(180deg, {PRIMARY}, {ACCENT});
}}
.metric-bar .mb-label {{
    color: {MUTED};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin-left: 6px;
}}
.metric-bar .mb-row {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 6px 0 8px 6px;
}}
.metric-bar .mb-value {{
    color: {INK};
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
}}
.metric-bar .mb-sub {{
    color: {MUTED};
    font-size: 0.7rem;
}}
.metric-bar .mb-track {{
    background: #f1f5f9;
    height: 6px;
    border-radius: 4px;
    overflow: hidden;
    margin-left: 6px;
}}
.metric-bar .mb-fill {{
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
    animation: barGrow 0.7s cubic-bezier(.21,.71,.34,1.01);
    transform-origin: left;
}}
.metric-bar .mb-fill.warn   {{ background: linear-gradient(90deg, {WARNING}, #d97706); }}
.metric-bar .mb-fill.danger {{ background: linear-gradient(90deg, #ef4444, {DANGER}); }}
.metric-bar .mb-fill.good   {{ background: linear-gradient(90deg, #10b981, {SUCCESS}); }}
@keyframes barGrow {{
    from {{ transform: scaleX(0); }}
    to   {{ transform: scaleX(1); }}
}}
.metric-bar.with-spark .mb-spark {{
    margin-top: -2px;
    margin-left: -10px;
    margin-right: -10px;
}}

/* ---- Advice card with category icon ---- */
.advice-card {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: white;
    border: 1px solid {BORDER};
    border-left: 4px solid {PRIMARY};
    border-radius: 12px;
    padding: 14px 16px;
    margin: 8px 0;
    animation: fadeUp 0.4s ease-out;
    transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
}}
.advice-card:hover {{
    transform: translateX(2px);
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
}}
.advice-card.thermal    {{ border-left-color: #f97316; }}
.advice-card.load       {{ border-left-color: #6366f1; }}
.advice-card.wear       {{ border-left-color: #a855f7; }}
.advice-card.speed      {{ border-left-color: #06b6d4; }}
.advice-card.alert      {{ border-left-color: {DANGER}; background: #fef2f2; }}
.advice-card.healthy    {{ border-left-color: {SUCCESS}; background: #f0fdf4; }}
.advice-card .ac-icon {{
    font-size: 1.6rem;
    line-height: 1;
    flex-shrink: 0;
    background: #f1f5f9;
    width: 42px; height: 42px;
    border-radius: 10px;
    display: grid; place-items: center;
}}
.advice-card.thermal .ac-icon {{ background: #fff7ed; }}
.advice-card.load    .ac-icon {{ background: #eef2ff; }}
.advice-card.wear    .ac-icon {{ background: #faf5ff; }}
.advice-card.speed   .ac-icon {{ background: #ecfeff; }}
.advice-card.alert   .ac-icon {{ background: #fee2e2; }}
.advice-card.healthy .ac-icon {{ background: #dcfce7; }}
.advice-card .ac-body {{
    color: {INK};
    font-size: 0.93rem;
    line-height: 1.55;
    padding-top: 6px;
}}
.advice-card .ac-tag {{
    display: inline-block;
    background: #f1f5f9;
    color: {MUTED};
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 2px 8px;
    border-radius: 999px;
    margin-bottom: 4px;
    margin-right: 6px;
}}

/* ---- Action bar ---- */
.action-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 10px 0 18px;
}}
.action-bar a, .action-bar .ab-link {{
    background: white;
    color: {PRIMARY_DARK};
    border: 1px solid {BORDER};
    padding: 8px 14px;
    border-radius: 10px;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.86rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: all 0.15s ease-out;
}}
.action-bar a:hover, .action-bar .ab-link:hover {{
    border-color: {PRIMARY};
    box-shadow: 0 6px 16px rgba(13, 148, 136, 0.18);
    transform: translateY(-1px);
}}
.action-bar a.primary {{
    background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_DARK});
    color: white !important;
    border-color: transparent;
}}

/* ---- Dashboard tiles ---- */
.dash-tile {{
    background: white;
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    animation: fadeUp 0.4s ease-out;
    transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
}}
.dash-tile:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}}
.dash-tile .dt-icon {{
    width: 40px; height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
    color: white;
    display: grid; place-items: center;
    font-size: 1.3rem;
    margin-bottom: 10px;
}}
.dash-tile .dt-title {{
    color: {INK};
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 4px;
}}
.dash-tile .dt-sub {{
    color: {MUTED};
    font-size: 0.84rem;
    line-height: 1.5;
}}

/* ---- Sidebar polish ---- */
section[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0.5rem;
}}
/* Sidebar nav buttons: full-width, left-aligned pills driven by active_page */
section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    padding: 9px 14px;
    margin: 1px 0;
    font-weight: 500;
}}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
    background: transparent;
    color: #334155;
}}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
    background: #f0fdfa;
    box-shadow: none;
}}
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
    background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
    border-radius: 14px;
    color: white;
    box-shadow: 0 6px 18px rgba(13, 148, 136, 0.22);
}}
.sidebar-brand .brand-mark {{
    font-size: 1.7rem;
    line-height: 1;
}}
.sidebar-brand .brand-title {{
    font-weight: 700;
    font-size: 0.92rem;
    line-height: 1.15;
    letter-spacing: 0.2px;
}}
.sidebar-brand .brand-sub {{
    font-size: 0.74rem;
    opacity: 0.88;
    margin-top: 2px;
}}

.sidebar-card {{
    background: linear-gradient(180deg, white, #f8fafc);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 12px 14px;
    margin: 6px 0;
    animation: fadeUp 0.4s ease-out;
}}
.sidebar-card .sc-eyebrow {{
    color: {MUTED};
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}}
.sidebar-card .sc-title {{
    color: {INK};
    font-weight: 600;
    font-size: 0.92rem;
    line-height: 1.2;
}}
.sidebar-card .sc-sub {{
    color: {MUTED};
    font-size: 0.78rem;
    margin-top: 2px;
}}
.sidebar-card .sc-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
}}
.sidebar-card .sc-row .sc-mini-label {{
    color: {MUTED};
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
.sidebar-card .sc-row .sc-mini-value {{
    color: {PRIMARY_DARK};
    font-weight: 700;
    font-size: 0.96rem;
}}

.sidebar-footer {{
    margin-top: 14px;
    padding: 10px 12px;
    color: {MUTED};
    font-size: 0.72rem;
    line-height: 1.5;
    border-top: 1px dashed {BORDER};
}}
.sidebar-footer a {{
    color: {PRIMARY_DARK};
    text-decoration: none;
    font-weight: 600;
}}
.sidebar-footer a:hover {{ text-decoration: underline; }}
.sidebar-footer .sf-pill {{
    display: inline-block;
    background: #f0fdfa;
    color: {PRIMARY_DARK};
    border: 1px solid #99f6e4;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.68rem;
    margin-right: 4px;
}}

/* ---- KPI strip cards ---- */
.kpi-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 4px 0 14px;
}}
.kpi-strip .kpi {{
    background: white;
    border-radius: 12px;
    border: 1px solid {BORDER};
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    animation: fadeUp 0.45s ease-out;
}}
.kpi .label {{ color: {MUTED}; font-size: 0.78rem; }}
.kpi .value {{ color: {INK}; font-size: 1.55rem; font-weight: 700; line-height: 1; margin: 4px 0; }}
.kpi .sub {{ color: {MUTED}; font-size: 0.74rem; }}
</style>
"""


def inject() -> None:
    """Inject the global CSS once at the top of the Streamlit script."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable HTML widgets (server-rendered markdown blocks)
# ---------------------------------------------------------------------------
def hero(eyebrow: str, title: str, subtitle: str,
         chips: list[str] | None = None) -> None:
    chips_html = ""
    if chips:
        items = "".join(f"<span>{c}</span>" for c in chips)
        chips_html = f'<div class="hero-chips">{items}</div>'
    _render(f"""
        <div class="hero-banner">
            <div class="hero-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {chips_html}
        </div>
    """)


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
    _render(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
    """)


# ---------------------------------------------------------------------------
# Higher-impact visual primitives
# ---------------------------------------------------------------------------
_ZONE_STYLES = {
    "mint":  "background: linear-gradient(135deg, #ecfdf5, #f0fdfa); border: 1px solid #d1fae5;",
    "sand":  "background: linear-gradient(135deg, #fffbeb, #fef9c3); border: 1px solid #fde68a;",
    "sky":   "background: linear-gradient(135deg, #eff6ff, #f0f9ff); border: 1px solid #bfdbfe;",
    "blush": "background: linear-gradient(135deg, #fef2f2, #fff1f2); border: 1px solid #fecaca;",
    "stone": "background: #f1f5f9; border: 1px solid #e2e8f0;",
}


def _css_container(key: str, css: str):
    """Native replacement for streamlit-extras' deprecated ``stylable_container``.

    Streamlit tags a keyed container's DOM node with the class
    ``st-key-{key}``, so we inject a scoped ``<style>`` block and hand back the
    container.  CSS selectors in ``css`` must already be qualified with
    ``.st-key-{key}``.
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    return st.container(key=key)


def zone(kind: str = "stone", key: str | None = None):
    """Context manager: render a coloured panel that actually wraps content.

    Usage::

        with style.zone("sky", key="form-zone"):
            st.form(...)
    """
    k = key or f"zone-{kind}"
    style_rules = _ZONE_STYLES.get(kind, _ZONE_STYLES["stone"])
    css = (
        f".st-key-{k} {{ {style_rules} padding: 18px 22px; border-radius: 14px;"
        " margin: 8px 0; animation: fadeUp 0.4s ease-out; }}"
    )
    return _css_container(k, css)


def big_stat(label: str, value: str, sub: str = "", tone: str = "primary") -> None:
    """Hero-sized gradient stat block."""
    cls_value = {
        "primary": "",
        "danger": "danger",
        "warn": "warn",
        "good": "good",
    }.get(tone, "")
    _render(f"""
        <div class="big-stat">
            <div class="label">{label}</div>
            <div class="value {cls_value}">{value}</div>
            <div class="sub">{sub}</div>
        </div>
    """)


def kpi_strip(items: list[dict]) -> None:
    """Render a responsive KPI grid from a list of dicts.

    Each item supports: ``label``, ``value``, ``sub``.
    """
    # Single-line cells so no Markdown code-block fallback can trip on them.
    cells = "".join(
        f'<div class="kpi"><div class="label">{it.get("label", "")}</div>'
        f'<div class="value">{it.get("value", "")}</div>'
        f'<div class="sub">{it.get("sub", "")}</div></div>'
        for it in items
    )
    _render(f'<div class="kpi-strip">{cells}</div>')


# ---------------------------------------------------------------------------
# Sidebar helpers (called via ``with st.sidebar:`` or st.sidebar.markdown)
# ---------------------------------------------------------------------------
def sidebar_brand(emoji: str, title: str, subtitle: str) -> None:
    html = (
        '<div class="sidebar-brand">'
        f'<div class="brand-mark">{emoji}</div>'
        '<div class="brand-text">'
        f'<div class="brand-title">{title}</div>'
        f'<div class="brand-sub">{subtitle}</div>'
        '</div></div>'
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


def sidebar_model_card(
    model_name: str, feature_set: str,
    f1: float, recall: float,
) -> None:
    html = (
        '<div class="sidebar-card">'
        '<div class="sc-eyebrow">Active model</div>'
        f'<div class="sc-title">{model_name}</div>'
        f'<div class="sc-sub">特徵組合 · {feature_set}</div>'
        '<div class="sc-row">'
        f'<div><div class="sc-mini-label">F1</div>'
        f'<div class="sc-mini-value">{f1:.3f}</div></div>'
        f'<div><div class="sc-mini-label">Recall</div>'
        f'<div class="sc-mini-value">{recall:.3f}</div></div>'
        '</div></div>'
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


def sidebar_dataset_card(name: str, note: str) -> None:
    html = (
        '<div class="sidebar-card">'
        '<div class="sc-eyebrow">Dataset</div>'
        f'<div class="sc-title">{name}</div>'
        f'<div class="sc-sub">{note}</div>'
        '</div>'
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


def sidebar_footer(html_inner: str) -> None:
    st.sidebar.markdown(
        f'<div class="sidebar-footer">{html_inner}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Richer primitives for Phase A
# ---------------------------------------------------------------------------
def metric_with_bar(label: str, value: str, ratio: float,
                    sub: str = "", tone: str = "primary") -> None:
    """Metric card with an animated progress bar showing ``ratio`` (0–1)."""
    cls_fill = {"primary": "", "warn": "warn",
                "danger": "danger", "good": "good"}.get(tone, "")
    pct = max(0.0, min(1.0, float(ratio))) * 100
    _render(f"""
        <div class="metric-bar">
            <div class="mb-label">{label}</div>
            <div class="mb-row">
                <div class="mb-value">{value}</div>
                <div class="mb-sub">{sub}</div>
            </div>
            <div class="mb-track">
                <div class="mb-fill {cls_fill}" style="width:{pct:.1f}%;"></div>
            </div>
        </div>
    """)


ADVICE_CATEGORIES = [
    (("散熱", "溫差", "通風"), "🌡️", "thermal", "Thermal"),
    (("扭矩", "負載", "夾持"), "⚙️", "load", "Load"),
    (("刀具", "磨耗", "換刀"), "🔧", "wear", "Wear"),
    (("轉速", "驅動"), "💨", "speed", "Speed"),
    (("立即", "通報", "高",), "🚨", "alert", "Alert"),
    (("健康", "例行", "監控"), "✅", "healthy", "Healthy"),
]


def _classify_advice(text: str) -> tuple[str, str, str]:
    for keys, icon, css, label in ADVICE_CATEGORIES:
        if any(k in text for k in keys):
            return icon, css, label
    return "💡", "", "Note"


def advice_card(text: str) -> None:
    icon, css, label = _classify_advice(text)
    _render(f"""
        <div class="advice-card {css}">
            <div class="ac-icon">{icon}</div>
            <div class="ac-body">
                <span class="ac-tag">{label}</span><br>{text}
            </div>
        </div>
    """)


def action_bar(links: list[dict]) -> None:
    """Render a horizontal row of action buttons / links.

    Each item: ``{label, url|onclick, primary, icon}``.
    """
    items = []
    for it in links:
        url = it.get("url", "#")
        primary = " primary" if it.get("primary") else ""
        icon = it.get("icon", "")
        items.append(
            f'<a class="ab-link{primary}" href="{url}" target="_blank">'
            f'{icon} {it.get("label", "")}</a>'
        )
    _render(f'<div class="action-bar">{"".join(items)}</div>')


def dash_tile(icon: str, title: str, sub: str) -> None:
    _render(f"""
        <div class="dash-tile">
            <div class="dt-icon">{icon}</div>
            <div class="dt-title">{title}</div>
            <div class="dt-sub">{sub}</div>
        </div>
    """)


def dash_button_tile(icon: str, title: str, sub: str, key: str) -> bool:
    """Tile-shaped clickable button. Returns True when clicked.

    Renders the icon + title + sub inside the button label (Streamlit wraps
    long labels) and styles the whole button to look like the static
    ``dash_tile``.  The CSS targets the button inside a keyed container.
    """
    k = f"tile-{key}"
    css = f"""
        .st-key-{k} button {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 18px 20px;
            text-align: left;
            min-height: 130px;
            color: {INK};
            font-weight: 500;
            font-size: 0.92rem;
            line-height: 1.55;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: all 0.15s ease-out;
            white-space: normal !important;
        }}
        .st-key-{k} button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.10);
            border-color: {PRIMARY};
        }}
        .st-key-{k} button:focus:not(:active) {{
            border-color: {PRIMARY};
            box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.18);
        }}
        .st-key-{k} button p {{
            margin: 0 !important;
            line-height: 1.55;
        }}
    """
    label = f"{icon}  {title}\n\n{sub}"
    with _css_container(k, css):
        return st.button(label, key=key, width='stretch')
