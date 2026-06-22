"""Reusable Plotly chart builders for the Streamlit dashboard.

All builders return a ``plotly.graph_objects.Figure`` so the caller can
``st.plotly_chart(fig, width='stretch')`` without re-decorating
each chart with the same layout boilerplate.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.ui.style import (
    ACCENT,
    DANGER,
    INK,
    MUTED,
    PRIMARY,
    SUCCESS,
    WARNING,
)


_LAYOUT_DEFAULTS = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
              color=INK, size=13),
    margin=dict(l=10, r=10, t=50, b=30),
)


def _style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(**_LAYOUT_DEFAULTS, height=height,
                      title=dict(text=title or "", x=0.01, y=0.95,
                                 font=dict(size=15, color=INK)))
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# SHAP horizontal bar
# ---------------------------------------------------------------------------
def shap_bar(df_sv: pd.DataFrame) -> go.Figure:
    """``df_sv`` must have columns: feature, value, shap (top-N already sorted)."""
    colors = [DANGER if v > 0 else SUCCESS for v in df_sv["shap"]]
    fig = go.Figure(
        go.Bar(
            x=df_sv["shap"],
            y=df_sv["feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="white", width=1)),
            text=[f"{v:+.2f}" for v in df_sv["shap"]],
            textposition="outside",
            customdata=df_sv["value"],
            hovertemplate="<b>%{y}</b><br>特徵值 = %{customdata:.3g}"
                          "<br>SHAP 貢獻 = %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=INK, opacity=0.6)
    fig = _style(fig, height=max(360, 36 * len(df_sv) + 80),
                 title="Top 特徵對故障 log-odds 的貢獻")
    fig.update_xaxes(title_text="SHAP 貢獻（log-odds 空間）")
    fig.update_yaxes(title_text="")
    return fig


# ---------------------------------------------------------------------------
# Failure-type probability bar (vertical)
# ---------------------------------------------------------------------------
def failure_type_bar(probs: dict, likely: List[str], threshold: float = 0.3) -> go.Figure:
    order = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    values = [probs.get(k, 0.0) for k in order]
    colors = [DANGER if k in likely else "#94a3b8" for k in order]
    labels_zh = {
        "TWF": "刀具磨耗", "HDF": "散熱", "PWF": "電源",
        "OSF": "過載", "RNF": "隨機",
    }
    fig = go.Figure(
        go.Bar(
            x=order, y=values,
            marker=dict(color=colors, line=dict(color="white", width=2)),
            text=[f"{p:.1%}" for p in values],
            textposition="outside",
            customdata=[labels_zh[k] for k in order],
            hovertemplate="<b>%{x}</b> (%{customdata})<br>機率 = %{y:.2%}<extra></extra>",
        )
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color=WARNING,
                  annotation_text=f"顯著門檻 {threshold:.2f}",
                  annotation_position="top right",
                  annotation_font_color=WARNING)
    fig = _style(fig, height=340, title="第二階段：各故障類型機率")
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%", title_text="機率")
    fig.update_xaxes(title_text="")
    return fig


# ---------------------------------------------------------------------------
# 1D sweep line
# ---------------------------------------------------------------------------
def one_d_sweep(xs: Sequence[float], ys: Sequence[float], feature: str,
                current_x: float, threshold: float = 0.5) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(xs), y=list(ys), mode="lines+markers",
            line=dict(color=PRIMARY, width=3),
            marker=dict(size=5, color=PRIMARY),
            hovertemplate=f"{feature} = %{{x:.2f}}<br>機率 = %{{y:.2%}}<extra></extra>",
            name="故障機率",
        )
    )
    fig.add_vline(x=current_x, line_dash="dash", line_color=WARNING,
                  annotation_text="目前值", annotation_font_color=WARNING)
    fig.add_hline(y=threshold, line_dash="dot", line_color=MUTED,
                  annotation_text=f"決策門檻 {threshold:.2f}",
                  annotation_position="bottom right",
                  annotation_font_color=MUTED)
    fig = _style(fig, height=360,
                 title=f"其他條件固定，{feature} 變動時的故障機率")
    fig.update_xaxes(title_text=feature)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", title_text="故障機率")
    return fig


# ---------------------------------------------------------------------------
# 2D risk landscape heatmap
# ---------------------------------------------------------------------------
def risk_landscape(xs: Sequence[float], ys: Sequence[float], Z: np.ndarray,
                   feat_x: str, feat_y: str,
                   current_x: float, current_y: float) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=Z, x=list(xs), y=list(ys),
            colorscale="RdYlGn_r", zmin=0, zmax=1,
            colorbar=dict(title=dict(text="故障機率"), tickformat=".0%"),
            hovertemplate=(f"{feat_x} = %{{x:.2f}}<br>"
                           f"{feat_y} = %{{y:.2f}}<br>"
                           "機率 = %{z:.2%}<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_x], y=[current_y], mode="markers",
            marker=dict(symbol="circle", size=18,
                        color="white", line=dict(color=INK, width=2)),
            name="目前運轉點",
            hovertemplate="目前運轉點<extra></extra>",
        )
    )
    fig = _style(fig, height=460, title="2D 風險地景（其他特徵固定）")
    fig.update_xaxes(title_text=feat_x)
    fig.update_yaxes(title_text=feat_y)
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Failure-probability gauge (Plotly Indicator)
# ---------------------------------------------------------------------------
def failure_probability_gauge(prob: float) -> go.Figure:
    """Big circular gauge — visual showpiece for the headline number."""
    pct = float(prob) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 44, "color": INK}},
            title={"text": "<b>故障機率</b>", "font": {"size": 16, "color": MUTED}},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                          "tickcolor": MUTED, "tickfont": {"size": 11}},
                "bar": {"color": PRIMARY, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},   # green zone
                    {"range": [30, 70], "color": "#fef3c7"},  # amber zone
                    {"range": [70, 100], "color": "#fee2e2"}, # red zone
                ],
                "threshold": {
                    "line": {"color": DANGER, "width": 3},
                    "thickness": 0.85,
                    "value": 70,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=18, r=18, t=40, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=INK),
    )
    return fig


# ---------------------------------------------------------------------------
# Input radar (運轉指紋)
# ---------------------------------------------------------------------------
_INPUT_BOUNDS = {
    "Air temperature [K]":      (295.0, 305.0, "Air T"),
    "Process temperature [K]":  (305.0, 315.0, "Proc T"),
    "Rotational speed [rpm]":   (1100.0, 2900.0, "RPM"),
    "Torque [Nm]":              (0.0, 80.0, "Torque"),
    "Tool wear [min]":          (0.0, 260.0, "Wear"),
}


def input_radar(record: dict, prob: float | None = None) -> go.Figure:
    """Polar chart of the five raw inputs, normalised to [0, 100]."""
    short_labels: List[str] = []
    norms: List[float] = []
    raw_vals: List[float] = []
    for col, (lo, hi, short) in _INPUT_BOUNDS.items():
        raw = float(record[col])
        n = (raw - lo) / (hi - lo) if hi > lo else 0.0
        norms.append(max(0.0, min(1.0, n)) * 100.0)
        raw_vals.append(raw)
        short_labels.append(short)
    # close the polygon
    norms.append(norms[0])
    short_labels.append(short_labels[0])
    raw_vals.append(raw_vals[0])

    # Color the fingerprint by risk band if probability is supplied
    if prob is None:
        line_color = PRIMARY
        fill_color = "rgba(13, 148, 136, 0.30)"
    elif prob >= 0.7:
        line_color = DANGER
        fill_color = "rgba(220, 38, 38, 0.30)"
    elif prob >= 0.3:
        line_color = WARNING
        fill_color = "rgba(245, 158, 11, 0.32)"
    else:
        line_color = SUCCESS
        fill_color = "rgba(22, 163, 74, 0.30)"

    fig = go.Figure()
    # neutral reference ring at 50%
    fig.add_trace(
        go.Scatterpolar(
            r=[50] * len(short_labels),
            theta=short_labels,
            line=dict(color="#cbd5e1", width=1, dash="dot"),
            fill="none",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=norms,
            theta=short_labels,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=3),
            marker=dict(size=10, color=line_color,
                        line=dict(color="white", width=2)),
            customdata=raw_vals,
            hovertemplate="<b>%{theta}</b><br>原始值=%{customdata:.2f}"
                          "<br>佔範圍 %{r:.0f}%<extra></extra>",
            name="目前運轉指紋",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="#f8fafc",
            radialaxis=dict(visible=True, range=[0, 100],
                            showticklabels=False, gridcolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=13, color=INK),
                             gridcolor="#e2e8f0"),
        ),
        title=dict(text="<b>運轉條件指紋</b>（依參考範圍正規化）",
                   x=0.5, y=0.97, xanchor="center",
                   font=dict(size=15, color=INK)),
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=60, b=30),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=INK),
    )
    return fig


# ---------------------------------------------------------------------------
# Batch summary: risk donut
# ---------------------------------------------------------------------------
def risk_donut(n_low: int, n_med: int, n_high: int) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Low (低)", "Medium (中)", "High (高)"],
            values=[n_low, n_med, n_high],
            hole=0.62,
            sort=False,
            direction="clockwise",
            marker=dict(
                colors=[SUCCESS, WARNING, DANGER],
                line=dict(color="white", width=3),
            ),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value} 筆<br>"
                          "%{percent}<extra></extra>",
        )
    )
    total = n_low + n_med + n_high
    fig.update_layout(
        annotations=[
            dict(text=f"<b>{total}</b><br><span style='font-size:11px;color:#64748b;'>批次筆數</span>",
                 x=0.5, y=0.5, showarrow=False,
                 font=dict(size=22, color=INK), align="center"),
        ],
        showlegend=False,
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Batch summary: probability histogram
# ---------------------------------------------------------------------------
def probability_histogram(probs) -> go.Figure:
    probs = list(probs)
    # Colour each bar by risk band: use bin edges to assign band
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=probs,
            nbinsx=30,
            marker=dict(
                color=probs,
                colorscale=[
                    [0.0, SUCCESS],
                    [0.3, "#bbf7d0"],
                    [0.5, WARNING],
                    [0.7, "#fda4af"],
                    [1.0, DANGER],
                ],
                showscale=False,
                line=dict(color="white", width=1),
            ),
            hovertemplate="機率區間 %{x}<br>筆數 %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0.3, line_dash="dash", line_color=WARNING, line_width=1.5,
                  annotation_text="中風險 0.3",
                  annotation_position="top", annotation_font_color=WARNING)
    fig.add_vline(x=0.7, line_dash="dash", line_color=DANGER, line_width=1.5,
                  annotation_text="高風險 0.7",
                  annotation_position="top", annotation_font_color=DANGER)
    fig = _style(fig, height=340, title="<b>故障機率分布</b>")
    fig.update_xaxes(title_text="故障機率", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title_text="筆數")
    return fig


# ---------------------------------------------------------------------------
# Model leaderboard horizontal bar
# ---------------------------------------------------------------------------
def leaderboard_bar(df: pd.DataFrame, metric: str = "f1",
                    top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of the top-N (model, feature_set) runs by ``metric``."""
    d = (
        df.sort_values(metric, ascending=False)
        .head(top_n)
        .iloc[::-1]
        .reset_index(drop=True)
    )
    labels = [
        f"{r['model_name']}<br><span style='color:#94a3b8;font-size:10px;'>"
        f"{r['feature_set']}</span>"
        for _, r in d.iterrows()
    ]
    # Highlight top-1 in primary color, rest in muted
    max_val = d[metric].max()
    colors = [PRIMARY if v == max_val else "#a7f3d0" for v in d[metric]]
    fig = go.Figure(
        go.Bar(
            x=d[metric], y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="white", width=1)),
            text=[f"{v:.3f}" for v in d[metric]],
            textposition="outside",
            textfont=dict(size=11),
            customdata=d[["recall", "precision", "pr_auc"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{metric} = %{{x:.3f}}<br>"
                "recall = %{customdata[0]:.3f}<br>"
                "precision = %{customdata[1]:.3f}<br>"
                "pr_auc = %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    fig = _style(fig, height=max(360, 32 * len(d) + 80),
                 title=f"<b>排行榜 · 依 {metric}</b>（前 {len(d)} 名）")
    fig.update_xaxes(title_text=metric, range=[0, 1.0])
    fig.update_yaxes(title_text="")
    return fig


# ---------------------------------------------------------------------------
# Sparkline (tiny inline trend chart, no axes)
# ---------------------------------------------------------------------------
def sparkline(values, color: str = PRIMARY, height: int = 60,
              fill: bool = True) -> go.Figure:
    values = list(values)
    if not values:
        values = [0]
    fig = go.Figure(
        go.Scatter(
            y=values,
            x=list(range(len(values))),
            mode="lines",
            line=dict(color=color, width=2.2, shape="spline", smoothing=0.6),
            fill="tozeroy" if fill else None,
            fillcolor=f"rgba(13, 148, 136, 0.16)" if color == PRIMARY else "rgba(0,0,0,0.04)",
            hovertemplate="%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[len(values) - 1], y=[values[-1]],
            mode="markers",
            marker=dict(size=8, color=color,
                        line=dict(color="white", width=2)),
            hovertemplate="目前 %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=4, b=0),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False,
                   range=[min(values) - 0.02, max(values) + 0.02]),
    )
    return fig


# ---------------------------------------------------------------------------
# Live confusion matrix (threshold tuner)
# ---------------------------------------------------------------------------
def confusion_heatmap(cm: np.ndarray, threshold: float) -> go.Figure:
    text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["pred 0 (健康)", "pred 1 (故障)"],
            y=["true 0 (健康)", "true 1 (故障)"],
            colorscale=[[0, "#f8fafc"], [1, PRIMARY]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>count = %{z}<extra></extra>",
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=18, color=INK),
        )
    )
    fig = _style(fig, height=320,
                 title=f"混淆矩陣 @ threshold = {threshold:.2f}")
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return fig
