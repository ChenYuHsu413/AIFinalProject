"""Reusable Plotly chart builders for the Streamlit dashboard.

All builders return a ``plotly.graph_objects.Figure`` so the caller can
``st.plotly_chart(fig, use_container_width=True)`` without re-decorating
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
