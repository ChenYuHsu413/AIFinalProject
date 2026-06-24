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
# Module B — bearing health-degradation curve (RUL track)
# ---------------------------------------------------------------------------
def health_curve(
    t: Sequence,
    health: Sequence[float],
    alarm_health: float = 30.0,
    fpt_t=None,
) -> go.Figure:
    """Data-driven health indicator (100 -> 0) over the run, with the degradation
    onset (FPT) and the maintenance-alarm crossing marked.

    ``fpt_t`` (the First Predicting Time) shows when degradation was first
    detected; the alarm crossing shows how far ahead of total failure the alert
    fires.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(t), y=list(health), mode="lines",
            line=dict(color=PRIMARY, width=3),
            name="健康指標（由振動推導）",
            hovertemplate="%{x}<br>健康 = %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=alarm_health, line_dash="dash", line_color=DANGER, line_width=1.5,
        annotation_text=f"維護告警線 {alarm_health:.0f}",
        annotation_position="bottom right", annotation_font_color=DANGER,
    )
    if fpt_t is not None:
        fig.add_vline(
            x=fpt_t, line_dash="dot", line_color=ACCENT, line_width=1.5,
            annotation_text="退化起點 FPT", annotation_position="top left",
            annotation_font_color=ACCENT,
        )

    # Mark the alarm crossing and the lead time before failure.
    hh = np.asarray(health, dtype=float)
    below = np.where(hh <= alarm_health)[0]
    if below.size:
        cross_t = list(t)[below[0]]
        fig.add_vline(
            x=cross_t, line_dash="dot", line_color=WARNING, line_width=1.5,
            annotation_text="告警觸發", annotation_position="top",
            annotation_font_color=WARNING,
        )
    fig = _style(fig, height=420, title="<b>軸承健康度退化曲線</b>（100 → 0）")
    fig.update_xaxes(title_text="時間")
    fig.update_yaxes(range=[-2, 104], title_text="健康分數")
    fig.update_layout(legend=dict(orientation="h", y=1.06, x=0.0))
    return fig


def rul_forecast(t: Sequence, rul_true: Sequence[float],
                 rul_pred: Sequence[float]) -> go.Figure:
    """Predicted vs actual RUL (hours) over the degradation region.

    The actual RUL is the straight ramp to zero; the predicted curve is expected
    to be rough early and converge toward the truth as failure approaches.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(t), y=list(rul_true), mode="lines",
            line=dict(color=MUTED, width=2, dash="dash"),
            name="實際 RUL",
            hovertemplate="%{x}<br>實際 RUL = %{y:.1f} h<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(t), y=list(rul_pred), mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=4, color=ACCENT),
            name="預測 RUL",
            hovertemplate="%{x}<br>預測 RUL = %{y:.1f} h<extra></extra>",
        )
    )
    fig = _style(fig, height=360, title="<b>剩餘壽命（RUL）預測 vs 實際</b>")
    fig.update_xaxes(title_text="時間")
    fig.update_yaxes(title_text="RUL（小時）", rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0.0))
    return fig


# ---------------------------------------------------------------------------
# Module B — raw vibration waveform & spectrum (interactive explorer)
# ---------------------------------------------------------------------------
def vibration_waveform(signal: Sequence[float], fs: float,
                       title: str = "原始振動波形") -> go.Figure:
    """Time-domain waveform of one snapshot (sub-sampled for a responsive plot)."""
    x = np.asarray(signal, dtype=float)
    step = max(1, x.size // 2000)
    idx = np.arange(0, x.size, step)
    t_ms = idx / fs * 1000.0
    fig = go.Figure(
        go.Scatter(
            x=t_ms, y=x[idx], mode="lines",
            line=dict(color=PRIMARY, width=1),
            hovertemplate="%{x:.2f} ms<br>%{y:.4f} g<extra></extra>",
        )
    )
    fig = _style(fig, height=300, title=f"<b>{title}</b>")
    fig.update_xaxes(title_text="時間 (ms)")
    fig.update_yaxes(title_text="加速度 (g)")
    return fig


def vibration_spectrum(signal: Sequence[float], fs: float, defect_freqs: dict,
                       title: str = "FFT 頻譜") -> go.Figure:
    """Magnitude spectrum with bearing defect frequencies marked (BPFO/BPFI/...)."""
    x = np.asarray(signal, dtype=float)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    fig = go.Figure(
        go.Scatter(
            x=freqs, y=mag, mode="lines",
            line=dict(color=ACCENT, width=1),
            hovertemplate="%{x:.0f} Hz<br>幅值 %{y:.1f}<extra></extra>",
        )
    )
    palette = {"BPFO": DANGER, "BPFI": WARNING, "BSF": SUCCESS, "FTF": MUTED}
    for name, f in defect_freqs.items():
        if f <= freqs[-1]:
            fig.add_vline(
                x=f, line_dash="dot", line_width=1.2,
                line_color=palette.get(name, MUTED),
                annotation_text=name, annotation_position="top",
                annotation_font=dict(size=10, color=palette.get(name, MUTED)),
            )
    fig = _style(fig, height=300, title=f"<b>{title}</b>")
    fig.update_xaxes(title_text="頻率 (Hz)", range=[0, min(2000, freqs[-1])])
    fig.update_yaxes(title_text="幅值")
    return fig


# ---------------------------------------------------------------------------
# Module B+ — XJTU multi-trajectory health-indicator overlay
# ---------------------------------------------------------------------------
def xjtu_health_overlay(curves: pd.DataFrame,
                        fpt_points: pd.DataFrame | None = None) -> go.Figure:
    """Overlay each bearing's smoothed health indicator vs % of life.

    ``curves``: long-form columns ``[bearing, life_pct, hi]``.
    ``fpt_points``: optional ``[bearing, life_pct, hi]``, one detected-onset
    marker per bearing.  Shows that one fixed-parameter pipeline catches
    degradation on every independent trajectory.
    """
    palette = [PRIMARY, ACCENT, WARNING, SUCCESS, DANGER, MUTED]
    fig = go.Figure()
    if "condition" in curves.columns:
        # 15 bearings -> colour by operating condition (one legend entry each).
        conds = list(dict.fromkeys(curves["condition"]))
        cmap = {c: palette[i % len(palette)] for i, c in enumerate(conds)}
        seen: set = set()
        for (cond, bearing), g in curves.groupby(["condition", "bearing"], sort=False):
            show = cond not in seen
            seen.add(cond)
            fig.add_trace(go.Scatter(
                x=g["life_pct"], y=g["hi"], mode="lines",
                line=dict(color=cmap[cond], width=1.6), name=str(cond),
                legendgroup=str(cond), showlegend=show,
                hovertemplate=f"{bearing} ({cond})<br>壽命 %{{x:.0f}}%"
                              f"<br>HI=%{{y:.3f}}<extra></extra>",
            ))
    else:
        for i, (bearing, g) in enumerate(curves.groupby("bearing")):
            fig.add_trace(go.Scatter(
                x=g["life_pct"], y=g["hi"], mode="lines",
                line=dict(color=palette[i % len(palette)], width=2), name=str(bearing),
                hovertemplate=f"{bearing}<br>壽命 %{{x:.0f}}%<br>HI=%{{y:.3f}}<extra></extra>",
            ))
    if fpt_points is not None and not fpt_points.empty:
        fig.add_trace(go.Scatter(
            x=fpt_points["life_pct"], y=fpt_points["hi"], mode="markers",
            marker=dict(symbol="star", size=11, color="#1e293b",
                        line=dict(width=1, color="white")),
            name="退化起點 FPT",
            hovertemplate="FPT<br>壽命 %{x:.0f}%<extra></extra>",
        ))
    fig = _style(fig, height=440,
                 title="<b>多軌跡健康指標</b>（15 顆軸承 × 3 工況，同一組固定參數）")
    fig.update_xaxes(title_text="壽命進度 (%)", range=[0, 100])
    fig.update_yaxes(title_text="健康指標 h_rms（平滑）")
    fig.update_layout(legend=dict(orientation="h", y=1.06, x=0.0))
    return fig


def xjtu_replay_animation(
    records: List[dict], hi_base: float, hi_fail: float,
    fpt_minute: float, fpt_hi: float, total_minutes: float, y_max: float,
) -> go.Figure:
    """Client-side animated streaming-replay monitor for a single bearing.

    Builds ONE figure whose ``frames`` are precomputed snapshots of the health
    indicator (HI) growing over time.  Plotly's native play / pause buttons and
    slider animate it entirely in the browser — no Streamlit reruns, so no
    flicker.  Each ``records`` item is a dict with the visible curve and the
    per-frame status annotation::

        {k, x, y, mx, my, star(bool), ann(str), color(hex)}

    The healthy-baseline / failure-threshold lines and their labels are held
    fixed across frames; only the curve, the current-point marker, the FPT star
    and the status box change.
    """
    static_labels = [
        dict(x=total_minutes, y=hi_base, xref="x", yref="y", xanchor="right",
             yanchor="bottom", text="健康基線", showarrow=False,
             font=dict(color=MUTED, size=11)),
        dict(x=total_minutes, y=hi_fail, xref="x", yref="y", xanchor="right",
             yanchor="top", text="失效門檻", showarrow=False,
             font=dict(color=DANGER, size=11)),
    ]

    def _status_ann(r: dict) -> dict:
        return dict(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left",
                    yanchor="top", text=r["ann"], showarrow=False, align="left",
                    bgcolor=r["color"], font=dict(color="white", size=13),
                    bordercolor="white", borderwidth=1, borderpad=6, opacity=0.95)

    def _traces(r: dict) -> list:
        return [
            go.Scatter(x=r["x"], y=r["y"], mode="lines",
                       line=dict(color=PRIMARY, width=3), name="健康指標 HI",
                       hovertemplate="第 %{x} 分鐘<br>HI=%{y:.3f}<extra></extra>"),
            go.Scatter(x=[r["mx"]], y=[r["my"]], mode="markers", showlegend=False,
                       marker=dict(size=13, color=DANGER if r["star"] else PRIMARY,
                                   line=dict(width=2, color="white")),
                       hovertemplate="目前<br>HI=%{y:.3f}<extra></extra>"),
            go.Scatter(x=[fpt_minute] if r["star"] else [],
                       y=[fpt_hi] if r["star"] else [], mode="markers",
                       showlegend=False, name="FPT",
                       marker=dict(symbol="star", size=15, color=ACCENT,
                                   line=dict(width=1, color="white"))),
        ]

    fig = go.Figure(
        data=_traces(records[0]),
        frames=[go.Frame(name=str(i), data=_traces(r),
                         layout=go.Layout(annotations=[_status_ann(r)] + static_labels))
                for i, r in enumerate(records)],
    )
    fig.add_hline(y=hi_base, line_dash="dot", line_color=MUTED, line_width=1.5)
    fig.add_hline(y=hi_fail, line_dash="dash", line_color=DANGER, line_width=1.5)

    # Client-side speed buttons.  ``mode="immediate"`` + ``fromcurrent`` means
    # clicking another speed mid-playback interrupts the current run and resumes
    # from the CURRENT frame at the new speed — no Streamlit rerun, no reset.
    # Playback is non-looping, so it naturally stops on the last frame.
    def _play(label: str, dur: int) -> dict:
        return dict(label=label, method="animate",
                    args=[None, {"frame": {"duration": dur, "redraw": True},
                                 "fromcurrent": True, "mode": "immediate",
                                 "transition": {"duration": 0}}])

    pause = dict(label="⏸ 暫停", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}])

    fig = _style(fig, height=480)  # title lives in the page section header above
    fig.update_xaxes(title_text="時間（分鐘）", range=[0, total_minutes])
    fig.update_yaxes(title_text="HI（h_rms 平滑）", range=[0, y_max])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=58, b=58),
        annotations=[_status_ann(records[0])] + static_labels,
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            x=0.0, y=1.10, xanchor="left", yanchor="bottom", pad=dict(b=4, r=6),
            bgcolor="#f1f5f9", bordercolor="#94a3b8", borderwidth=1,
            font=dict(size=13, color="#0f172a"),
            buttons=[_play("▶ 0.5x", 300), _play("▶ 1x", 150),
                     _play("▶ 2x", 70), _play("▶ 4x", 35), pause],
        )],
        sliders=[dict(
            active=0, x=0.0, y=0, len=1.0, xanchor="left", yanchor="top",
            pad=dict(t=4, b=0), currentvalue=dict(prefix="快照 ", visible=True),
            steps=[dict(method="animate", label=str(r["k"]),
                        args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate", "transition": {"duration": 0}}])
                   for i, r in enumerate(records)],
        )],
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


def class_confusion_heatmap(cm, labels: List[str], title: str) -> go.Figure:
    """Generic multi-class confusion-matrix heatmap (rows = true, cols = pred).

    Unlike ``confusion_heatmap`` (binary / threshold-specific), the axes and title
    are parametrised so it serves any label set — e.g. Module C's
    ``["healthy", "outer", "inner"]``.
    """
    cm = np.asarray(cm)
    text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"pred {l}" for l in labels],
            y=[f"true {l}" for l in labels],
            colorscale=[[0, "#f8fafc"], [1, PRIMARY]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>count = %{z}<extra></extra>",
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=18, color=INK),
        )
    )
    fig = _style(fig, height=320, title=title)
    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")
    return fig
