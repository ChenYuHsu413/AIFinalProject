"""Data analysis for the v4.1 synthetic servo dataset (Live Monitor track).

Reproducible EDA + fault-signature + model-diagnostic analysis, generated from
the vendored v4.1 generator (no raw download). Writes figures to
``outputs/figures/monitor_v4/`` and a summary to
``outputs/metrics/monitor_v4_analysis.json``.

HONESTY: this is AI-synthetic data whose labels are a deterministic function of a
single per-scenario severity ramp, and whose sensor channels are the SAME ramp +
small noise. The model-diagnostic section quantifies exactly that — it explains
why held-out F1 ≈ 1.0 and why more data would not make the task harder. Demo
track, separate from and not superseding the real PHM main line.

Run: python -m src.monitor.analyze_v4
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.monitor.schema import (
    FEATURE_TAGS,
    RAW_HZ,
    SCENARIO_ROWS,
    SEED,
    STAGE_TO_INT,
    SUBSYSTEMS,
    WARNING_LEVEL,
    healthy_baseline,
    radar_severity,
    window_features,
)
from src.monitor.servo_v3_generator import _SCENARIOS, generate_servo_data
from src.utils.paths import resolve

sns.set_theme(style="whitegrid")
FIG = resolve("outputs/figures/monitor_v4")
OUT_JSON = resolve("outputs/metrics/monitor_v4_analysis.json")

OUT_HZ = 40
STEP = RAW_HZ // OUT_HZ
WIN = int(RAW_HZ * 0.3)          # 300 ms feature window (matches the build)
AHEAD = int(round(2.0 * OUT_HZ))  # 2 s early-warning horizon (frames)

# One representative raw tag per subsystem, for readable signature/EDA panels.
REP_TAGS = {
    "temperature": "motor_temp_c",
    "current": "current_rms_a",
    "vibration": "vibration_rms_g",
    "encoder": "encoder_error_count",
    "motion": "speed_error_rpm",
    "communication": "network_jitter_ms",
}


def _save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return f"outputs/figures/monitor_v4/{name}"


def generate_all() -> tuple[dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Return {sid: full df}, the healthy baseline, and a pooled window-feature df."""
    scenarios = {sid: generate_servo_data(rows=SCENARIO_ROWS, scenario_id=sid,
                                          sampling_hz=RAW_HZ, seed=SEED)
                 for sid in range(1, len(_SCENARIOS) + 1)}
    base = healthy_baseline(scenarios[1])

    # Pooled, down-sampled window features + early-warning target (build parity).
    rows = []
    for sid, df in scenarios.items():
        idx = np.arange(0, len(df), STEP)
        feats = window_features(df, WIN).iloc[idx].reset_index(drop=True)
        stage_int = df["fault_stage"].map(STAGE_TO_INT).to_numpy()[idx]
        warn_now = (stage_int >= WARNING_LEVEL).astype(int)
        y = np.array([warn_now[i:min(len(warn_now), i + AHEAD + 1)].max()
                      for i in range(len(warn_now))])
        feats["_sid"] = sid
        feats["_category"] = str(df["fault_category"].iloc[0])
        feats["_y_warn"] = y
        rows.append(feats)
    pooled = pd.concat(rows, ignore_index=True)
    return scenarios, base, pooled


# --- EDA ---------------------------------------------------------------------
def fig_stage_timeline(scenarios) -> str:
    show = [(2, "Motor Over Temperature"), (14, "Bearing Wear"),
            (24, "EtherCAT Packet Loss"), (30, "Progressive Failure + Shutdown")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    colors = {"early_degradation": "#fde68a", "warning": "#fbbf24",
              "alarm": "#fb923c", "trip": "#ef4444"}
    for ax, (sid, name) in zip(axes.ravel(), show):
        df = scenarios[sid]
        t = df["time_s"].to_numpy()
        for tag in [REP_TAGS[s] for s in ("temperature", "current", "vibration")]:
            x = df[tag].to_numpy()
            xn = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
            ax.plot(t, xn, lw=1.0, label=tag)
        stages = df["fault_stage"].to_numpy()
        for st, col in colors.items():
            m = stages == st
            if m.any():
                ax.axvspan(t[m].min(), t[m].max(), color=col, alpha=0.18)
        ax.set_title(f"{sid:02d} · {name}", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("normalised")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Signal + fault-stage evolution (representative scenarios)", fontsize=12)
    return _save(fig, "stage_timeline.png")


def fig_correlation(pooled) -> tuple[str, float]:
    tags = [f"{t}_mean" for t in REP_TAGS.values()] + [
        "actual_torque_nm_mean", "load_pct_mean", "vibration_kurtosis_mean",
        "dc_bus_voltage_v_mean", "following_error_abs_pulse_mean", "resonance_amp_mean",
    ]
    tags = [t for t in tags if t in pooled.columns]
    corr = pooled[tags].corr()
    labels = [t.replace("_mean", "") for t in tags]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, xticklabels=labels, yticklabels=labels, cmap="coolwarm",
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature correlation (window means, all scenarios pooled)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    off = corr.where(~np.eye(len(corr), dtype=bool))
    return _save(fig, "correlation_heatmap.png"), float(np.nanmean(np.abs(off.to_numpy())))


# --- Fault-signature analysis ------------------------------------------------
def _peak_window(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask over the alarm+trip portion (peak fault state)."""
    si = df["fault_stage"].map(STAGE_TO_INT).to_numpy()
    m = si >= STAGE_TO_INT["alarm"]
    return m if m.any() else np.ones(len(df), dtype=bool)


def fig_subsystem_signature(scenarios, base) -> tuple[str, list]:
    subs = list(SUBSYSTEMS.keys())
    faults = [(sid, _SCENARIOS[sid - 1][1]) for sid in range(2, len(_SCENARIOS) + 1)]
    mat = np.zeros((len(faults), len(subs)))
    for r, (sid, _name) in enumerate(faults):
        df = scenarios[sid]
        sev = radar_severity(df, base)
        m = _peak_window(df)
        mat[r] = sev[subs].to_numpy()[m].mean(axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 11))
    sns.heatmap(mat, xticklabels=subs,
                yticklabels=[f"{sid:02d} {n}" for sid, n in faults],
                cmap="rocket_r", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "mean radar severity (alarm+trip)"})
    ax.set_title("Fault signature: which subsystem lights up per fault")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), fontsize=8)
    # argmax subsystem per fault, for the report
    argmax = [{"scenario_id": sid, "name": n, "top_subsystem": subs[int(mat[i].argmax())],
               "severity": round(float(mat[i].max()), 3)}
              for i, (sid, n) in enumerate(faults)]
    return _save(fig, "subsystem_signature_heatmap.png"), argmax


def fig_pca(pooled) -> tuple[str, float]:
    feat_cols = [c for c in pooled.columns if not c.startswith("_")]
    X = StandardScaler().fit_transform(pooled[feat_cols].to_numpy())
    pcs = PCA(n_components=2, random_state=0).fit(X)
    Z = pcs.transform(X)
    fig, ax = plt.subplots(figsize=(9, 7))
    cats = pooled["_category"].to_numpy()
    for cat in sorted(set(cats)):
        m = cats == cat
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.4, label=cat)
    ax.set_title("PCA of window-feature space (colour = fault category)")
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]*100:.0f}%)")
    ax.legend(fontsize=7, markerscale=2, ncol=2, loc="best")
    return _save(fig, "feature_separability_pca.png"), float(pcs.explained_variance_ratio_[:2].sum())


# --- Model diagnostics -------------------------------------------------------
def fig_univariate_auc(pooled) -> tuple[str, list]:
    """Honest 'importance': each feature's univariate AUC for the warning target."""
    feat_cols = [c for c in pooled.columns if not c.startswith("_")]
    y = pooled["_y_warn"].to_numpy()
    aucs = []
    for c in feat_cols:
        v = pooled[c].to_numpy()
        if np.nanstd(v) < 1e-9:
            continue
        a = roc_auc_score(y, v)
        aucs.append((c, max(a, 1 - a)))  # direction-agnostic separability
    aucs.sort(key=lambda kv: kv[1], reverse=True)
    top = aucs[:15]
    fig, ax = plt.subplots(figsize=(9, 6))
    names = [c.replace("_mean", "μ").replace("_std", "σ").replace("_max", "↑") for c, _ in top]
    ax.barh(names[::-1], [a for _, a in top][::-1], color="#3b82f6")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("univariate AUC for 'warning within 2 s'")
    ax.set_title("Single-feature separability (top 15) — no model needed")
    return _save(fig, "univariate_auc_importance.png"), [
        {"feature": c, "auc": round(a, 4)} for c, a in top]


def fig_why_f1(scenarios, pooled, top_feature: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # A: top feature separates warning vs normal near-perfectly (ECDF makes the
    # gap visible even though the 'normal' mass piles up at one value).
    ax = axes[0]
    y = pooled["_y_warn"].to_numpy()
    v = pooled[top_feature].to_numpy()
    for lab, m, col in [("normal (0)", y == 0, "#22c55e"), ("warning≤2s (1)", y == 1, "#ef4444")]:
        s = np.sort(v[m])
        ax.plot(s, np.linspace(0, 1, len(s)), color=col, lw=2.2, label=lab)
    ax.set_title(f"A · ECDF of '{top_feature}' — classes barely overlap")
    ax.set_xlabel(top_feature)
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=9, loc="lower right")

    # B: every sensor channel is the SAME severity ramp + noise.
    ax = axes[1]
    df = scenarios[30]
    t = df["time_s"].to_numpy()
    sev = (df["anomaly_score"].to_numpy())
    ax.plot(t, sev / (sev.max() + 1e-9), lw=2.2, color="k", label="severity (label driver)")
    for tag in ("motor_temp_c", "current_rms_a", "vibration_rms_g", "position_error_pulse"):
        x = np.abs(df[tag].to_numpy())
        xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
        ax.plot(t, xn, lw=1.0, alpha=0.8, label=tag)
    ax.set_title("B · scenario 30: all channels track one ramp")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Why held-out F1 ≈ 1.0 (synthetic separability — not a hard benchmark)", fontsize=12)
    return _save(fig, "why_f1_high.png")


def main() -> None:
    print("[analyze] generating 30 scenarios via vendored v4.1 generator ...")
    scenarios, base, pooled = generate_all()

    # EDA missingness (scenario 6 encoder signal loss injects NaN encoder_count).
    na_by_scn = {sid: int(df.isna().sum().sum()) for sid, df in scenarios.items()}
    na_scn = {sid: n for sid, n in na_by_scn.items() if n}

    figs = {}
    figs["stage_timeline"] = fig_stage_timeline(scenarios)
    figs["correlation"], mean_abs_corr = fig_correlation(pooled)
    figs["subsystem_signature"], argmax = fig_subsystem_signature(scenarios, base)
    figs["pca"], pc12 = fig_pca(pooled)
    figs["univariate_auc"], top_aucs = fig_univariate_auc(pooled)
    figs["why_f1"] = fig_why_f1(scenarios, pooled, top_aucs[0]["feature"])

    n_perfect = sum(1 for a in top_aucs if a["auc"] >= 0.999)
    summary = {
        "dataset": "Servo AI Dataset v4.1 Enterprise (synthetic, AI-generated via vendored generator)",
        "n_scenarios": len(scenarios),
        "rows_per_scenario": SCENARIO_ROWS,
        "n_columns": int(scenarios[1].shape[1]),
        "n_model_features": len([c for c in pooled.columns if not c.startswith("_")]),
        "missing_values": {"scenarios_with_na": na_scn,
                           "note": "only scenario 06 (encoder signal loss) injects NaN encoder_count by design"},
        "mean_abs_pairwise_corr": round(mean_abs_corr, 3),
        "pca_pc1_pc2_variance": round(pc12, 3),
        "top_univariate_auc": top_aucs[:8],
        "n_features_auc_ge_0999": n_perfect,
        "subsystem_argmax_per_fault": argmax,
        "figures": figs,
        "honesty": "Labels and sensor channels derive from one per-scenario severity ramp; "
                   "single features reach AUC≈1.0 for the warning target, so held-out F1≈1.0 "
                   "reflects synthetic separability, not benchmark difficulty. More data does "
                   "not make it harder (generator structure). Demo track, not the real PHM line.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[analyze] figures -> {FIG}")
    print(f"[analyze] summary -> {OUT_JSON}")
    print(f"  mean|corr|={mean_abs_corr:.3f}  PC1+PC2={pc12:.2%}  "
          f"features with AUC>=0.999: {n_perfect}  top={top_aucs[0]['feature']} "
          f"({top_aucs[0]['auc']})")


if __name__ == "__main__":
    main()
