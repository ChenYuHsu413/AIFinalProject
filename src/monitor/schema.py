"""Shared feature/radar schema for Live Monitor v3.

Single source of truth for the subsystem radar axes, the model feature tags, and
the transparent radar-severity / window-feature computations. Used by BOTH the
offline build (`build_servo_v3.py`) and the online SSE stream (backend), so the
early-warning model sees identical features in replay and live modes.

Tags are limited to what the vendored generator (`servo_v3_generator.py`)
emits (~142 columns, v4.1 physics). The generator is the single reproducible
data source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_HZ = 1000
SCENARIO_ROWS = 15_000  # 15 s @ 1000 Hz per scenario (matches the enterprise pack)
SEED = 42               # reproduces the original 30-scenario physics

# --- 6 radar subsystems: representative physical tags per axis -----------------
# (all present in the generator output; see servo_v3_generator.generate_servo_data)
SUBSYSTEMS: dict[str, list[str]] = {
    # igbt_temp_c (drive junction) is a v4.1 addition — a direct drive-thermal signal.
    "temperature": ["motor_temp_c", "drive_temp_c", "heatsink_temp_c", "bearing_temp_c", "igbt_temp_c"],
    "current": ["current_rms_a", "phase_current_imbalance_pct"],
    # vibration_kurtosis is excluded from the radar (the generator raises it
    # broadly, even in non-vibration faults) — it stays a model feature only.
    # v4.1 completes the bearing bands: bsf (ball spin) + ftf (cage).
    "vibration": ["vibration_rms_g", "vibration_peak_g",
                  "bearing_bpfo_amp", "bearing_bpfi_amp", "bearing_bsf_amp", "bearing_ftf_amp"],
    # v4.1 restores the real encoder_error_count (drift / noise / signal-loss land
    # here directly); position error / digital-twin residual stay as backups.
    "encoder": ["encoder_error_count", "position_error_pulse", "digital_twin_pos_residual"],
    "motion": ["speed_error_rpm", "torque_error_nm"],
    "communication": ["network_jitter_ms", "packet_loss_pct", "ethercat_sync_error_us"],
}

# Physical telemetry fed to the model (no labels / ids / context leakage).
FEATURE_TAGS: list[str] = sorted({t for tags in SUBSYSTEMS.values() for t in tags} | {
    "ambient_temp_c",
    "u_phase_current_a", "v_phase_current_a", "w_phase_current_a",
    "voltage_v", "dc_bus_voltage_v", "dc_bus_ripple_v",
    "vibration_x_g", "vibration_y_g", "vibration_z_g", "vibration_kurtosis",
    "encoder_velocity_count_s", "actual_torque_nm", "load_pct",
    "digital_twin_speed_residual", "digital_twin_torque_residual",
    "plc_scan_time_ms",
    # v4.1 extra physical signals now available.
    "following_error_abs_pulse", "resonance_amp",
})

# Ordinal fault stages.
STAGE_ORDER = ["normal", "early_degradation", "warning", "alarm", "trip"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_ORDER)}
WARNING_LEVEL = STAGE_TO_INT["warning"]  # early-warning positive threshold

Z_CAP = 8.0  # z-score that maps to radar severity 1.0


def healthy_baseline(healthy_df: pd.DataFrame) -> pd.DataFrame:
    """Per-tag mean & std from a healthy scenario, for radar normalisation."""
    stats = healthy_df[FEATURE_TAGS].agg(["mean", "std"]).T
    stats["std"] = stats["std"].replace(0, np.nan)  # flag-like tags handled below
    return stats


def window_features(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """Rolling mean/std/max over the physical telemetry tags."""
    roll = df[FEATURE_TAGS].rolling(win, min_periods=1)
    return pd.concat(
        [roll.mean().add_suffix("_mean"),
         roll.std().fillna(0.0).add_suffix("_std"),
         roll.max().add_suffix("_max")],
        axis=1,
    )


def radar_severity(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Transparent 0-1 severity per subsystem = clipped max robust deviation."""
    out = {}
    for name, tags in SUBSYSTEMS.items():
        sev = np.zeros(len(df))
        for t in tags:
            mean = base.loc[t, "mean"]
            std = base.loc[t, "std"]
            col = df[t].to_numpy()
            if np.isnan(std) or std < 1e-9:
                z = np.abs(col - mean) / (np.abs(mean) + 1.0)
            else:
                z = np.abs(col - mean) / std / Z_CAP
            sev = np.maximum(sev, z)
        out[name] = np.clip(sev, 0.0, 1.0)
    return pd.DataFrame(out, index=df.index)
