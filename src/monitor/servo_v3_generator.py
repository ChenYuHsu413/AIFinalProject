#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servo AI Dataset generator — vendored v4.1 Enterprise physics.

Vendored copy of the *Servo AI Dataset v4.1 Enterprise* reproducible generator
(`servo_ai_dataset_v4_1_10M_generator.py`). It is the single, reproducible data
source for Live Monitor v3 — no 476 MB raw CSV download needed.

What v4.1 adds over the earlier v3.0 generator this file replaces:
  - per-scenario `equipment_profile` / `operation_profile` with distinct motion
    kinematics (see ``motion_profile``) instead of one shared axis formula,
  - the columns the reduced generator was missing: ``igbt_temp_c``,
    ``encoder_error_count``, ``following_error_abs_pulse``, ``bearing_bsf_amp``,
    ``bearing_ftf_amp``, ``resonance_amp`` (~142 columns total).

Kept identical to the v3 contract so ``build_servo_v3.py`` and the live stream
are unchanged: ``generate_servo_data(..., stage_fracs=...)`` (the live stream
jitters warning/alarm/trip onsets through ``stage_fracs``), ``SCENARIOS``,
``safe_name``, ``generate_all_scenarios`` and the CLI ``main``.

Honesty: this is AI-SYNTHETIC data. Provenance tags (``line_id=..._V4_1``,
``drive_model=Mitsubishi MR-J5-G simulated``, alarm codes ``V4_*``) are
simulated / generalized labels — not real OEM telemetry. Demo track, separate
from and not superseding the real PHM main line.

範例：
    python -m src.monitor.servo_v3_generator --rows 15000 --scenario 14 --output bearing.csv
    python -m src.monitor.servo_v3_generator --rows 15000 --scenario all --output_dir output
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# (scenario_id, name, fault_category, alarm_code, equipment_profile, operation_profile, root_cause)
_SCENARIOS = [
    (1,  "Healthy Baseline",               "NORMAL",        "NONE",       "Pick and Place Indexer",           "Smooth bidirectional indexing",        "Normal production operation"),
    (2,  "Motor Over Temperature",         "TEMPERATURE",   "V4_TMP_001", "Heavy Press Axis",                 "Heavy continuous duty",                "Motor winding temperature rises progressively"),
    (3,  "Drive Over Temperature",         "TEMPERATURE",   "V4_TMP_002", "Tray Loader",                      "High duty rapid indexing",             "Servo amplifier heat sink temperature rises"),
    (4,  "Encoder Drift",                  "ENCODER",       "V4_ENC_001", "Long Stroke XY Gantry",            "Long stroke positioning",              "Encoder absolute position gradually drifts"),
    (5,  "Encoder Noise",                  "ENCODER",       "V4_ENC_002", "Vision Alignment Stage",           "Fine pitch micro positioning",         "Encoder signal contains jitter and count noise"),
    (6,  "Encoder Signal Loss",            "ENCODER",       "V4_ENC_003", "High Speed Conveyor",              "Intermittent high speed transfer",     "Intermittent loss of encoder feedback"),
    (7,  "Position Following Error",       "POSITION",      "V4_POS_001", "Robot Joint Axis",                 "High acceleration point to point",     "Actual position cannot follow command"),
    (8,  "Overspeed",                      "SPEED",         "V4_SPD_001", "High Speed Spinner",               "High speed rotary axis",               "Actual speed exceeds safe threshold"),
    (9,  "Acceleration Overshoot",         "MOTION",        "V4_MOT_001", "S-Curve Indexer",                  "Aggressive S curve acceleration",      "Acceleration spike during motion profile"),
    (10, "Deceleration Failure",           "MOTION",        "V4_MOT_002", "Stop-Go Inspection Axis",          "Short pitch stop and go",              "Deceleration tracking failure"),
    (11, "Over Current",                   "CURRENT",       "V4_CUR_001", "Press Fit Axis",                   "Press fit load axis",                  "Phase current exceeds allowed range"),
    (12, "Torque Saturation",              "TORQUE",        "V4_TRQ_001", "Torque Clamp Axis",                "Clamp and hold torque operation",      "Torque command saturates under load"),
    (13, "Mechanical Jam",                 "MECHANICAL",    "V4_MEC_001", "Transfer Arm",                     "Transfer arm blocked motion",          "Axis movement blocked by mechanical interference"),
    (14, "Bearing Wear",                   "MECHANICAL",    "V4_MEC_002", "Rotary Table",                     "High cycle rotary axis",               "Bearing degradation increases vibration"),
    (15, "Bearing Lubrication Failure",    "MECHANICAL",    "V4_MEC_003", "Heavy Linear Slide",               "Slow heavy load slide",                "Friction increases due to lubrication failure"),
    (16, "Rotor Imbalance",                "MECHANICAL",    "V4_MEC_004", "Spindle Assist Axis",              "Constant speed spindle assist",        "Rotor imbalance creates periodic vibration"),
    (17, "Coupling Misalignment",          "MECHANICAL",    "V4_MEC_005", "Coupling Drive Axis",              "Reversing load transfer",              "Coupling misalignment creates torque ripple"),
    (18, "Ball Screw Wear",                "MECHANICAL",    "V4_MEC_006", "Ball Screw Linear Stage",          "Long travel linear stage",             "Ball screw wear increases friction and position error"),
    (19, "Backlash",                       "MECHANICAL",    "V4_MEC_007", "Bidirectional Index Table",        "Frequent direction reversal",          "Mechanical play during direction reversal"),
    (20, "High Vibration",                 "VIBRATION",     "V4_VIB_001", "High Speed Scanner",               "Resonant mid speed scanning",          "Axis vibration exceeds production threshold"),
    (21, "Power Supply Fluctuation",       "POWER",         "V4_PWR_001", "Multi Axis Start Station",         "Multi axis simultaneous start",        "Input voltage and DC bus are unstable"),
    (22, "Voltage Drop",                   "POWER",         "V4_PWR_002", "Peak Load Acceleration Axis",      "Peak load acceleration",               "DC bus under-voltage during acceleration or load"),
    (23, "Communication Timeout",          "COMMUNICATION", "V4_COM_001", "PLC Sync Motion Axis",             "PLC synchronized indexing",            "PLC or motion controller command timeout"),
    (24, "EtherCAT Packet Loss",           "COMMUNICATION", "V4_COM_002", "Distributed EtherCAT Axis",        "Distributed motion control",           "EtherCAT packet loss and sync jitter"),
    (25, "Servo Gain Instability",         "CONTROL",       "V4_CTL_001", "Precision Settling Axis",          "High stiffness micro settling",        "Servo gain too high causing oscillation"),
    (26, "Resonance",                      "CONTROL",       "V4_CTL_002", "Resonance Test Axis",              "Mid frequency oscillating load",       "Mechanical resonance frequency excited by motion"),
    (27, "Brake Failure",                  "SAFETY",        "V4_SAF_001", "Vertical Z Axis",                  "Vertical Z axis with holding brake",   "Holding brake fails to release or hold correctly"),
    (28, "Emergency Stop",                 "SAFETY",        "V4_SAF_002", "Production Line E-Stop",           "Normal run interrupted by E-stop",     "Emergency stop input activated"),
    (29, "Combined Fault",                 "MULTI_FAULT",   "V4_MUL_001", "Factory Multi-Failure Axis",       "Overload vibration thermal coupling",  "Overload with vibration and temperature rise"),
    (30, "Progressive Failure + Shutdown", "MULTI_FAULT",   "V4_MUL_002", "Predictive Maintenance Demo Axis", "Long degradation to final shutdown",   "Progressive bearing/load failure ending in servo trip"),
]

# Backward-compatible 4-tuple view (name, category, alarm_code, root_cause).
# Consumers only need len()==30 and the first four fields.
SCENARIOS = [(name, cat, alarm, root) for (_id, name, cat, alarm, _eq, _op, root) in _SCENARIOS]


def safe_name(s: str) -> str:
    return s.lower().replace(" + ", "_plus_").replace(" ", "_").replace("/", "_").replace("-", "_")


def stage_arrays(t, scenario_id, duration,
                 stage_fracs: tuple[float, float, float] = (0.53, 0.73, 0.88)):
    """Ordinal fault stages + warning/alarm/trip flags + a 0-1 severity ramp.

    Fault onset is fixed at 30 % of the run; the warning/alarm/trip label onsets
    are ``stage_fracs`` (configurable so the live stream can jitter fault timing;
    the defaults reproduce the original replay pack).
    """
    if scenario_id == 1:
        z = np.zeros(len(t), dtype=np.int8)
        return np.array(["normal"] * len(t), dtype=object), z, z, z, np.zeros(len(t), dtype=np.float32)
    warn_f, alarm_f, trip_f = stage_fracs
    fs = duration * 0.30
    wt, at, tt = duration * warn_f, duration * alarm_f, duration * trip_f
    sev = np.clip((t - fs) / max(duration - fs, 1e-9), 0, 1).astype(np.float32)
    stage = np.where(t < fs, "normal",
             np.where(t < wt, "early_degradation",
             np.where(t < at, "warning",
             np.where(t < tt, "alarm", "trip"))))
    return stage, (t >= wt).astype(np.int8), (t >= at).astype(np.int8), (t >= tt).astype(np.int8), sev


def motion_profile(t, sid):
    """Per-equipment position command + speed scale + base load (v4.1)."""
    dur = max(t[-1] + (t[1] - t[0] if len(t) > 1 else 0.001), 1.0)
    if sid == 1: pos = 100000 + 4200 * np.sin(2 * np.pi * t / 1.5) + 520 * t; ss = 1600; lb = 18
    elif sid == 2: pos = 120000 + 2600 * np.sin(2 * np.pi * t / 2.2) + 900 * np.sin(2 * np.pi * t / 0.55); ss = 1250; lb = 45
    elif sid == 3: pos = 90000 + 3000 * np.tanh(3 * np.sin(2 * np.pi * t / 0.85)) + 250 * t; ss = 2400; lb = 35
    elif sid == 4: pos = 80000 + 12000 * (t / dur) + 1500 * np.sin(2 * np.pi * t / 3.0); ss = 1400; lb = 25
    elif sid == 5: pos = 100000 + 280 * np.sin(2 * np.pi * t / 0.35) + 60 * np.sin(2 * np.pi * t / 0.08); ss = 450; lb = 15
    elif sid == 6: pos = 110000 + 5000 * np.sin(2 * np.pi * t / 0.70) + 300 * t; ss = 2600; lb = 22
    elif sid == 7: pos = 100000 + 5200 * np.sin(2 * np.pi * t / 0.95) ** 3 + 650 * t; ss = 2300; lb = 30
    elif sid == 8: pos = 100000 + 6200 * np.sin(2 * np.pi * t / 0.60) + 800 * t; ss = 2950; lb = 25
    elif sid == 9:
        cyc = 1.1; s = (t % cyc) / cyc; pos = 100000 + 5000 * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5) + 500 * (t // cyc); ss = 2200; lb = 28
    elif sid == 10:
        cyc = 0.75; s = (t % cyc) / cyc; pos = 100000 + 3000 * np.where(s < 0.55, s / 0.55, 1 - (s - 0.55) / 0.45); ss = 1800; lb = 24
    elif sid == 11: pos = 100000 + 1800 * np.sin(2 * np.pi * t / 1.8); ss = 900; lb = 70
    elif sid == 12: pos = 100000 + 1200 * np.tanh(4 * np.sin(2 * np.pi * t / 3.0)); ss = 700; lb = 80
    elif sid == 13: pos = 100000 + 4600 * np.sin(2 * np.pi * t / 1.3) + 300 * t; ss = 1500; lb = 50
    elif sid == 14: pos = 100000 + 2500 * np.sin(2 * np.pi * t / 0.50); ss = 2100; lb = 30
    elif sid == 15: pos = 100000 + 2300 * np.sin(2 * np.pi * t / 2.6) + 120 * t; ss = 850; lb = 65
    elif sid == 16: pos = 100000 + 1800 * np.sin(2 * np.pi * t / 0.45); ss = 2200; lb = 20
    elif sid == 17: pos = 100000 + 4200 * np.sin(2 * np.pi * t / 1.0); ss = 1600; lb = 45
    elif sid == 18: pos = 80000 + 15000 * (t / dur) + 800 * np.sin(2 * np.pi * t / 2.5); ss = 1000; lb = 55
    elif sid == 19: pos = 100000 + 2500 * np.sign(np.sin(2 * np.pi * t / 0.8)) + 300 * np.sin(2 * np.pi * t / 0.8); ss = 1200; lb = 38
    elif sid == 20: pos = 100000 + 3600 * np.sin(2 * np.pi * t / 1.2); ss = 1500; lb = 28
    elif sid == 21: pos = 100000 + 4000 * np.sin(2 * np.pi * t / 0.9) + 500 * np.sin(2 * np.pi * t / 0.3); ss = 2000; lb = 40
    elif sid == 22: pos = 100000 + 5200 * np.sin(2 * np.pi * t / 0.95) ** 3; ss = 2400; lb = 60
    elif sid == 23: pos = 100000 + 4200 * np.sin(2 * np.pi * t / 1.5); ss = 1500; lb = 25
    elif sid == 24: pos = 100000 + 3800 * np.sin(2 * np.pi * t / 0.7); ss = 1800; lb = 26
    elif sid == 25: pos = 100000 + 450 * np.sin(2 * np.pi * t / 0.42) + 80 * np.sin(2 * np.pi * t / 0.09); ss = 650; lb = 20
    elif sid == 26: pos = 100000 + 3600 * np.sin(2 * np.pi * t / 1.15); ss = 1450; lb = 32
    elif sid == 27: pos = 100000 + 6000 * np.maximum(0, np.sin(2 * np.pi * t / 2.0)); ss = 900; lb = 75
    elif sid == 28: pos = 100000 + 4300 * np.sin(2 * np.pi * t / 1.2) + 250 * t; ss = 1500; lb = 30
    elif sid == 29: pos = 100000 + 4800 * np.sin(2 * np.pi * t / 1.0) ** 3 + 250 * t; ss = 1900; lb = 70
    else: pos = 100000 + 5200 * np.sin(2 * np.pi * t / 1.4) + 350 * t; ss = 1700; lb = 75
    return pos, ss, lb


def generate_servo_data(rows: int = 1000, scenario_id: int = 1, sampling_hz: int = 1000,
                        seed: int = 42, global_start_s: float = 0.0,
                        stage_fracs: tuple[float, float, float] = (0.53, 0.73, 0.88)) -> pd.DataFrame:
    """Generate one scenario's servo PLC/drive log (v4.1 physics).

    ``stage_fracs`` = (warning, alarm, trip) label onsets as fractions of the run
    duration; the live stream jitters these so lead time varies realistically.
    """
    if scenario_id < 1 or scenario_id > len(_SCENARIOS):
        raise ValueError("scenario_id must be 1~30")
    sid, name, cat, alarm, equip, op, root = _SCENARIOS[scenario_id - 1]
    rng = np.random.default_rng(seed + scenario_id)
    dt = 1 / sampling_hz
    duration = rows / sampling_hz
    t = np.arange(rows, dtype=np.float32) * dt
    gt = global_start_s + t
    fault_stage, warning, alarm_flag, trip, sev = stage_arrays(t, scenario_id, duration, stage_fracs)
    sev2 = sev ** 1.8
    pos_cmd, speed_scale, load_base = motion_profile(t, scenario_id)
    speed_raw = np.gradient(pos_cmd, dt)
    speed_cmd = speed_raw / (np.max(np.abs(speed_raw)) + 1e-9) * speed_scale
    accel_cmd = np.gradient(speed_cmd, dt)
    jerk_cmd = np.gradient(accel_cmd, dt)
    pos_error = rng.normal(0, 3.5, rows)
    actual_speed = speed_cmd + rng.normal(0, speed_scale * 0.003, rows)
    actual_accel = np.gradient(actual_speed, dt)
    torque_cmd = 0.25 + 0.004 * load_base + 0.25 * np.abs(accel_cmd) / (np.max(np.abs(accel_cmd)) + 1e-9) + rng.normal(0, 0.012, rows)
    actual_torque = torque_cmd + rng.normal(0, 0.022, rows)
    current = 0.8 + 0.025 * load_base + 1.7 * np.abs(actual_torque) + rng.normal(0, 0.035, rows)
    voltage = 220 + rng.normal(0, 0.55, rows)
    dc_bus = 310 + rng.normal(0, 1.1, rows)
    motor_temp = 32 + 0.018 * load_base + 0.05 * t + rng.normal(0, 0.1, rows)
    drive_temp = 36 + 0.012 * load_base + 0.045 * t + rng.normal(0, 0.1, rows)
    bearing_temp = 31 + 0.010 * load_base + 0.035 * t + rng.normal(0, 0.08, rows)
    load_pct = load_base + 20 * np.abs(actual_torque) + rng.normal(0, 1.2, rows)
    vib_x = 0.018 + 0.000015 * speed_scale + rng.normal(0, 0.003, rows)
    vib_y = 0.016 + rng.normal(0, 0.003, rows)
    vib_z = 0.015 + rng.normal(0, 0.003, rows)
    network_jitter = np.abs(rng.normal(0.10, 0.025, rows))
    packet_loss = np.zeros(rows)
    plc_scan = 2.0 + rng.normal(0, 0.06, rows)
    safety_sto = np.zeros(rows, dtype=np.int8)
    brake_release = np.ones(rows, dtype=np.int8)

    # unique fault evolution
    if scenario_id == 2: motor_temp += 55 * sev2; current += 0.8 * sev
    elif scenario_id == 3: drive_temp += 50 * sev2; dc_bus += 6 * sev * np.sin(2 * np.pi * 3 * t)
    elif scenario_id == 4: pos_error += (160 * sev2) / 10
    elif scenario_id == 5: pos_error += rng.normal(0, 12 * sev, rows)
    elif scenario_id == 6: pos_error += 100 * sev
    elif scenario_id == 7: pos_error += 220 * sev2; actual_speed -= 120 * sev
    elif scenario_id == 8: actual_speed += 950 * sev2
    elif scenario_id == 9: actual_accel += 4200 * sev * np.sin(2 * np.pi * 4 * t); actual_speed += 220 * sev * np.sin(2 * np.pi * 4 * t)
    elif scenario_id == 10: actual_speed += np.where(((t % 0.75) / 0.75) > 0.62, 380 * sev2, 0)
    elif scenario_id == 11: current += 6.2 * sev2; actual_torque += 0.9 * sev
    elif scenario_id == 12: torque_cmd += 1.1 * sev2; actual_torque += 1.25 * sev2; current += 2.8 * sev2
    elif scenario_id == 13:
        jam = np.clip((t - duration * 0.6) / max(duration * 0.18, 1e-9), 0, 1); actual_speed *= (1 - 0.9 * jam); pos_error += 550 * jam; current += 6.5 * jam
    elif scenario_id == 14: vib_x += 0.23 * sev2 + 0.06 * np.sin(2 * np.pi * 95 * t) * sev; bearing_temp += 26 * sev2
    elif scenario_id == 15: current += 2.0 * sev2; bearing_temp += 36 * sev2; vib_y += 0.16 * sev2
    elif scenario_id == 16: vib_x += 0.18 * np.sin(2 * np.pi * 60 * t) * sev + 0.12 * sev; vib_z += 0.14 * np.sin(2 * np.pi * 60 * t + 1.2) * sev
    elif scenario_id == 17: actual_torque += 0.42 * np.sin(2 * np.pi * 8 * t) * sev; vib_y += 0.14 * sev2; pos_error += 45 * np.sin(2 * np.pi * 8 * t) * sev
    elif scenario_id == 18: load_pct += 34 * sev2; current += 2.7 * sev2; pos_error += 110 * sev2
    elif scenario_id == 19: pos_error += 90 * sev * np.where(np.sign(np.gradient(speed_cmd, dt)) >= 0, 1, -1)
    elif scenario_id == 20: vib_x += 0.30 * sev2; vib_y += 0.27 * sev2; vib_z += 0.24 * sev2
    elif scenario_id == 21: voltage += 22 * sev * np.sin(2 * np.pi * 2.5 * t); dc_bus += 30 * sev * np.sin(2 * np.pi * 2.5 * t)
    elif scenario_id == 22: voltage -= 45 * sev2; dc_bus -= 66 * sev2; current += 1.5 * sev
    elif scenario_id == 23: network_jitter += 10 * sev2; plc_scan += 12 * sev2
    elif scenario_id == 24: packet_loss += 22 * sev2; network_jitter += 5.2 * sev
    elif scenario_id == 25: pos_error += 85 * np.sin(2 * np.pi * 16 * t) * sev; actual_speed += 260 * np.sin(2 * np.pi * 16 * t) * sev
    elif scenario_id == 26: vib_x += 0.25 * np.sin(2 * np.pi * 120 * t) * sev; actual_torque += 0.32 * np.sin(2 * np.pi * 120 * t) * sev
    elif scenario_id == 27: brake_release = np.where(t > duration * 0.73, 0, 1); actual_speed *= np.where(t > duration * 0.73, 0.5, 1); current += np.where(t > duration * 0.73, 3.2 * sev, 0)
    elif scenario_id == 28: safety_sto = (t > duration * 0.70).astype(np.int8); actual_speed = np.where(t > duration * 0.70, actual_speed * np.exp(-(t - duration * 0.70) * 3), actual_speed)
    elif scenario_id == 29: current += 3.4 * sev2; motor_temp += 33 * sev2; vib_x += 0.20 * sev2; pos_error += 150 * sev2; load_pct += 24 * sev2
    elif scenario_id == 30: current += 4.6 * sev2; motor_temp += 40 * sev2; bearing_temp += 34 * sev2; vib_x += 0.26 * sev2; load_pct += 42 * sev2; pos_error += 360 * sev2

    actual_pos = pos_cmd + pos_error
    encoder_count = actual_pos * 10 + 52312 + rng.normal(0, 1.8, rows)
    if scenario_id == 4: encoder_count += 160 * sev2
    if scenario_id == 5: encoder_count += rng.normal(0, 95 * sev, rows)
    if scenario_id == 6:
        mask = (t > duration * 0.60) & (np.sin(2 * np.pi * 22 * t) > 0.90); encoder_count[mask] = np.nan
    encoder_i = pd.Series(encoder_count).interpolate(limit_direction="both").to_numpy()
    encoder_vel = np.gradient(encoder_i, dt)

    vib_rms = np.sqrt(vib_x ** 2 + vib_y ** 2 + vib_z ** 2)
    power_w = voltage * current * 0.85
    anomaly = np.clip(sev * 0.86 + warning * 0.04 + alarm_flag * 0.06 + trip * 0.04 + rng.normal(0, 0.012, rows), 0, 1) if scenario_id != 1 else np.clip(rng.normal(0.018, 0.008, rows), 0, 0.07)
    health = np.clip(100 - anomaly * 96 - trip * 4, 0, 100)
    trip_time = duration * stage_fracs[2]
    rul_sec = np.where(scenario_id == 1, 9999, np.maximum(trip_time - t, 0))
    u = current / np.sqrt(3) + rng.normal(0, 0.025, rows); v = current / np.sqrt(3) + rng.normal(0, 0.025, rows); w = current / np.sqrt(3) + rng.normal(0, 0.025, rows)

    # vision / robot / IO synthetic process tags
    vision_score = np.clip(98 - anomaly * 18 + rng.normal(0, 0.4, rows), 0, 100)
    robot_joint_load = np.clip(load_pct * 0.4 + np.abs(actual_torque) * 8 + rng.normal(0, 0.5, rows), 0, 100)
    io_input_word = (warning.astype(int) * 2 + alarm_flag.astype(int) * 4 + trip.astype(int) * 8).astype(np.int32)
    io_output_word = ((1 - trip).astype(int) * 1 + brake_release.astype(int) * 16).astype(np.int32)

    df = pd.DataFrame({
        "scenario_id": scenario_id, "scenario_name": name, "equipment_profile": equip, "operation_profile": op,
        "fault_type": "NORMAL" if scenario_id == 1 else name, "fault_category": cat,
        "alarm_code": np.where(alarm_flag == 1, alarm, "NONE"), "root_cause": root,
        "time_s": t, "global_time_s": gt, "sample_index": np.arange(rows),
        "fault_stage": fault_stage, "warning": warning, "alarm": alarm_flag, "trip": trip,
        "pos_cmd_pulse": pos_cmd, "actual_pos_pulse": actual_pos, "position_error_pulse": pos_error,
        "following_error_abs_pulse": np.abs(pos_error), "encoder_count": encoder_count,
        "encoder_count_interpolated": encoder_i, "encoder_velocity_count_s": encoder_vel,
        "encoder_error_count": encoder_count - (actual_pos * 10 + 52312), "absolute_position_valid": np.where(np.isnan(encoder_count), 0, 1),
        "speed_cmd_rpm": speed_cmd, "actual_speed_rpm": actual_speed, "speed_error_rpm": actual_speed - speed_cmd,
        "speed_abs_rpm": np.abs(actual_speed), "accel_cmd_rpm_s": accel_cmd, "actual_accel_rpm_s": actual_accel,
        "accel_error_rpm_s": actual_accel - accel_cmd, "jerk_cmd_rpm_s2": jerk_cmd,
        "torque_cmd_nm": torque_cmd, "actual_torque_nm": actual_torque, "torque_error_nm": actual_torque - torque_cmd,
        "torque_limit_pct": np.clip(np.abs(actual_torque) / 2.5 * 100, 0, 180),
        "current_rms_a": current, "u_phase_current_a": u, "v_phase_current_a": v, "w_phase_current_a": w,
        "phase_current_imbalance_pct": np.abs(u - v) / (current + 1e-9) * 100, "current_limit_pct": np.clip(current / 8 * 100, 0, 200),
        "voltage_v": voltage, "input_voltage_v": voltage + rng.normal(0, 0.4, rows), "dc_bus_voltage_v": dc_bus,
        "dc_bus_ripple_v": np.abs(np.gradient(dc_bus, dt)) / 1000, "regen_load_pct": np.clip(7 + np.abs(accel_cmd) / (np.max(np.abs(accel_cmd)) + 1e-9) * 22 + rng.normal(0, 0.8, rows), 0, 100),
        "power_w": power_w, "energy_wh": np.cumsum(power_w) * dt / 3600, "pwm_duty_pct": np.clip(18 + np.abs(actual_speed) / 3000 * 55 + np.abs(actual_torque) * 8, 0, 100),
        "motor_temp_c": motor_temp, "drive_temp_c": drive_temp, "heatsink_temp_c": drive_temp + 2 + rng.normal(0, 0.1, rows),
        "bearing_temp_c": bearing_temp, "ambient_temp_c": 25 + 0.35 * np.sin(2 * np.pi * t / max(duration, 1e-9)) + rng.normal(0, 0.04, rows),
        "cabinet_temp_c": 29 + 0.2 * np.sin(2 * np.pi * t / max(duration, 1e-9)) + rng.normal(0, 0.05, rows),
        "cooling_fan_speed_rpm": 3200 + drive_temp * 18 + rng.normal(0, 20, rows), "igbt_temp_c": drive_temp + 4 + rng.normal(0, 0.1, rows),
        "load_pct": np.clip(load_pct, 0, 180), "load_inertia_ratio": 2.4 + scenario_id * 0.08 + rng.normal(0, 0.02, rows),
        "friction_estimate_nm": 0.04 + current * 0.014 + rng.normal(0, 0.002, rows),
        "backlash_estimate_pulse": np.clip(np.abs(pos_error) * 0.08 + (scenario_id == 19) * sev * 45, 0, 999),
        "mechanical_clearance_um": np.clip(8 + np.abs(pos_error) * 0.015 + (scenario_id in [18, 19]) * sev * 25, 0, 999),
        "ball_screw_temp_c": bearing_temp + 1 + rng.normal(0, 0.07, rows), "coupling_temp_c": 31 + np.abs(actual_torque) * 1.2 + rng.normal(0, 0.05, rows),
        "brake_release_status": brake_release, "brake_current_a": brake_release * 0.42 + rng.normal(0, 0.01, rows),
        "vibration_x_g": vib_x, "vibration_y_g": vib_y, "vibration_z_g": vib_z, "vibration_rms_g": vib_rms,
        "vibration_peak_g": vib_rms * 2.8 + rng.normal(0, 0.01, rows), "vibration_kurtosis": 3 + 8 * sev2 + rng.normal(0, 0.08, rows),
        "fft_1x_amp": np.abs(vib_x) * 10 + rng.normal(0, 0.01, rows), "fft_2x_amp": np.abs(vib_y) * 8 + rng.normal(0, 0.01, rows),
        "fft_3x_amp": np.abs(vib_z) * 7 + rng.normal(0, 0.01, rows), "fft_noise_floor": 0.02 + anomaly * 0.08 + rng.normal(0, 0.003, rows),
        "bearing_bpfo_amp": np.abs(vib_z) * 6 + (scenario_id in [14, 15]) * sev * 1.5,
        "bearing_bpfi_amp": np.abs(vib_y) * 6 + (scenario_id in [14, 15]) * sev * 1.2,
        "bearing_bsf_amp": np.abs(vib_x) * 5 + (scenario_id in [14, 15]) * sev * 1.0,
        "bearing_ftf_amp": np.abs(vib_z) * 4 + (scenario_id in [14, 15]) * sev * 0.8,
        "resonance_amp": (scenario_id in [25, 26]) * sev * 2.0 + np.abs(vib_x) * 4,
        "network_jitter_ms": network_jitter, "packet_loss_pct": packet_loss,
        "ethercat_sync_error_us": np.abs(rng.normal(4, 1.2, rows)) + packet_loss * 2,
        "cc_link_ie_cycle_error": ((scenario_id == 24) & (alarm_flag == 1)).astype(np.int8),
        "ethercat_wkc_error_count": ((scenario_id == 24) * sev * 100).astype(np.int32),
        "plc_scan_time_ms": plc_scan, "plc_cycle_counter": (gt * 1000).astype(np.int64),
        "plc_command_id": scenario_id * 1000 + (t // 1.5).astype(np.int32), "plc_step_no": np.select([t % 1.5 < 0.3, t % 1.5 < 0.8, t % 1.5 < 1.2], [10, 20, 30], default=40),
        "motion_controller_online": (1 - trip).astype(np.int8), "drive_ready": (1 - trip).astype(np.int8), "servo_ready": (1 - trip).astype(np.int8),
        "in_position": (np.abs(pos_error) < 35).astype(np.int8), "motion_complete": ((t % 1.5) > 1.43).astype(np.int8),
        "sto_status": safety_sto, "safety_relay_closed": (1 - safety_sto).astype(np.int8),
        "door_interlock_closed": np.ones(rows, dtype=np.int8), "light_curtain_clear": np.ones(rows, dtype=np.int8),
        "vision_alignment_score": vision_score, "vision_offset_x_um": rng.normal(0, 2, rows) + pos_error * 0.01,
        "vision_offset_y_um": rng.normal(0, 2, rows) + pos_error * 0.008, "aoi_pass_flag": (vision_score > 85).astype(np.int8),
        "robot_joint_load_pct": robot_joint_load, "robot_tcp_speed_mm_s": np.abs(actual_speed) * 0.12,
        "robot_position_error_mm": np.abs(pos_error) * 0.001, "io_input_word": io_input_word, "io_output_word": io_output_word,
        "control_mode": "position", "operation_state": np.where(trip == 1, "TRIP", np.where(alarm_flag == 1, "ALARM", np.where(warning == 1, "WARNING", "RUN"))),
        "gain_p": 120 + scenario_id * 0.4 + rng.normal(0, 0.18, rows), "gain_i": 28 + scenario_id * 0.05 + rng.normal(0, 0.04, rows),
        "gain_d": 0.8 + scenario_id * 0.005 + rng.normal(0, 0.003, rows), "notch_filter_hz": 120 + (scenario_id % 6) * 20,
        "adaptive_tuning_level": np.clip(50 + sev * 25 + rng.normal(0, 0.5, rows), 0, 100),
        "servo_stiffness_level": np.full(rows, 18 + scenario_id % 8),
        "digital_twin_pos_residual": pos_error + rng.normal(0, 1.5, rows),
        "digital_twin_speed_residual": actual_speed - speed_cmd + rng.normal(0, 2.0, rows),
        "digital_twin_torque_residual": actual_torque - torque_cmd + rng.normal(0, 0.008, rows),
        "digital_twin_temp_residual": motor_temp - (32 + 0.018 * load_base + 0.05 * t) + rng.normal(0, 0.1, rows),
        "digital_twin_vibration_residual": vib_rms - (0.03 + 0.00001 * speed_scale) + rng.normal(0, 0.002, rows),
        "anomaly_score": anomaly, "failure_probability": np.clip(anomaly ** 1.25, 0, 1), "health_index": health,
        "bearing_health_index": np.clip(100 - (bearing_temp - 33) * 1.0 - vib_rms * 65, 0, 100),
        "rul_sec": rul_sec, "rul_cycle": np.where(scenario_id == 1, 9999, np.maximum(rul_sec / 1.5, 0)),
        "maintenance_required": (alarm_flag | trip).astype(np.int8),
        "maintenance_priority": np.where(trip == 1, 3, np.where(alarm_flag == 1, 2, np.where(warning == 1, 1, 0))),
        "predictive_maintenance_label": np.where(trip == 1, "TRIP", np.where(alarm_flag == 1, "REPAIR_NOW", np.where(warning == 1, "PLAN_MAINTENANCE", "NORMAL"))),
        "failure_within_1s": ((rul_sec <= 1) & (scenario_id != 1)).astype(np.int8),
        "failure_within_3s": ((rul_sec <= 3) & (scenario_id != 1)).astype(np.int8),
        "failure_within_5s": ((rul_sec <= 5) & (scenario_id != 1)).astype(np.int8),
        "maintenance_action": np.where(scenario_id == 1, "No action", np.where(alarm_flag == 1, "Stop and inspect axis", "Monitor trend")),
        "line_id": "TSMC_ASE_AUTO_LINE_SIM_V4_1", "station_id": f"STATION_{scenario_id % 6 + 1:02d}",
        "axis_name": f"AXIS_{scenario_id:02d}", "drive_model": "Mitsubishi MR-J5-G simulated",
        "motor_model": "HK-KT / HG-KR simulated", "recipe_id": f"RECIPE_{scenario_id % 5 + 1:02d}", "lot_id": f"LOT_SIM_{4000 + scenario_id:04d}",
    })
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].astype("float32").round(5)
    return df


def generate_all_scenarios(rows_per_scenario: int, output_dir: Path, file_format: str, sampling_hz: int, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    global_start = 0.0
    for scenario_id, (name, category, alarm_code, root_cause) in enumerate(SCENARIOS, start=1):
        df = generate_servo_data(rows=rows_per_scenario, scenario_id=scenario_id,
                                 sampling_hz=sampling_hz, seed=seed, global_start_s=global_start)
        filename = f"scenario_{scenario_id:02d}_{safe_name(name)}.{file_format}"
        path = output_dir / filename
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError("file_format must be csv or parquet")
        index_rows.append({
            "scenario_id": scenario_id, "scenario_name": name, "fault_category": category,
            "alarm_code": alarm_code, "rows": len(df), "sampling_hz": sampling_hz,
            "duration_sec": rows_per_scenario / sampling_hz, "global_start_s": global_start,
            "global_end_s": global_start + rows_per_scenario / sampling_hz - 1 / sampling_hz,
            "file_name": filename, "root_cause": root_cause,
        })
        global_start += rows_per_scenario / sampling_hz
    pd.DataFrame(index_rows).to_csv(output_dir / "scenario_index_generated.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Servo AI Dataset generator (vendored v4.1 physics)")
    parser.add_argument("--rows", type=int, default=1000, help="rows per dataset or per scenario, default=1000")
    parser.add_argument("--scenario", default="1", help="1~30, all, or mixed. default=1")
    parser.add_argument("--sampling_hz", type=int, default=1000, help="sampling frequency, default=1000")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output", default="servo_generated_1000.csv", help="single output file path")
    parser.add_argument("--output_dir", default="servo_generated_output", help="output directory for all scenarios")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="csv or parquet")
    args = parser.parse_args()

    if args.scenario == "all":
        generate_all_scenarios(rows_per_scenario=args.rows, output_dir=Path(args.output_dir),
                               file_format=args.format, sampling_hz=args.sampling_hz, seed=args.seed)
        print(f"Generated all 30 scenarios in: {args.output_dir}")
        return

    if args.scenario == "mixed":
        parts = []
        rows_left = args.rows
        global_start = 0.0
        scenario_id = 1
        while rows_left > 0:
            chunk_rows = min(rows_left, max(1000, args.sampling_hz * 15))
            df = generate_servo_data(rows=chunk_rows, scenario_id=scenario_id, sampling_hz=args.sampling_hz,
                                     seed=args.seed + scenario_id, global_start_s=global_start)
            parts.append(df)
            rows_left -= chunk_rows
            global_start += chunk_rows / args.sampling_hz
            scenario_id = scenario_id + 1 if scenario_id < 30 else 1
        out = pd.concat(parts, ignore_index=True)
    else:
        out = generate_servo_data(rows=args.rows, scenario_id=int(args.scenario),
                                  sampling_hz=args.sampling_hz, seed=args.seed, global_start_s=0)

    output_path = Path(args.output)
    if args.format == "csv":
        out.to_csv(output_path, index=False)
    else:
        out.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
