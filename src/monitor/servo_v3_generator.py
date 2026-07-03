#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servo AI Dataset v3.0 Enterprise Generator

用途：
1. 自行生成 Mitsubishi Servo 風格 PLC / Drive Log 模擬資料
2. 可指定 rows，例如 1000、15000、450000、1000000
3. 可指定 scenario 故障模式
4. 可輸出 CSV 或 Parquet
5. 適用 ML / DL / LSTM / Transformer / AutoEncoder / XGBoost / LightGBM / RUL / Digital Twin

範例：
    python servo_ai_dataset_v3_generator.py --rows 1000 --scenario 2 --output servo_1000.csv
    python servo_ai_dataset_v3_generator.py --rows 15000 --scenario all --output_dir output
    python servo_ai_dataset_v3_generator.py --rows 1000000 --scenario mixed --output servo_1M.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


SCENARIOS = [
    ("Healthy Baseline", "NORMAL", "NONE", "Normal production motion without degradation"),
    ("Motor Over Temperature", "TEMPERATURE", "MITSUBISHI_SIM_AL.050", "Motor winding temperature rises progressively"),
    ("Drive Over Temperature", "TEMPERATURE", "MITSUBISHI_SIM_AL.051", "Servo amplifier heat sink temperature rises"),
    ("Encoder Drift", "ENCODER", "MITSUBISHI_SIM_AL.016", "Encoder absolute position gradually drifts"),
    ("Encoder Noise", "ENCODER", "MITSUBISHI_SIM_AL.020", "Encoder signal contains jitter and count noise"),
    ("Encoder Signal Loss", "ENCODER", "MITSUBISHI_SIM_AL.021", "Intermittent loss of encoder feedback"),
    ("Excessive Following Error", "POSITION", "MITSUBISHI_SIM_AL.052", "Actual position cannot follow command"),
    ("Overspeed", "SPEED", "MITSUBISHI_SIM_AL.031", "Actual motor speed exceeds safe threshold"),
    ("Acceleration Overshoot", "MOTION", "MITSUBISHI_SIM_AL.032", "Acceleration spike during motion profile"),
    ("Deceleration Failure", "MOTION", "MITSUBISHI_SIM_AL.033", "Deceleration tracking failure"),
    ("Over Current", "CURRENT", "MITSUBISHI_SIM_AL.010", "Phase current exceeds allowed range"),
    ("Torque Saturation", "TORQUE", "MITSUBISHI_SIM_AL.024", "Torque command saturates under load"),
    ("Mechanical Jam", "MECHANICAL", "MITSUBISHI_SIM_AL.040", "Axis blocked by mechanical interference"),
    ("Bearing Wear", "MECHANICAL", "MITSUBISHI_SIM_AL.041", "Bearing degradation increases vibration"),
    ("Bearing Lubrication Failure", "MECHANICAL", "MITSUBISHI_SIM_AL.042", "Friction increases due to lubrication failure"),
    ("Rotor Imbalance", "MECHANICAL", "MITSUBISHI_SIM_AL.043", "Imbalance creates periodic vibration"),
    ("Coupling Misalignment", "MECHANICAL", "MITSUBISHI_SIM_AL.044", "Coupling misalignment creates torque ripple"),
    ("Ball Screw Wear", "MECHANICAL", "MITSUBISHI_SIM_AL.045", "Ball screw wear increases friction and error"),
    ("Backlash", "MECHANICAL", "MITSUBISHI_SIM_AL.046", "Mechanical play during direction reversal"),
    ("High Vibration", "VIBRATION", "MITSUBISHI_SIM_AL.047", "Axis vibration exceeds threshold"),
    ("Power Supply Fluctuation", "POWER", "MITSUBISHI_SIM_AL.013", "Input voltage and DC bus unstable"),
    ("Voltage Drop", "POWER", "MITSUBISHI_SIM_AL.012", "DC bus under-voltage during load"),
    ("Communication Timeout", "COMMUNICATION", "MITSUBISHI_SIM_AL.086", "PLC or controller command timeout"),
    ("EtherCAT Packet Loss", "COMMUNICATION", "MITSUBISHI_SIM_AL.087", "Network packet loss and sync jitter"),
    ("Servo Gain Instability", "CONTROL", "MITSUBISHI_SIM_AL.037", "Servo gain too high causing oscillation"),
    ("Mechanical Resonance", "CONTROL", "MITSUBISHI_SIM_AL.038", "Resonance frequency excited by motion"),
    ("Brake Failure", "SAFETY", "MITSUBISHI_SIM_AL.060", "Holding brake abnormal response"),
    ("Emergency Stop", "SAFETY", "MITSUBISHI_SIM_AL.061", "Emergency stop signal activated"),
    ("Combined Fault", "MULTI_FAULT", "MITSUBISHI_SIM_AL.090", "Overload with vibration and temperature rise"),
    ("Progressive Failure Shutdown", "MULTI_FAULT", "MITSUBISHI_SIM_AL.091", "Progressive failure ending in servo trip"),
]


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def make_stages(t: np.ndarray, scenario_id: int, duration_sec: float):
    if scenario_id == 1:
        normal = np.array(["normal"] * len(t), dtype=object)
        zero = np.zeros(len(t), dtype=np.int8)
        return normal, zero, zero, zero, np.zeros(len(t), dtype=np.float32)

    fault_start = duration_sec * 0.30
    warning_t = duration_sec * 0.53
    alarm_t = duration_sec * 0.73
    trip_t = duration_sec * 0.88

    severity = np.clip((t - fault_start) / max(duration_sec - fault_start, 1e-6), 0, 1).astype(np.float32)
    stage = np.where(
        t < fault_start, "normal",
        np.where(t < warning_t, "early_degradation",
        np.where(t < alarm_t, "warning",
        np.where(t < trip_t, "alarm", "trip")))
    )
    warning = (t >= warning_t).astype(np.int8)
    alarm = (t >= alarm_t).astype(np.int8)
    trip = (t >= trip_t).astype(np.int8)
    return stage, warning, alarm, trip, severity


def generate_servo_data(
    rows: int = 1000,
    scenario_id: int = 2,
    sampling_hz: int = 1000,
    seed: int = 42,
    global_start_s: float = 0.0,
) -> pd.DataFrame:
    """
    產生單一 Scenario 的 Servo PLC/Drive Log.

    Args:
        rows: 要產生的筆數，例如 1000、15000、1000000
        scenario_id: 1~30
        sampling_hz: 取樣頻率，預設 1000 Hz
        seed: 隨機種子
        global_start_s: 串接資料時的起始時間

    Returns:
        pandas.DataFrame
    """
    if scenario_id < 1 or scenario_id > len(SCENARIOS):
        raise ValueError("scenario_id must be 1~30")

    rng = np.random.default_rng(seed + scenario_id)
    name, category, alarm_code, root_cause = SCENARIOS[scenario_id - 1]

    dt = 1.0 / sampling_hz
    duration_sec = rows / sampling_hz
    t = np.arange(rows, dtype=np.float32) * dt
    global_time = global_start_s + t

    stage, warning, alarm, trip, sev = make_stages(t, scenario_id, duration_sec)
    sev2 = sev ** 1.8

    cycle = 1.5
    phase = (t % cycle) / cycle

    pos_cmd = 100000 + scenario_id * 16000 + 4200 * np.sin(2 * np.pi * phase) + 520 * t
    cmd_rate = np.gradient(pos_cmd, dt)
    speed_cmd = cmd_rate / (np.max(np.abs(cmd_rate)) + 1e-9) * (1600 + (scenario_id % 7) * 120)
    accel_cmd = np.gradient(speed_cmd, dt)
    jerk_cmd = np.gradient(accel_cmd, dt)

    pos_error = rng.normal(0, 3.8, rows)
    actual_speed = speed_cmd + rng.normal(0, 5.5, rows)
    actual_accel = np.gradient(actual_speed, dt)
    encoder = (pos_cmd + pos_error) * 10 + 52312 + rng.normal(0, 1.8, rows)

    torque_cmd = 0.28 + 0.23 * np.abs(accel_cmd) / (np.max(np.abs(accel_cmd)) + 1e-9) + rng.normal(0, 0.012, rows)
    actual_torque = torque_cmd + rng.normal(0, 0.022, rows)
    current = 0.9 + 1.95 * np.abs(actual_torque) + rng.normal(0, 0.035, rows)

    voltage = 220 + rng.normal(0, 0.55, rows)
    dc_bus = 310 + rng.normal(0, 1.1, rows)
    motor_temp = 33.5 + 0.07 * t + rng.normal(0, 0.10, rows)
    drive_temp = 37.5 + 0.055 * t + rng.normal(0, 0.10, rows)
    bearing_temp = 32.5 + 0.04 * t + rng.normal(0, 0.07, rows)
    load_pct = 18 + 35 * np.abs(actual_torque) + rng.normal(0, 1.1, rows)

    vib_x = 0.020 + rng.normal(0, 0.003, rows)
    vib_y = 0.018 + rng.normal(0, 0.003, rows)
    vib_z = 0.016 + rng.normal(0, 0.003, rows)

    net_jitter = np.abs(rng.normal(0.10, 0.025, rows))
    packet_loss = np.zeros(rows)
    plc_scan = 2.0 + rng.normal(0, 0.06, rows)
    safety_sto = np.zeros(rows, dtype=np.int8)
    brake_release = np.ones(rows, dtype=np.int8)
    phase_imbalance = rng.normal(0.0, 0.015, rows)

    # Fault injection
    if scenario_id == 2:
        motor_temp += 50 * sev2
        current += 0.7 * sev
        load_pct += 7 * sev
    elif scenario_id == 3:
        drive_temp += 45 * sev2
        dc_bus += 5 * sev * np.sin(2 * np.pi * 3 * t)
    elif scenario_id == 4:
        drift = 140 * sev2
        encoder += drift
        pos_error += drift / 10
    elif scenario_id == 5:
        noise = rng.normal(0, 95 * sev, rows)
        encoder += noise
        pos_error += noise / 10
    elif scenario_id == 6:
        mask = (t > duration_sec * 0.60) & (np.sin(2 * np.pi * 22 * t) > 0.90)
        encoder[mask] = np.nan
        pos_error += 90 * sev
    elif scenario_id == 7:
        pos_error += 190 * sev2
        actual_speed -= 100 * sev
    elif scenario_id == 8:
        actual_speed += 880 * sev2
    elif scenario_id == 9:
        actual_speed += 190 * sev * np.sin(2 * np.pi * 4 * t)
        actual_accel += 3800 * sev * np.sin(2 * np.pi * 4 * t)
    elif scenario_id == 10:
        actual_speed += np.where(phase > 0.62, 340 * sev2, 0)
    elif scenario_id == 11:
        current += 5.7 * sev2
        actual_torque += 0.85 * sev
        phase_imbalance += 0.20 * sev2
    elif scenario_id == 12:
        torque_cmd += 0.95 * sev2
        actual_torque += 1.15 * sev2
        current += 2.5 * sev2
    elif scenario_id == 13:
        jam = np.clip((t - duration_sec * 0.60) / max(duration_sec * 0.18, 1e-6), 0, 1)
        actual_speed *= (1 - 0.88 * jam)
        pos_error += 520 * jam
        current += 6.2 * jam
        actual_torque += 1.5 * jam
    elif scenario_id == 14:
        vib_x += 0.20 * sev2 + 0.055 * np.sin(2 * np.pi * 95 * t) * sev
        bearing_temp += 24 * sev2
    elif scenario_id == 15:
        current += 1.8 * sev2
        bearing_temp += 33 * sev2
        vib_y += 0.14 * sev2
    elif scenario_id == 16:
        vib_x += 0.17 * np.sin(2 * np.pi * 60 * t) * sev + 0.11 * sev
        vib_z += 0.13 * np.sin(2 * np.pi * 60 * t + 1.2) * sev
    elif scenario_id == 17:
        actual_torque += 0.38 * np.sin(2 * np.pi * 8 * t) * sev
        vib_y += 0.13 * sev2
        pos_error += 40 * np.sin(2 * np.pi * 8 * t) * sev
    elif scenario_id == 18:
        load_pct += 30 * sev2
        current += 2.4 * sev2
        pos_error += 95 * sev2
    elif scenario_id == 19:
        direction = np.sign(np.gradient(speed_cmd, dt))
        pos_error += 80 * sev * np.where(direction >= 0, 1, -1)
    elif scenario_id == 20:
        vib_x += 0.27 * sev2
        vib_y += 0.24 * sev2
        vib_z += 0.22 * sev2
    elif scenario_id == 21:
        voltage += 20 * sev * np.sin(2 * np.pi * 2.5 * t)
        dc_bus += 28 * sev * np.sin(2 * np.pi * 2.5 * t)
    elif scenario_id == 22:
        voltage -= 42 * sev2
        dc_bus -= 62 * sev2
        current += 1.4 * sev
    elif scenario_id == 23:
        net_jitter += 9.5 * sev2
        plc_scan += 11 * sev2
    elif scenario_id == 24:
        packet_loss += 20 * sev2
        net_jitter += 4.8 * sev
    elif scenario_id == 25:
        pos_error += 75 * np.sin(2 * np.pi * 16 * t) * sev
        actual_speed += 240 * np.sin(2 * np.pi * 16 * t) * sev
    elif scenario_id == 26:
        vib_x += 0.22 * np.sin(2 * np.pi * 120 * t) * sev
        actual_torque += 0.28 * np.sin(2 * np.pi * 120 * t) * sev
    elif scenario_id == 27:
        brake_release = np.where(t > duration_sec * 0.73, 0, 1)
        actual_speed *= np.where(t > duration_sec * 0.73, 0.56, 1)
        current += np.where(t > duration_sec * 0.73, 2.9 * sev, 0)
    elif scenario_id == 28:
        safety_sto = (t > duration_sec * 0.70).astype(np.int8)
        actual_speed = np.where(t > duration_sec * 0.70, actual_speed * np.exp(-(t - duration_sec * 0.70) * 3), actual_speed)
        current += np.where(t > duration_sec * 0.70, 0.9, 0)
    elif scenario_id == 29:
        current += 3.1 * sev2
        motor_temp += 30 * sev2
        vib_x += 0.18 * sev2
        pos_error += 135 * sev2
        load_pct += 20 * sev2
    elif scenario_id == 30:
        current += 4.3 * sev2
        motor_temp += 37 * sev2
        bearing_temp += 30 * sev2
        vib_x += 0.24 * sev2
        load_pct += 38 * sev2
        pos_error += 330 * sev2

    actual_pos = pos_cmd + pos_error
    actual_accel = np.gradient(actual_speed, dt)
    encoder_i = pd.Series(encoder).interpolate(limit_direction="both").to_numpy()
    encoder_vel = np.gradient(encoder_i, dt)

    anomaly = (
        np.clip(sev * 0.86 + warning * 0.04 + alarm * 0.06 + trip * 0.04 + rng.normal(0, 0.012, rows), 0, 1)
        if scenario_id != 1 else
        np.clip(rng.normal(0.018, 0.008, rows), 0, 0.07)
    )
    health = np.clip(100 - anomaly * 96 - trip * 4, 0, 100)
    trip_time = duration_sec * 0.88
    rul_sec = np.where(scenario_id == 1, 9999, np.maximum(trip_time - t, 0))
    failure_prob = np.clip(anomaly ** 1.25, 0, 1)

    vib_rms = np.sqrt(vib_x ** 2 + vib_y ** 2 + vib_z ** 2)
    power_w = voltage * current * 0.85
    bus_ripple = np.abs(np.gradient(dc_bus, dt)) / 1000
    regen_load = np.clip(7 + np.abs(accel_cmd) / (np.max(np.abs(accel_cmd)) + 1e-9) * 22 + rng.normal(0, 0.8, rows), 0, 100)

    u = current / np.sqrt(3) + rng.normal(0, 0.025, rows)
    v = current / np.sqrt(3) * (1 + phase_imbalance) + rng.normal(0, 0.025, rows)
    w = current / np.sqrt(3) * (1 - phase_imbalance) + rng.normal(0, 0.025, rows)

    df = pd.DataFrame({
        "scenario_id": scenario_id,
        "scenario_name": name,
        "time_s": t,
        "global_time_s": global_time,
        "sample_index": np.arange(rows),
        "sequence_id": f"S{scenario_id:02d}",
        "fault_category": category,
        "fault_type": "NORMAL" if scenario_id == 1 else name,
        "fault_stage": stage,
        "alarm_code": np.where(alarm == 1, alarm_code, "NONE"),
        "warning": warning,
        "alarm": alarm,
        "trip": trip,
        "pos_cmd_pulse": pos_cmd,
        "actual_pos_pulse": actual_pos,
        "position_error_pulse": pos_error,
        "encoder_count": encoder,
        "encoder_count_interpolated": encoder_i,
        "encoder_velocity_count_s": encoder_vel,
        "speed_cmd_rpm": speed_cmd,
        "actual_speed_rpm": actual_speed,
        "speed_error_rpm": actual_speed - speed_cmd,
        "accel_cmd_rpm_s": accel_cmd,
        "actual_accel_rpm_s": actual_accel,
        "accel_error_rpm_s": actual_accel - accel_cmd,
        "jerk_cmd_rpm_s2": jerk_cmd,
        "torque_cmd_nm": torque_cmd,
        "actual_torque_nm": actual_torque,
        "torque_error_nm": actual_torque - torque_cmd,
        "current_rms_a": current,
        "u_phase_current_a": u,
        "v_phase_current_a": v,
        "w_phase_current_a": w,
        "phase_current_imbalance_pct": np.abs(u - v) / (current + 1e-9) * 100,
        "voltage_v": voltage,
        "dc_bus_voltage_v": dc_bus,
        "dc_bus_ripple_v": bus_ripple,
        "regen_load_pct": regen_load,
        "power_w": power_w,
        "energy_wh": np.cumsum(power_w) * dt / 3600,
        "motor_temp_c": motor_temp,
        "drive_temp_c": drive_temp,
        "heatsink_temp_c": drive_temp + 2 + rng.normal(0, 0.1, rows),
        "bearing_temp_c": bearing_temp,
        "ambient_temp_c": 25 + 0.35 * np.sin(2 * np.pi * t / max(duration_sec, 1e-6)) + rng.normal(0, 0.04, rows),
        "cabinet_temp_c": 29 + 0.2 * np.sin(2 * np.pi * t / max(duration_sec, 1e-6)) + rng.normal(0, 0.06, rows),
        "cooling_fan_speed_rpm": 3200 + drive_temp * 18 + rng.normal(0, 20, rows),
        "load_pct": np.clip(load_pct, 0, 180),
        "load_inertia_ratio": 2.4 + scenario_id * 0.08 + rng.normal(0, 0.02, rows),
        "friction_estimate_nm": 0.04 + current * 0.014 + rng.normal(0, 0.002, rows),
        "backlash_estimate_pulse": np.clip(np.abs(pos_error) * 0.08 + (scenario_id == 19) * sev * 45, 0, 999),
        "mechanical_clearance_um": np.clip(8 + np.abs(pos_error) * 0.015 + (scenario_id in [18, 19]) * sev * 25, 0, 999),
        "brake_release_status": brake_release,
        "brake_current_a": brake_release * 0.42 + rng.normal(0, 0.01, rows),
        "vibration_x_g": vib_x,
        "vibration_y_g": vib_y,
        "vibration_z_g": vib_z,
        "vibration_rms_g": vib_rms,
        "vibration_peak_g": vib_rms * 2.8 + rng.normal(0, 0.01, rows),
        "vibration_kurtosis": 3 + 8 * sev2 + rng.normal(0, 0.08, rows),
        "fft_1x_amp": np.abs(vib_x) * 10 + rng.normal(0, 0.01, rows),
        "fft_2x_amp": np.abs(vib_y) * 8 + rng.normal(0, 0.01, rows),
        "bearing_bpfo_amp": np.abs(vib_z) * 6 + (scenario_id in [14, 15]) * sev * 1.5,
        "bearing_bpfi_amp": np.abs(vib_y) * 6 + (scenario_id in [14, 15]) * sev * 1.2,
        "network_jitter_ms": net_jitter,
        "packet_loss_pct": packet_loss,
        "ethercat_sync_error_us": np.abs(rng.normal(4, 1.2, rows)) + packet_loss * 2,
        "plc_scan_time_ms": plc_scan,
        "servo_ready": (1 - trip).astype(np.int8),
        "drive_ready": (1 - trip).astype(np.int8),
        "in_position": (np.abs(pos_error) < 35).astype(np.int8),
        "motion_complete": ((phase > 0.95) & (np.abs(actual_speed) < 120)).astype(np.int8),
        "sto_status": safety_sto,
        "safety_relay_closed": (1 - safety_sto).astype(np.int8),
        "control_mode": "position",
        "operation_state": np.where(trip == 1, "TRIP", np.where(alarm == 1, "ALARM", np.where(warning == 1, "WARNING", "RUN"))),
        "gain_p": 120 + scenario_id * 0.4 + rng.normal(0, 0.18, rows),
        "gain_i": 28 + scenario_id * 0.05 + rng.normal(0, 0.04, rows),
        "gain_d": 0.8 + scenario_id * 0.005 + rng.normal(0, 0.003, rows),
        "notch_filter_hz": 120 + (scenario_id % 6) * 20,
        "adaptive_tuning_level": np.clip(50 + sev * 25 + rng.normal(0, 0.5, rows), 0, 100),
        "digital_twin_pos_residual": pos_error + rng.normal(0, 1.5, rows),
        "digital_twin_speed_residual": actual_speed - speed_cmd + rng.normal(0, 2.0, rows),
        "digital_twin_torque_residual": actual_torque - torque_cmd + rng.normal(0, 0.008, rows),
        "anomaly_score": anomaly,
        "failure_probability": failure_prob,
        "health_index": health,
        "bearing_health_index": np.clip(100 - (bearing_temp - 33) * 1.0 - vib_rms * 65, 0, 100),
        "rul_sec": rul_sec,
        "rul_cycle": np.where(scenario_id == 1, 9999, np.maximum(rul_sec / 1.5, 0)),
        "maintenance_required": (alarm | trip).astype(np.int8),
        "maintenance_action": np.where(scenario_id == 1, "No action", np.where(alarm == 1, "Stop and inspect axis", "Monitor trend")),
        "root_cause": root_cause,
        "line_id": "TSMC_ASE_AUTO_LINE_SIM_V3",
        "station_id": f"STATION_{scenario_id % 6 + 1:02d}",
        "axis_name": f"X{scenario_id % 4 + 1}_SERVO_AXIS",
        "drive_model": "Mitsubishi MR-J5-G simulated",
        "motor_model": "HK-KT / HG-KR simulated",
        "recipe_id": f"RECIPE_{scenario_id % 5 + 1:02d}",
        "lot_id": f"LOT_SIM_{2000 + scenario_id:04d}",
    })

    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].astype("float32").round(5)
    return df


def generate_all_scenarios(rows_per_scenario: int, output_dir: Path, file_format: str, sampling_hz: int, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    global_start = 0.0

    for scenario_id, (name, category, alarm_code, root_cause) in enumerate(SCENARIOS, start=1):
        df = generate_servo_data(
            rows=rows_per_scenario,
            scenario_id=scenario_id,
            sampling_hz=sampling_hz,
            seed=seed,
            global_start_s=global_start,
        )

        filename = f"scenario_{scenario_id:02d}_{safe_name(name)}.{file_format}"
        path = output_dir / filename
        if file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError("file_format must be csv or parquet")

        index_rows.append({
            "scenario_id": scenario_id,
            "scenario_name": name,
            "fault_category": category,
            "alarm_code": alarm_code,
            "rows": len(df),
            "sampling_hz": sampling_hz,
            "duration_sec": rows_per_scenario / sampling_hz,
            "global_start_s": global_start,
            "global_end_s": global_start + rows_per_scenario / sampling_hz - 1 / sampling_hz,
            "file_name": filename,
            "root_cause": root_cause,
        })
        global_start += rows_per_scenario / sampling_hz

    pd.DataFrame(index_rows).to_csv(output_dir / "scenario_index_generated.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Servo AI Dataset v3.0 Enterprise Generator")
    parser.add_argument("--rows", type=int, default=1000, help="rows per dataset or per scenario, default=1000")
    parser.add_argument("--scenario", default="2", help="1~30, all, or mixed. default=2")
    parser.add_argument("--sampling_hz", type=int, default=1000, help="sampling frequency, default=1000")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output", default="servo_generated_1000.csv", help="single output file path")
    parser.add_argument("--output_dir", default="servo_generated_output", help="output directory for all scenarios")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="csv or parquet")
    args = parser.parse_args()

    if args.scenario == "all":
        generate_all_scenarios(
            rows_per_scenario=args.rows,
            output_dir=Path(args.output_dir),
            file_format=args.format,
            sampling_hz=args.sampling_hz,
            seed=args.seed,
        )
        print(f"Generated all 30 scenarios in: {args.output_dir}")
        return

    if args.scenario == "mixed":
        parts = []
        rows_left = args.rows
        global_start = 0.0
        scenario_id = 1
        while rows_left > 0:
            chunk_rows = min(rows_left, max(1000, args.sampling_hz * 15))
            df = generate_servo_data(
                rows=chunk_rows,
                scenario_id=scenario_id,
                sampling_hz=args.sampling_hz,
                seed=args.seed + scenario_id,
                global_start_s=global_start,
            )
            parts.append(df)
            rows_left -= chunk_rows
            global_start += chunk_rows / args.sampling_hz
            scenario_id = scenario_id + 1 if scenario_id < 30 else 1

        out = pd.concat(parts, ignore_index=True)
    else:
        out = generate_servo_data(
            rows=args.rows,
            scenario_id=int(args.scenario),
            sampling_hz=args.sampling_hz,
            seed=args.seed,
            global_start_s=0,
        )

    output_path = Path(args.output)
    if args.format == "csv":
        out.to_csv(output_path, index=False)
    else:
        out.to_parquet(output_path, index=False)

    print(f"Generated: {output_path}")
    print(f"Rows: {len(out):,}")
    print(f"Columns: {len(out.columns):,}")
    print(f"Scenario: {args.scenario}")


if __name__ == "__main__":
    main()
