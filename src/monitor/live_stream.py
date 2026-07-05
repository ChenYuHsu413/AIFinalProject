"""Live sensor-feed simulation for Live Monitor v3.

Generates telemetry on the fly with the vendored generator, cycling through the
30 fault scenarios forever, runs the early-warning model per frame, and yields
Server-Sent Events (SSE) at real-time cadence. Feels like a sensor pushing data
now — the frontend has no foreknowledge of future frames.

Synthetic simulation, not a real sensor. Same feature schema (schema.py) and
model as the replay packs, so replay and live agree.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any, AsyncIterator, Dict, List

import joblib
import numpy as np

from src.monitor.schema import (
    RAW_HZ,
    SCENARIO_ROWS,
    SEED,
    STAGE_TO_INT,
    SUBSYSTEMS,
    radar_severity,
    scenario_baseline,
    window_features,
)
from src.monitor.servo_v3_generator import SCENARIOS, generate_servo_data
from src.utils.paths import load_config, resolve

_STATE: Dict[str, Any] = {"model": None}


def _load():
    """Lazily load the classifier bundle (once per process)."""
    if _STATE["model"] is None:
        model_path = resolve(load_config()["monitor"]["model"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"監控模型未建立（{model_path.name}）。先跑 python -m src.monitor.build_servo_v3。")
        _STATE["model"] = joblib.load(model_path)
    return _STATE["model"]


DEFAULT_STAGE_FRACS = (0.53, 0.73, 0.88)


def _jittered_stage_fracs(rng: "random.Random") -> tuple[float, float, float]:
    """Randomize warning/alarm/trip onsets so fault evolution (and thus the
    model's lead time) varies realistically instead of being a fixed constant."""
    warn = rng.uniform(0.44, 0.60)
    alarm = warn + rng.uniform(0.12, 0.28)
    trip = min(alarm + rng.uniform(0.07, 0.14), 0.97)
    return (round(warn, 3), round(alarm, 3), round(trip, 3))


def _prepare_scenario(scenario_id: int, out_hz: int, win: int,
                      stage_fracs: tuple[float, float, float] = DEFAULT_STAGE_FRACS
                      ) -> List[Dict[str, Any]]:
    """Generate one scenario and pre-compute every streamed frame (fast, off-loop)."""
    model = _load()
    df = generate_servo_data(rows=SCENARIO_ROWS, scenario_id=scenario_id,
                             sampling_hz=RAW_HZ, seed=SEED, stage_fracs=stage_fracs)
    step = RAW_HZ // out_hz
    idx = np.arange(0, len(df), step)

    feats = window_features(df, win).iloc[idx][model["feat_cols"]].to_numpy()
    pred_prob = model["warn_clf"].predict_proba(feats)[:, 1]
    pred_cat = model["cat_clf"].predict(feats)
    radar = radar_severity(df, scenario_baseline(df)).iloc[idx].reset_index(drop=True)

    subs = list(SUBSYSTEMS.keys())
    stage = df["fault_stage"].map(STAGE_TO_INT).to_numpy()[idx]
    t = df["time_s"].to_numpy()[idx]
    health = df["health_index"].to_numpy()[idx]
    rul = df["rul_sec"].to_numpy()[idx]
    warn = df["warning"].to_numpy()[idx]
    alarm = df["alarm"].to_numpy()[idx]
    trip = df["trip"].to_numpy()[idx]
    name = str(df["scenario_name"].iloc[0])
    cat = str(df["fault_category"].iloc[0])
    root = str(df["root_cause"].iloc[-1])
    equip = str(df["equipment_profile"].iloc[0])  # v4.1 per-scenario equipment

    frames: List[Dict[str, Any]] = []
    for i in range(len(idx)):
        fr: Dict[str, Any] = {
            "sid": scenario_id, "seq": i, "t": round(float(t[i]), 3),
            "scenario_name": name, "fault_category": cat, "root_cause": root,
            "equipment_profile": equip,
            "stage_int": int(stage[i]), "health": round(float(health[i]), 1),
            "rul": round(float(rul[i]), 2),
            "warning": int(warn[i]), "alarm": int(alarm[i]), "trip": int(trip[i]),
            "pred_prob": round(float(pred_prob[i]), 4), "pred_cat": str(pred_cat[i]),
        }
        for s in subs:
            fr[s] = round(float(radar[s].iloc[i]), 3)
        frames.append(fr)
    return frames


async def stream_frames(speed: float = 1.0, out_hz: int = 20) -> AsyncIterator[str]:
    """Infinite SSE feed: inject a RANDOM fault scenario each round.

    Scenarios are picked at random (no immediate repeat) so the sequence of
    injected faults is unpredictable. Each connection gets its own random order.
    The per-scenario data is still deterministic (fixed generator seed).
    """
    speed = float(np.clip(speed, 0.25, 16.0))
    out_hz = int(out_hz) if out_hz in (10, 20, 40) else 20
    win = int(RAW_HZ * 0.3)
    n = len(SCENARIOS)
    global_t = 0.0
    delay = 1.0 / (out_hz * speed)
    rng = random.Random()

    def pick(prev: int | None) -> int:
        sid = rng.randint(1, n)
        while n > 1 and sid == prev:
            sid = rng.randint(1, n)
        return sid

    # Prefetch the NEXT scenario while streaming the current one, so the
    # ~300 ms generation cost doesn't stall the feed at each scenario boundary.
    # Each injection gets jittered fault timing -> the lead time varies per fault.
    current = pick(None)
    frames = await asyncio.to_thread(
        _prepare_scenario, current, out_hz, win, _jittered_stage_fracs(rng))
    next_task: asyncio.Task | None = None
    try:
        while True:
            nxt = pick(current)
            next_task = asyncio.create_task(
                asyncio.to_thread(_prepare_scenario, nxt, out_hz, win,
                                  _jittered_stage_fracs(rng)))
            for k, fr in enumerate(frames):
                fr["global_t"] = round(global_t, 2)
                fr["new_scenario"] = k == 0
                global_t += 1.0 / out_hz
                yield f"data: {json.dumps(fr, separators=(',', ':'))}\n\n"
                await asyncio.sleep(delay)
            frames = await next_task
            next_task = None
            current = nxt
    finally:
        # Client disconnect raises GeneratorExit at the yield above; cancel the
        # in-flight prefetch so it isn't orphaned ("Task was destroyed but it is
        # pending"). The underlying thread still finishes (~300 ms) but the task
        # reference is released instead of leaking across connect/disconnect churn.
        if next_task is not None:
            next_task.cancel()
