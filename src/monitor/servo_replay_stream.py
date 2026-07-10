"""Task B — FMCRD replay publisher core (Module Servo S1 live stream).

Reuses the Live Monitor SSE skeleton (see ``src/monitor/live_stream.py``): an
async generator yields Server-Sent-Events frames ``data: {json}\\n\\n`` at a
configurable cadence, wrapped by a ``StreamingResponse`` in the publisher app
(``scripts/servo_replay_publisher.py``). The message schema is exactly
``servo_features.RAW_COLUMNS`` — one raw per-timestep row per frame — so the
receiver can window the rows into the SAME features the reference model trained
on (see ``scripts/servo_replay_consumer.py``).

Two modes:
  * ``replay`` (default, used for the demo): stream the real FMCRD replay
    segments (``data/demo/replay/``, produced by
    ``scripts/extract_replay_segments.py``) in ``order`` — LN -> LO -> HI — so
    the receiver watches the health state evolve across the degradation journey.
  * ``fake``: emit synthetic RAW_COLUMNS rows for PIPELINE CONNECTIVITY TESTING
    ONLY. This data is NOT drawn from the model's training distribution, so any
    prediction made on it is INVALID. The demo never uses this mode.
"""
from __future__ import annotations

import asyncio
import json
import math
from typing import AsyncIterator, Dict, Iterator, List

import numpy as np
import pandas as pd

from src.features.servo_features import RAW_COLUMNS
from src.utils.paths import load_config, resolve


def _replay_config() -> Dict:
    return load_config()["servo_replay"]


# ---------------------------------------------------------------------------
# Row sources — each yields plain dicts keyed by RAW_COLUMNS.
# ---------------------------------------------------------------------------
def iter_replay_rows(order: List[str] | None = None,
                     loop: bool | None = None) -> Iterator[Dict]:
    """Yield real FMCRD replay rows, segment by segment, in ``order``.

    Reads the manifest written by ``extract_replay_segments.py`` and streams each
    segment CSV row by row. Concatenated across segments this replays the
    LN -> LO -> HI degradation journey.
    """
    cfg = _replay_config()
    order = order or cfg["order"]
    loop = cfg.get("loop", False) if loop is None else loop
    manifest_path = resolve(cfg["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"找不到 replay manifest：{manifest_path}\n"
            "請先執行：python scripts/extract_replay_segments.py（需要真實 FMCRD zip）。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_seg = {s["segment"]: s for s in manifest["segments"]}
    replay_dir = manifest_path.parent

    def one_pass() -> Iterator[Dict]:
        for seg_name in order:
            if seg_name not in by_seg:
                raise KeyError(f"manifest 缺少段落 {seg_name}（有：{list(by_seg)}）")
            df = pd.read_csv(replay_dir / by_seg[seg_name]["file"])
            # Emit exactly RAW_COLUMNS in order; keep native dtypes (ylabel is str).
            df = df[RAW_COLUMNS]
            for rec in df.to_dict(orient="records"):
                yield rec

    while True:
        yield from one_pass()
        if not loop:
            return


def iter_fake_rows() -> Iterator[Dict]:
    """Yield synthetic RAW_COLUMNS rows — PIPELINE TEST ONLY, predictions INVALID.

    Deliberately simple sinusoids + noise; NOT physically calibrated and NOT in
    the reference model's training distribution. The ``ylabel`` is tagged "FAKE"
    so the receiver can flag that any prediction on this stream is meaningless.
    """
    rng = np.random.default_rng(0)
    t = 0.0
    dt = 1.0 / 1000.0
    run_index = 0
    i = 0
    while True:
        # New "run" every 1000 rows so the receiver's per-run logic still ticks.
        if i % 1000 == 0:
            run_index += 1
        phase = 2 * math.pi * 1.0 * t
        speed = 100.0 * math.sin(phase) + rng.normal(0, 2)
        torque = 5.0 * math.sin(phase + 0.5) + rng.normal(0, 1)
        cur = 2.0 * math.sin(phase)
        demand = 500.0 + 10.0 * math.sin(phase)
        actual = demand + rng.normal(0, 0.5)
        yield {
            "time": round(t, 5), "DV": 0.0,
            "rod_demand_pos": demand, "rod_actual_pos": actual,
            "torque": torque, "rotor_speed": speed,
            "i_3p_a": cur, "i_3p_b": cur * -0.5, "i_3p_c": cur * -0.5,
            "direct": rng.normal(0, 0.01), "quadrature": cur,
            "run_index": run_index, "transitions": 2,
            "del_pos": actual - 514.0, "ylabel": "FAKE",
        }
        t += dt
        i += 1


# ---------------------------------------------------------------------------
# SSE async generator (mirrors live_stream.stream_frames).
# ---------------------------------------------------------------------------
async def stream_replay(mode: str = "replay", emit_hz: float | None = None,
                        speed: float = 1.0, order: List[str] | None = None,
                        loop: bool | None = None) -> AsyncIterator[str]:
    """Infinite/one-shot SSE feed of RAW_COLUMNS rows.

    ``emit_hz`` sets the wall-clock publish cadence (messages/sec); faster than
    the data's real rate = accelerated replay (the default). ``speed`` multiplies
    it. ``mode`` selects the real FMCRD replay or the synthetic test source.
    """
    cfg = _replay_config()
    emit_hz = float(cfg.get("emit_hz", 1000)) if emit_hz is None else float(emit_hz)
    emit_hz = float(np.clip(emit_hz, 1.0, 50_000.0))
    speed = float(np.clip(speed, 0.05, 100.0))
    delay = 1.0 / (emit_hz * speed)

    if mode == "fake":
        rows: Iterator[Dict] = iter_fake_rows()
    elif mode == "replay":
        rows = iter_replay_rows(order=order, loop=loop)
    else:
        raise ValueError(f"未知 mode：{mode}（可用 replay / fake）")

    for rec in rows:
        yield f"data: {json.dumps(rec, separators=(',', ':'))}\n\n"
        await asyncio.sleep(delay)
