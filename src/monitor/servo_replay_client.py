"""Shared receiver-side pipeline for the Servo replay stream (S1 → S2).

Single source of truth for "SSE frame -> sliding window -> reference-model
prediction", reused by the S1 CLI consumer (``scripts/servo_replay_consumer.py``),
the S2 alert engine (``src/monitor/alert_engine.py``) and the S2 Streamlit live
monitor page. Windowing granularity is fixed to one FMCRD run cycle (S1 OOD
lesson; W/S locked in ``config.yaml::servo_replay.window``).

Features are computed by REUSING ``servo_features.aggregate_run`` (the same
statistics ``build_feature_table`` uses per run) — never re-implemented here — so
there is no feature-definition drift, and fed to ``predict_servo`` (read-only).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, Iterable, Iterator, Tuple

import numpy as np
import pandas as pd

from src.features.servo_features import RAW_COLUMNS, aggregate_run
from src.models.servo_predict import predict_servo

HEALTH_ZH = {"LN": "健康", "LO": "輕度退化", "MED": "中度退化", "HI": "重度退化"}


def sse_rows(url: str) -> Iterator[Dict]:
    """Yield parsed JSON payloads from an SSE stream (stdlib, no deps)."""
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
            if line.startswith("data:"):
                yield json.loads(line[5:].strip())


class Windower:
    """Sliding window over a monotonic stream clock derived from row ``time``.

    The clock survives the per-run ``time`` reset at FMCRD run boundaries
    (``delta<=0`` -> use the nominal dt), so windowing stays correct across the
    LN→LO→HI segment joins.
    """

    def __init__(self, w_s: float, s_s: float):
        self.w_s, self.s_s = w_s, s_s
        self.buf: Deque[Tuple[float, Dict]] = deque()
        self.stream_t = 0.0
        self.prev_time: float | None = None
        self.nominal_dt: float | None = None
        self._dt_samples: list[float] = []
        self.next_emit = w_s  # first window once we have W seconds of data

    def _update_clock(self, t: float) -> None:
        if self.prev_time is None:
            delta = 0.0
        else:
            delta = t - self.prev_time
            if delta <= 0:  # run/segment boundary: time reset -> use nominal dt
                delta = self.nominal_dt or 0.0
            elif len(self._dt_samples) < 200:
                self._dt_samples.append(delta)
                self.nominal_dt = float(np.median(self._dt_samples))
        self.prev_time = t
        self.stream_t += delta

    def push(self, rec: Dict) -> list[list[Dict]]:
        """Add a row; return window snapshots (lists of recs) that became due."""
        self._update_clock(float(rec["time"]))
        self.buf.append((self.stream_t, rec))
        lo = self.stream_t - self.w_s
        while self.buf and self.buf[0][0] < lo:
            self.buf.popleft()
        emitted = []
        while self.stream_t >= self.next_emit:
            emitted.append([r for _, r in self.buf])
            self.next_emit += self.s_s
        return emitted


def window_prediction(recs: Iterable[Dict], stream_t: float) -> Dict[str, Any]:
    """Aggregate one window's raw rows into the reference-model structured output.

    Adds bookkeeping fields prefixed with ``_``: ``_true`` (the window's true
    ``ylabel`` mode, for display/validation), ``_rows``, ``_stream_t``.
    """
    df = pd.DataFrame(list(recs), columns=RAW_COLUMNS)
    feats = aggregate_run(df)                 # SAME stats as build_feature_table
    out = predict_servo(feats)                # reference model (read-only)
    labels = df["ylabel"].astype(str)
    out["_true"] = labels.mode().iloc[0] if not labels.mode().empty else "?"
    out["_rows"] = int(len(df))
    out["_stream_t"] = round(float(stream_t), 3)
    out["_features"] = feats                  # aggregated features (drift detector reads these)
    return out


def iter_window_predictions_from_rows(rows: Iterable[Dict], w_s: float, s_s: float
                                      ) -> Iterator[Dict[str, Any]]:
    """Window a raw-row iterable and yield one structured prediction per window.

    Server-independent (used by tests and any in-process source)."""
    win = Windower(w_s, s_s)
    for rec in rows:
        for recs in win.push(rec):
            if recs:
                yield window_prediction(recs, win.stream_t)


def iter_window_predictions(url: str, w_s: float, s_s: float
                            ) -> Iterator[Dict[str, Any]]:
    """Connect to the publisher's SSE feed and yield per-window predictions."""
    yield from iter_window_predictions_from_rows(sse_rows(url), w_s, s_s)
