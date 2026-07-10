#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task C — Module Servo replay receiver + window aggregator + live inference.

Connects to the publisher's SSE feed (``scripts/servo_replay_publisher.py``),
which sends one raw ``RAW_COLUMNS`` row per ``data:`` frame, and inserts a
sliding-window aggregation layer between "receive" and "display":

  1. Accumulate raw rows in a sliding window of ``window.length_s`` (W) DATA
     seconds, sliding every ``window.step_s`` (S) seconds. Window time uses the
     row ``time`` column via a MONOTONIC stream clock (so it survives the
     per-run ``time`` reset at FMCRD run boundaries).
  2. For each window, compute the 21-dim ``full`` features EXACTLY as training
     does — by REUSING ``servo_features.aggregate_run`` (the same statistics
     ``build_feature_table`` uses per run), never re-implementing them here, so
     there is no feature-definition drift.
  3. Send the features to ``predict_servo`` and print the structured
     ``predicted_health_state`` + ``degradation_score`` over time.

Full dashboard is deferred to S2 — this stage just prints the health evolution.

Run (publisher must be up first)::

    python scripts/servo_replay_publisher.py            # terminal 1
    python scripts/servo_replay_consumer.py             # terminal 2
    python scripts/servo_replay_consumer.py --url http://127.0.0.1:8008/servo/stream --mode replay
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.servo_features import RAW_COLUMNS, aggregate_run  # noqa: E402
from src.models.servo_predict import predict_servo  # noqa: E402
from src.utils.paths import load_config  # noqa: E402

_ZH = {"LN": "健康", "LO": "輕度退化", "MED": "中度退化", "HI": "重度退化"}


def _sse_rows(url: str):
    """Yield parsed JSON payloads from an SSE stream (stdlib, no deps)."""
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
            if line.startswith("data:"):
                yield json.loads(line[5:].strip())


class Windower:
    """Sliding window over a monotonic stream clock derived from row ``time``."""

    def __init__(self, w_s: float, s_s: float):
        self.w_s, self.s_s = w_s, s_s
        self.buf: Deque[Tuple[float, Dict]] = deque()
        self.stream_t = 0.0
        self.prev_time: float | None = None
        self.nominal_dt = None
        self._dt_samples = []
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

    def push(self, rec: Dict):
        """Add a row; return a window snapshot (list of recs) when one is due."""
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


def _window_prediction(recs) -> Dict:
    df = pd.DataFrame(recs, columns=RAW_COLUMNS)
    feats = aggregate_run(df)              # SAME stats as build_feature_table
    out = predict_servo(feats)             # reference model (read-only)
    labels = df["ylabel"].astype(str)
    out["_true"] = labels.mode().iloc[0] if not labels.mode().empty else "?"
    out["_rows"] = len(df)
    return out


def run(url: str, w_s: float, s_s: float, max_windows: int | None) -> None:
    print(f"[consumer] connect {url}")
    print(f"[consumer] window W={w_s}s  step S={s_s}s  "
          f"(features reuse aggregate_run; model is read-only)\n")
    win = Windower(w_s, s_s)
    n = 0
    header = f"{'win':>4} {'stream_t':>9} {'rows':>5}  {'true':>5} -> {'pred':>4}  {'conf':>5}  {'DV':>6}  {'health':>6}  risk"
    print(header)
    print("-" * len(header))
    try:
        for rec in _sse_rows(url):
            for recs in win.push(rec):
                if not recs:
                    continue
                out = _window_prediction(recs)
                n += 1
                true = out["_true"]
                fake = true == "FAKE"
                tag = "FAKE(invalid)" if fake else f"{true}({_ZH.get(true, '?')})"
                print(f"{n:>4} {win.stream_t:>9.3f} {out['_rows']:>5}  "
                      f"{tag:>5} -> {out['predicted_health_state']:>4}  "
                      f"{out['model_confidence']:>5.2f}  "
                      f"{out['degradation_score']:>6.3f}  "
                      f"{out['health_score']:>6.1f}  {out['risk_level']}"
                      + ("   ⚠ 假數據，預測無效" if fake else ""))
                if max_windows and n >= max_windows:
                    print("\n[consumer] reached --max-windows, stopping.")
                    return
    except KeyboardInterrupt:
        print("\n[consumer] interrupted.")
    except urllib.error.URLError as e:
        raise SystemExit(f"[consumer] 連不上發布端 {url}：{e}\n"
                         "請先啟動：python scripts/servo_replay_publisher.py")
    print(f"\n[consumer] stream ended — {n} windows.")


def main() -> None:
    cfg = load_config()["servo_replay"]
    win_cfg = cfg["window"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8008/servo/stream")
    ap.add_argument("--mode", choices=["replay", "fake"], default=None,
                    help="覆寫發布端預設模式（附加 ?mode=）")
    ap.add_argument("--window-s", type=float, default=float(win_cfg["length_s"]))
    ap.add_argument("--step-s", type=float, default=float(win_cfg["step_s"]))
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    url = args.url + (f"?mode={args.mode}" if args.mode else "")
    run(url, args.window_s, args.step_s, args.max_windows)


if __name__ == "__main__":
    main()
