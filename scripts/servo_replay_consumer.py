#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task C (S1) — Module Servo replay receiver + window aggregator + live inference.

Thin CLI over the shared receiver pipeline (``src/monitor/servo_replay_client.py``):
connects to the publisher's SSE feed, windows the raw ``RAW_COLUMNS`` rows into
the SAME features training uses (``aggregate_run``), runs ``predict_servo`` and
prints the structured ``predicted_health_state`` + ``degradation_score`` over time.
The window/predict logic is shared with the S2 alert engine and live monitor page.

Run (publisher must be up first)::

    python scripts/servo_replay_publisher.py            # terminal 1
    python scripts/servo_replay_consumer.py             # terminal 2
    python scripts/servo_replay_consumer.py --url http://127.0.0.1:8008/servo/stream --mode replay
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitor.servo_replay_client import (  # noqa: E402
    HEALTH_ZH, iter_window_predictions,
)
from src.utils.paths import load_config  # noqa: E402


def run(url: str, w_s: float, s_s: float, max_windows: int | None) -> None:
    print(f"[consumer] connect {url}")
    print(f"[consumer] window W={w_s}s  step S={s_s}s  "
          f"(features reuse aggregate_run; model is read-only)\n")
    header = f"{'win':>4} {'stream_t':>9} {'rows':>5}  {'true':>5} -> {'pred':>4}  {'conf':>5}  {'DV':>6}  {'health':>6}  risk"
    print(header)
    print("-" * len(header))
    n = 0
    try:
        for out in iter_window_predictions(url, w_s, s_s):
            n += 1
            true = out["_true"]
            fake = true == "FAKE"
            tag = "FAKE(invalid)" if fake else f"{true}({HEALTH_ZH.get(true, '?')})"
            print(f"{n:>4} {out['_stream_t']:>9.3f} {out['_rows']:>5}  "
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
    win_cfg = load_config()["servo_replay"]["window"]
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
