#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task B — Module Servo FMCRD replay publisher (SSE endpoint).

Serves the replay stream over the SAME transport as the Live Monitor prototype:
a FastAPI ``StreamingResponse`` of Server-Sent-Events (``text/event-stream``),
one ``data: {json}`` frame per raw RAW_COLUMNS row (see
``src/monitor/servo_replay_stream.py``). Kept as a tiny standalone app so the
demo starts fast without loading the full backend.

    GET /servo/stream?mode=replay&speed=1        # real FMCRD LN->LO->HI (default)
    GET /servo/stream?mode=fake                  # synthetic — PIPELINE TEST ONLY,
                                                 # predictions INVALID (not training-dist)
    GET /healthz

Run::

    python scripts/servo_replay_publisher.py                 # replay @ config emit_hz
    python scripts/servo_replay_publisher.py --mode replay --emit-hz 2000
    python scripts/servo_replay_publisher.py --mode fake     # connectivity test only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from src.monitor.servo_replay_stream import stream_replay  # noqa: E402


def build_app(default_mode: str = "replay",
              default_emit_hz: float | None = None) -> FastAPI:
    app = FastAPI(title="Servo FMCRD Replay Publisher")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "default_mode": default_mode}

    @app.get("/servo/stream")
    def servo_stream(mode: str | None = None, speed: float = 1.0,
                     emit_hz: float | None = None):
        """SSE feed of raw RAW_COLUMNS rows.

        `mode=replay` streams the real FMCRD LN->LO->HI journey (demo default).
        `mode=fake` emits synthetic rows for pipeline testing only — those
        predictions are INVALID (data is outside the model's training distribution).
        """
        return StreamingResponse(
            stream_replay(mode=mode or default_mode, speed=speed,
                          emit_hz=emit_hz if emit_hz is not None else default_emit_hz),
            media_type="text/event-stream",
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--mode", choices=["replay", "fake"], default="replay",
                    help="預設串流模式（replay=真實 FMCRD；fake=合成、僅連通性測試、預測無效）")
    ap.add_argument("--emit-hz", type=float, default=None,
                    help="覆寫發布頻率（訊息/秒）；預設用 config.servo_replay.emit_hz")
    args = ap.parse_args()

    import uvicorn
    app = build_app(default_mode=args.mode, default_emit_hz=args.emit_hz)
    print(f"[publisher] mode={args.mode}  http://{args.host}:{args.port}/servo/stream",
          flush=True)
    if args.mode == "fake":
        print("[publisher] ⚠ fake 模式：資料為合成、不在模型訓練分布內，預測輸出無效。",
              flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
