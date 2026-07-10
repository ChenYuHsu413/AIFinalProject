#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task A — extract small, representative FMCRD replay segments for the S1 demo.

The real PHM FMCRD test data ships as 8 CSVs inside one zip (train/test ×
{LN ``load0``, noisy LO/MED/HI}).  Each test file is a SINGLE health class and
each ``run_index`` segment is a full 6 s positioning cycle (a 5-step position
staircase, ~300k timesteps @ ~50 kHz).

CRITICAL — window granularity: the reference model was trained on features
aggregated over a WHOLE run (``build_feature_table`` groups by ``run_index``),
so a valid streaming window must also span a full run cycle.  A short head-slice
of a run (only the first position step) is OUT of the training distribution and
predicts nonsense.  We therefore extract several COMPLETE runs per class and
uniformly DECIMATE each run so the full 6 s motion cycle fits in a few MB while
preserving the column set and temporal order (decimation keeps sampling ORDER,
only the interval widens — recorded in the manifest).  A window of
``window.length_s`` = one run cycle then reproduces the training aggregation.

Concatenated in manifest ``order`` (LN -> LO -> HI) the segments play the full
"healthy -> degraded" journey for the publisher (Task B).

Note: FMCRD test files are single-class, so the degradation journey is
assembled ACROSS files (a few runs each), not from temporally-successive
segments of one file — the manifest records every segment's true source.

Run::

    python scripts/extract_replay_segments.py
    python scripts/extract_replay_segments.py --zip D:/data/FMCRD_Data.zip --runs-per-seg 3 --rows-per-run 5000
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on sys.path so ``src.*`` imports work when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.servo_features import RAW_COLUMNS  # noqa: E402
from src.utils.paths import load_config, resolve  # noqa: E402

_DEFAULT_ZIP = "C:/Users/alung/Downloads/FMCRD_Data.zip"

# Which test CSV inside the zip provides each health class. Matched case-insensitively
# by substring against the zip member basenames (FMCRD names them
# ``test_load0_…`` = LN and ``test_noisy_…{LO,MED,HI}`` = the noisy classes).
_CLASS_SOURCE = {
    "LN": "test_load0",
    "LO": "test_noisy",   # + suffix LO
    "HI": "test_noisy",   # + suffix HI
}
_CLASS_SUFFIX = {"LN": "", "LO": "LO", "HI": "HI"}

# Extract order = the degradation journey the publisher replays.
_ORDER = ["LN", "LO", "HI"]


def _find_member(names: list[str], klass: str) -> str:
    """Locate the zip member (test CSV) that holds this health class."""
    stem, suffix = _CLASS_SOURCE[klass], _CLASS_SUFFIX[klass]
    for n in names:
        base = Path(n).name.lower()
        if not base.startswith("test"):
            continue
        if stem in base and (not suffix or suffix.lower() in base):
            # Avoid LO matching a "MED"/"HI" file etc.: require the exact suffix
            # tag when one is expected, and reject the plain load0 for noisy classes.
            if suffix and suffix.lower() not in base:
                continue
            if klass == "LN" and "noisy" in base:
                continue
            return n
    raise FileNotFoundError(
        f"zip 內找不到 {klass} 對應的測試檔（期望檔名含 '{stem}'"
        + (f" 且以 '{suffix}' 結尾" if suffix else "")
        + f"）。實際 .csv 成員：{[Path(n).name for n in names]}"
    )


_RUN_ROWS_GUESS = 305_000  # ~one FMCRD run cycle (6 s @ ~50 kHz)


def _decimate_run(run: pd.DataFrame, rows_per_run: int) -> pd.DataFrame:
    """Uniformly subsample one run to ~rows_per_run rows spanning its full cycle."""
    step = max(1, len(run) // rows_per_run)
    return run.iloc[::step].head(rows_per_run)


def _extract_segment(zf: zipfile.ZipFile, member: str, runs_per_seg: int,
                     rows_per_run: int) -> tuple[pd.DataFrame, list[int]]:
    """First ``runs_per_seg`` COMPLETE runs of ``member``, each decimated.

    Reads enough rows to cover the wanted runs plus the start of the next one
    (so the last wanted run is guaranteed complete), then keeps only complete
    runs and decimates each to ~rows_per_run rows.
    """
    with zf.open(member) as f:
        df = pd.read_csv(f, nrows=_RUN_ROWS_GUESS * (runs_per_seg + 1))
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{member} 缺少 RAW_COLUMNS 欄位：{missing}（實際：{list(df.columns)}）")
    df = df[RAW_COLUMNS]  # enforce column set + order == RAW_COLUMNS
    seen = list(dict.fromkeys(df["run_index"].tolist()))  # first-seen order
    # Drop the last run_index (may be truncated by nrows); keep complete ones.
    complete = seen[:-1] if len(seen) > runs_per_seg else seen
    wanted = complete[:runs_per_seg]
    if len(wanted) < runs_per_seg:
        raise ValueError(
            f"{member} 只讀到 {len(wanted)} 個完整 run（需要 {runs_per_seg}）；"
            "請調小 --runs-per-seg 或確認來源檔完整。")
    parts = [_decimate_run(df[df["run_index"] == ri], rows_per_run) for ri in wanted]
    seg = pd.concat(parts, ignore_index=True)
    return seg, [int(r) for r in wanted]


def _dt_estimate(seg: pd.DataFrame) -> float:
    dt = np.diff(pd.to_numeric(seg["time"], errors="coerce").to_numpy())
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else float("nan")


def run(zip_path: str, runs_per_seg: int, rows_per_run: int) -> Path:
    cfg = load_config()["servo_replay"]
    out_dir = resolve(cfg["replay_dir"])
    manifest_path = resolve(cfg["manifest"])

    zp = Path(zip_path)
    if not zp.exists():
        raise SystemExit(
            "\n[extract_replay_segments] 找不到 FMCRD 原始 zip：\n"
            f"  {zp}\n\n"
            "本腳本需要真實 FMCRD 測試資料才能抽取 replay 素材。請：\n"
            "  1) 取得 FMCRD_Data.zip（內含 test_load0…(LN) 與 test_noisy…LO/MED/HI），且\n"
            "  2) 放到預設路徑 ~/Downloads/FMCRD_Data.zip，或用 --zip <路徑> 指定。\n"
            "（僅供管線測試時，可改用發布端的 fake 模式，但其資料不在模型訓練分布內、預測無效。）\n"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    with zipfile.ZipFile(zp) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for klass in _ORDER:
            member = _find_member(names, klass)
            seg, run_ids = _extract_segment(zf, member, runs_per_seg, rows_per_run)
            ylabels = sorted(seg["ylabel"].dropna().astype(str).unique().tolist())
            out_name = f"seg_{klass}.csv"
            out_path = out_dir / out_name
            seg.to_csv(out_path, index=False)
            size_mb = out_path.stat().st_size / 1e6
            entry = {
                "segment": klass,
                "file": out_name,
                "source_zip_member": member,
                "run_indexes": run_ids,
                "ylabel": ylabels[0] if len(ylabels) == 1 else ylabels,
                "rows": int(len(seg)),
                "runs": len(run_ids),
                "decimated_sampling_dt_s": round(_dt_estimate(seg), 8),
                "size_mb": round(size_mb, 3),
            }
            segments.append(entry)
            print(f"  [{klass}] {member} runs={run_ids} "
                  f"rows={entry['rows']:,} ylabel={entry['ylabel']} "
                  f"-> {out_name} ({size_mb:.2f} MB)", flush=True)

    manifest = {
        "dataset": "FMCRD (real PHM servomotor-driven ballscrew) — test split",
        "purpose": "S1 live streaming demo replay material (degradation journey).",
        "columns": RAW_COLUMNS,
        "order": _ORDER,
        "note": ("FMCRD test files are single-class; the LN->LO->HI journey is "
                 "assembled across files (a few complete runs each). Each run is a "
                 "full 6 s / 5-step positioning cycle, uniformly DECIMATED to keep "
                 "segments small — a window of ~one run cycle then matches the "
                 "per-run training aggregation. Columns & temporal order preserved; "
                 "only the sampling interval is widened by decimation."),
        "run_cycle_s": 6.0,
        "source_zip": str(zp),
        "segments": segments,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    total = sum(s["size_mb"] for s in segments)
    print(f"\n  manifest -> {manifest_path}")
    print(f"  {len(segments)} 段、合計 {total:.2f} MB。順序：{' -> '.join(_ORDER)}")
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=_DEFAULT_ZIP, help="FMCRD_Data.zip 路徑")
    ap.add_argument("--runs-per-seg", type=int, default=3,
                    help="每段納入的完整 run 數（每 run = 一個 6s 動作循環；預設 3）")
    ap.add_argument("--rows-per-run", type=int, default=5000,
                    help="每個 run 抽稀後的列數（控制檔案大小；預設 5000 -> 約 3 MB/段）")
    args = ap.parse_args()
    run(args.zip, args.runs_per_seg, args.rows_per_run)


if __name__ == "__main__":
    main()
