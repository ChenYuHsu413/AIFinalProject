"""Stream-aggregate the real PHM FMCRD servo dataset into the Module Servo
feature table — WITHOUT extracting or loading the 106 GB whole.

The dataset ships as 8 CSVs inside one zip (train/test × {LN ``load0``, and
``noisy`` LO/MED/HI}). Each file holds ~200 ``run_index`` segments of ~285k
timesteps. We stream every file from the zip in chunks and accumulate ONLINE
per-(file, run_index) statistics (sum / sum-of-squares / min / max / count), so
peak memory stays tiny regardless of total size.

Output schema == ``servo_features.build_feature_table`` (so ``train_servo`` /
``servo_dl`` work unchanged) PLUS:
  * a ``split`` column ("train" / "test") taken from the file name, and
  * ``DV`` normalised to 0..1 (real DV is in physical units ~0..3300); the raw
    max is printed so the risk bands can be recalibrated.

Run::

    python -m src.data.build_servo_from_zip --zip "C:/Users/<you>/Downloads/FMCRD_Data.zip"
    python -m src.data.build_servo_from_zip --validate   # math check on a small sample
"""
from __future__ import annotations

import argparse
import gc
import time
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.features.servo_features import (
    BASE_SIGNALS,
    HEALTH_LABELS,
    aggregate_run,
    all_feature_columns,
)
from src.utils.paths import ensure_output_dirs, load_config, resolve

# Raw columns we actually need (skip ``time`` and ``transitions``).
_USECOLS = [
    "DV", "rod_demand_pos", "rod_actual_pos", "torque", "rotor_speed",
    "i_3p_a", "i_3p_b", "i_3p_c", "direct", "quadrature",
    "run_index", "del_pos", "ylabel",
]
_DEFAULT_ZIP = "C:/Users/alung/Downloads/FMCRD_Data.zip"


def _new_acc(ylabel: str, split: str) -> Dict:
    return {
        # per signal: [sum, sumsq, min, max, count]
        "sig": {s: [0.0, 0.0, np.inf, -np.inf, 0] for s in BASE_SIGNALS},
        "cur_sumsq": 0.0,   # sum over rows of (ia^2 + ib^2 + ic^2)
        "cur_rows": 0,      # row count (3 phases per row)
        "dv_sum": 0.0,
        "dv_count": 0,
        "ylabel": ylabel,
        "split": split,
    }


_NUMERIC_COLS = [
    "DV", "rod_demand_pos", "rod_actual_pos", "torque", "rotor_speed",
    "i_3p_a", "i_3p_b", "i_3p_c", "direct", "quadrature", "del_pos", "run_index",
]


def _process_chunk(chunk: pd.DataFrame, fname: str, split: str, acc: Dict) -> int:
    """Fold one chunk into the per-(file, run_index) accumulators."""
    # Some files carry a few malformed rows (non-numeric junk in a signal column,
    # read as object dtype). Coerce to numeric (bad -> NaN, excluded from stats,
    # mirroring aggregate_run) and drop rows whose run_index itself is unparseable.
    for c in _NUMERIC_COLS:
        if chunk[c].dtype == object:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    if chunk["run_index"].isna().any():
        chunk = chunk[chunk["run_index"].notna()]
    chunk["run_index"] = chunk["run_index"].astype("int64")
    chunk["position_error"] = chunk["rod_actual_pos"] - chunk["rod_demand_pos"]
    for sig in BASE_SIGNALS:
        chunk[f"{sig}__sq"] = chunk[sig].to_numpy() ** 2
    chunk["__cur_sq3"] = (
        chunk["i_3p_a"].to_numpy() ** 2
        + chunk["i_3p_b"].to_numpy() ** 2
        + chunk["i_3p_c"].to_numpy() ** 2
    )

    aggspec: Dict[str, object] = {}
    for sig in BASE_SIGNALS:
        aggspec[sig] = ["sum", "min", "max", "count"]
        aggspec[f"{sig}__sq"] = "sum"
    aggspec["DV"] = ["sum", "count"]
    aggspec["__cur_sq3"] = ["sum", "count"]
    aggspec["ylabel"] = "first"

    res = chunk.groupby("run_index", sort=False).agg(aggspec)
    for ri, row in res.iterrows():
        key = (fname, int(ri))
        a = acc.get(key)
        if a is None:
            a = _new_acc(str(row[("ylabel", "first")]), split)
            acc[key] = a
        for sig in BASE_SIGNALS:
            S = a["sig"][sig]
            S[0] += float(row[(sig, "sum")])
            S[1] += float(row[(f"{sig}__sq", "sum")])
            S[2] = min(S[2], float(row[(sig, "min")]))
            S[3] = max(S[3], float(row[(sig, "max")]))
            S[4] += int(row[(sig, "count")])
        a["dv_sum"] += float(row[("DV", "sum")])
        a["dv_count"] += int(row[("DV", "count")])
        a["cur_sumsq"] += float(row[("__cur_sq3", "sum")])
        a["cur_rows"] += int(row[("__cur_sq3", "count")])
    return len(chunk)


# Per-file checkpoints — a crash only loses the current file; rerun resumes.
_CKPT_DIR = Path(
    "C:/Users/alung/AppData/Local/Temp/claude/"
    "C--Users-alung-Documents-WorkSpace-AIHW-FinalProject/"
    "9e33789f-92f2-49cc-a5a6-3f7256289049/scratchpad/servo_ckpt"
)


def _finalize_file(acc: Dict[Tuple[str, int], Dict], split: str) -> pd.DataFrame:
    """One file's accumulators -> per-run feature rows (raw DV, no global id)."""
    rows = []
    for (fname, ri), a in acc.items():
        feat: Dict[str, object] = {}
        for sig in BASE_SIGNALS:
            s, sq, mn, mx, c = a["sig"][sig]
            if c:
                mean = s / c
                var = max(sq / c - mean * mean, 0.0)
                feat[f"{sig}_mean"] = mean
                feat[f"{sig}_std"] = np.sqrt(var)
                feat[f"{sig}_min"] = mn
                feat[f"{sig}_max"] = mx
                feat[f"{sig}_rms"] = np.sqrt(sq / c)
            else:  # entirely-NaN signal -> 0.0 (mirror aggregate_run's len-guard)
                for st in ("mean", "std", "min", "max", "rms"):
                    feat[f"{sig}_{st}"] = 0.0
        feat["current_rms"] = (
            np.sqrt(a["cur_sumsq"] / (3 * a["cur_rows"])) if a["cur_rows"] else 0.0)
        feat["ylabel"] = a["ylabel"]
        feat["DV_raw"] = a["dv_sum"] / a["dv_count"] if a["dv_count"] else 0.0
        feat["split"] = split
        feat["__ri"] = ri
        rows.append(feat)
    return pd.DataFrame(rows)


def _process_one_file(z: zipfile.ZipFile, name: str, split: str,
                      chunksize: int) -> pd.DataFrame:
    acc: Dict[Tuple[str, int], Dict] = {}
    rows = 0
    t0 = time.perf_counter()
    with z.open(name) as f:
        for chunk in pd.read_csv(f, usecols=_USECOLS, chunksize=chunksize):
            _process_chunk(chunk, name, split, acc)
            rows += len(chunk)
            del chunk
            if rows % (chunksize * 30) == 0:
                rate = rows / max(time.perf_counter() - t0, 1e-9)
                print(f"    {rows:,} rows  ({rate/1e6:.2f} M rows/s)", flush=True)
    gc.collect()
    print(f"    -> {rows:,} rows, {len(acc)} runs in {time.perf_counter()-t0:.0f}s",
          flush=True)
    return _finalize_file(acc, split)


def run(zip_path: str = _DEFAULT_ZIP, chunksize: int = 300_000) -> Path:
    ensure_output_dirs()
    cfg = load_config()
    sv = cfg["servo"]
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    with zipfile.ZipFile(Path(zip_path)) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
        for fi, name in enumerate(names, 1):
            split = "train" if Path(name).name.lower().startswith("train") else "test"
            ck = _CKPT_DIR / (Path(name).stem + ".parquet")
            if ck.exists():
                print(f"[{fi}/{len(names)}] {name}: checkpoint hit, skip", flush=True)
                parts.append(pd.read_parquet(ck))
                continue
            print(f"[{fi}/{len(names)}] {name}  (split={split}) ...", flush=True)
            part = _process_one_file(z, name, split, chunksize)
            part.to_parquet(ck, index=False)
            parts.append(part)

    if not parts:
        raise ValueError("zip 內找不到任何 .csv，無法建表。")
    table = pd.concat(parts, ignore_index=True)
    table = table.sort_values(["split", "ylabel", "__ri"]).reset_index(drop=True)
    table.insert(0, "run_index", np.arange(len(table)))  # globally-unique id
    table = table.drop(columns=["__ri"])

    # --- DV normalisation to 0..1 (real DV is physical units) ---
    dv_raw_max = float(table["DV_raw"].max())
    denom = dv_raw_max if dv_raw_max > 0 else 1.0
    table["DV"] = (table["DV_raw"] / denom).clip(0.0, 1.0)
    table = table.drop(columns=["DV_raw"])
    print(f"[DV] raw max = {dv_raw_max:.3f}; normalised to 0..1.")
    print("[DV] normalised by class:\n",
          table.groupby("ylabel")["DV"].agg(["mean", "min", "max"]).round(4))
    print("[split] counts:\n", table.groupby(["split", "ylabel"]).size())

    unknown = sorted(set(table["ylabel"]) - set(HEALTH_LABELS))
    if unknown:
        raise ValueError(f"未知健康標籤 {unknown}，預期 {HEALTH_LABELS}。")

    feat_path = resolve(sv["processed_features"])
    table.to_parquet(feat_path, index=False)
    print(f"    -> 特徵表：{feat_path}（{table.shape[0]} 列 × {table.shape[1]} 欄）")

    rs = int(sv.get("random_state", 42))
    table.sample(frac=1.0, random_state=rs).reset_index(drop=True).to_csv(
        resolve(sv["feature_demo"]), index=False)
    table.groupby("ylabel", group_keys=False).head(3).reset_index(drop=True).to_csv(
        resolve(sv["sample_predictions"]), index=False)
    print(f"    -> demo / sample 已輸出。dv_raw_max={dv_raw_max:.3f}（重校 dv_risk 用）")
    return feat_path


# ---------------------------------------------------------------------------
# correctness check: online stats must match the existing aggregate_run
# ---------------------------------------------------------------------------
def validate(zip_path: str = _DEFAULT_ZIP, nrows: int = 700_000) -> None:
    """Read a small slice (a couple of full runs) and assert the streaming
    accumulator matches ``aggregate_run`` on the same rows."""
    with zipfile.ZipFile(zip_path) as z:
        name = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))[0]
        with z.open(name) as f:
            frame = pd.read_csv(f, usecols=_USECOLS, nrows=nrows)
    # keep only complete run_index groups (drop the last, possibly truncated)
    last_ri = frame["run_index"].iloc[-1]
    frame = frame[frame["run_index"] != last_ri].reset_index(drop=True)
    assert frame["run_index"].nunique() >= 1, "need at least one complete run"

    # reference: existing per-run aggregation
    ref_rows = []
    for ri, seg in frame.groupby("run_index"):
        r = aggregate_run(seg)
        r["DV"] = float(pd.to_numeric(seg["DV"]).mean())
        r["run_index"] = int(ri)
        ref_rows.append(r)
    ref = pd.DataFrame(ref_rows).set_index("run_index")

    # mine: stream the frame as chunks, finalize keyed by raw run_index
    acc: Dict = {}
    for start in range(0, len(frame), 200_000):
        _process_chunk(frame.iloc[start:start + 200_000], name, "train", acc)
    mine = {ri: a for (fn, ri), a in acc.items()}
    cols = all_feature_columns() + ["DV"]
    max_rel = 0.0
    for ri in ref.index:
        a = mine[ri]
        feat = {}
        for sig in BASE_SIGNALS:
            s, sq, mn, mx, c = a["sig"][sig]
            mean = s / c
            feat[f"{sig}_mean"] = mean
            feat[f"{sig}_std"] = np.sqrt(max(sq / c - mean * mean, 0.0))
            feat[f"{sig}_min"] = mn
            feat[f"{sig}_max"] = mx
            feat[f"{sig}_rms"] = np.sqrt(sq / c)
        feat["current_rms"] = np.sqrt(a["cur_sumsq"] / (3 * a["cur_rows"]))
        feat["DV"] = a["dv_sum"] / a["dv_count"]
        for c in cols:
            denom = abs(ref.loc[ri, c]) + 1e-9
            rel = abs(feat[c] - ref.loc[ri, c]) / denom
            max_rel = max(max_rel, rel)
    print(f"[validate] {len(ref)} runs × {len(cols)} cols; max relative diff = {max_rel:.2e}")
    assert max_rel < 1e-6, f"streaming stats diverge from aggregate_run ({max_rel:.2e})"
    print("[validate] OK — online stats match aggregate_run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=_DEFAULT_ZIP)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        validate(args.zip)
    else:
        run(args.zip, args.chunksize)
