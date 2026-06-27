"""Phase B — build a raw-signal *sequence* dataset for the Servo 1D-CNN.

A 1D-CNN needs the raw time-ordered waveform, but a single short window (~1024 of
a run's ~285k timesteps) is a noisy sample — the degradation signal lives in the
run's long-timescale energy/variance (which is exactly why the per-run aggregate
models reach macro-F1 ~0.76). So each **run** becomes ONE sample: we stream the
run's raw signal and reduce it to a fixed-length **energy envelope** — per channel,
the std within each of ``ENV_LEN`` contiguous time-blocks. This keeps the amplitude
/ variance degradation signature, stays time-ordered (the CNN still convolves over
time), and is stable. Channels stay raw/physical (no hand-picked features).

Honesty / scope:
  * **MVP subset** — at most ``MAX_RUNS_PER_FILE`` runs per file (streamed from the
    zip head), so the set is small and the CNN trains on CPU in minutes.
  * **Split by source file** (train_* vs test_*) — train/test runs never share a
    file (no leakage). File name fixes the class (load0->LN; noisy HI/LO/MED).
  * The ``.npz`` is transient (gitignored, rebuilt from the zip); only the trained
    model's results JSON is committed and read by the cloud app.

Run::

    python -m src.data.build_servo_windows --zip "C:/Users/<you>/Downloads/FMCRD_Data.zip"
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.servo_features import HEALTH_LABELS
from src.utils.paths import ensure_output_dirs, load_config, resolve

# Physical channels fed to the CNN (position_error is derived rod_actual-demand).
CHANNELS = [
    "torque", "rotor_speed", "i_3p_a", "i_3p_b", "i_3p_c",
    "direct", "quadrature", "position_error",
]
# ylabel is intentionally absent: the class is taken from the file name in run(),
# so we never parse the per-row label column here.
_USECOLS = [
    "rod_demand_pos", "rod_actual_pos", "torque", "rotor_speed",
    "i_3p_a", "i_3p_b", "i_3p_c", "direct", "quadrature", "run_index",
]

ENV_LEN = 256            # envelope length (time-blocks per run)
MIN_RUN_ROWS = 50_000    # ignore truncated tail runs shorter than this
MAX_RUNS_PER_FILE = 40   # cap runs (samples) per file
_CHUNK = 500_000
_DEFAULT_ZIP = "C:/Users/alung/Downloads/FMCRD_Data.zip"


def _envelope(run_rows: list[np.ndarray]) -> np.ndarray | None:
    """Reduce one run's raw signal (T, C) to an energy envelope (C, ENV_LEN).

    Returns None if, after dropping NaN rows, there are fewer valid samples than
    ENV_LEN — np.array_split would then make empty blocks whose std is NaN, which
    would silently poison training. Such a run is skipped instead.
    """
    sig = np.concatenate(run_rows, axis=0)        # (T, C)
    sig = sig[~np.isnan(sig).any(axis=1)]
    if len(sig) < ENV_LEN:
        return None
    blocks = np.array_split(sig, ENV_LEN, axis=0)  # ENV_LEN non-empty time-blocks
    env = np.stack([b.std(axis=0) for b in blocks], axis=1)  # (C, ENV_LEN)
    return env.astype(np.float32)


def _runs_from_file(z: zipfile.ZipFile, name: str) -> list[np.ndarray]:
    """Stream a file run-by-run; return up to MAX_RUNS_PER_FILE envelopes."""
    envs: list[np.ndarray] = []
    cur_ri: int | None = None
    cur_rows: list[np.ndarray] = []
    cur_n = 0

    def flush():
        nonlocal cur_rows, cur_n
        if cur_n >= MIN_RUN_ROWS:
            env = _envelope(cur_rows)
            if env is not None:
                envs.append(env)
        cur_rows, cur_n = [], 0

    with z.open(name) as f:
        for chunk in pd.read_csv(f, usecols=_USECOLS, chunksize=_CHUNK):
            for c in _USECOLS:
                if chunk[c].dtype == object:
                    chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            chunk = chunk[chunk["run_index"].notna()]
            if chunk.empty:  # whole chunk had unparseable run_index — nothing to fold
                continue
            chunk["run_index"] = chunk["run_index"].astype("int64")
            chunk["position_error"] = chunk["rod_actual_pos"] - chunk["rod_demand_pos"]
            ri = chunk["run_index"].to_numpy()
            sig = chunk[CHANNELS].to_numpy(dtype=np.float32)
            # split this chunk at run-index boundaries
            bounds = np.flatnonzero(np.diff(ri)) + 1
            for part_ri, part in zip(np.split(ri, bounds), np.split(sig, bounds)):
                r = int(part_ri[0])
                if cur_ri is None:
                    cur_ri = r
                if r != cur_ri:
                    flush()
                    cur_ri = r
                    if len(envs) >= MAX_RUNS_PER_FILE:
                        return envs
                cur_rows.append(part)
                cur_n += len(part)
            if len(envs) >= MAX_RUNS_PER_FILE:
                return envs
    flush()
    return envs[:MAX_RUNS_PER_FILE]


def run(zip_path: str = _DEFAULT_ZIP) -> Path:
    ensure_output_dirs()
    cfg = load_config()["servo"]
    lab_to_idx = {lab: i for i, lab in enumerate(HEALTH_LABELS)}

    X_parts, y_parts, split_parts = [], [], []
    with zipfile.ZipFile(Path(zip_path)) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
        for name in names:
            base = Path(name).name.lower()
            split = "train" if base.startswith("train") else "test"
            ylabel = ("LN" if "load0" in base else
                      "HI" if "hi" in base else "LO" if "lo" in base else
                      "MED" if "med" in base else None)
            if ylabel not in lab_to_idx:
                raise ValueError(f"無法從檔名判定健康標籤：{name}")
            envs = _runs_from_file(z, name)
            arr = np.stack(envs) if envs else np.empty((0, len(CHANNELS), ENV_LEN), np.float32)
            X_parts.append(arr)
            y_parts.append(np.full(len(arr), lab_to_idx[ylabel], dtype=np.int64))
            split_parts.append(np.array([split] * len(arr)))
            print(f"  {Path(name).name}: {len(arr)} runs  (ylabel={ylabel}, split={split})",
                  flush=True)

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    split = np.concatenate(split_parts)
    if len(X) == 0:
        raise ValueError("沒有抽到任何 run 序列，請確認 zip 內容與欄位。")

    out = resolve(cfg["windows_path"])
    np.savez_compressed(
        out, X=X, y=y, split=split,
        channels=np.array(CHANNELS), labels=np.array(HEALTH_LABELS),
        win_len=ENV_LEN, repr="per_run_energy_envelope_blockstd",
    )
    tr = (split == "train").sum()
    print(f"[windows] X={X.shape}  train={tr}  test={len(X) - tr}  -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=_DEFAULT_ZIP)
    args = ap.parse_args()
    run(args.zip)
