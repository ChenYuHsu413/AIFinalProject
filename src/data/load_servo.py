"""Load raw PHM servomotor CSV file(s) from ``servo.raw_dir``.

The real PHM servomotor-driven ballscrew dataset ships as per-timestep CSV(s)
with the columns in ``servo_features.RAW_COLUMNS``.  This loader concatenates
whatever CSVs are present; if none are, callers fall back to the synthetic
placeholder generator (``build_servo_dataset``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.utils.paths import load_config, resolve


def list_servo_files() -> List[Path]:
    cfg = load_config()
    raw_dir = resolve(cfg["servo"]["raw_dir"])
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.glob("*.csv"))


def load_raw_servo() -> pd.DataFrame:
    """Concatenate all raw servo CSVs. Empty frame if none are present.

    Each frame is tagged with ``__source_file__`` so ``build_feature_table`` can
    keep segments from different files separate even when their ``run_index``
    values collide (PHM ships multiple experiment files each restarting at 0).
    """
    files = list_servo_files()
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        fr = pd.read_csv(f)
        fr["__source_file__"] = f.name
        frames.append(fr)
    return pd.concat(frames, ignore_index=True)
