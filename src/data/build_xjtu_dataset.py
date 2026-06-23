"""Assemble XJTU-SY Condition-1 feature tables with RUL / health labels (Module B+).

Mirrors ``src/data/build_ims_dataset.py``, but over *multiple* run-to-failure
bearings (Bearing1_1 .. Bearing1_5) so the same fixed-parameter health pipeline
can be validated across independent trajectories — the cross-bearing
generalization that IMS Set 2's single trajectory cannot provide.

Each snapshot is recorded one minute apart, so for a bearing whose last (failure)
snapshot has index ``N``::

    rul_hours = (N - i) * snapshot_interval_min / 60     # last snapshot -> RUL 0
    health    = rul_hours / rul_max * 100                 # 100 at start, 0 at failure

Only time-domain features are extracted (the MVP scope); XJTU-specific
defect-band energies are deferred.  Run ``python -m src.data.build_xjtu_dataset``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.load_xjtu import list_xjtu_files, load_xjtu_file
from src.features.vibration_features import time_domain_features
from src.utils.paths import load_config, resolve

_CHANNEL_INDEX = {"horizontal": 0, "vertical": 1}


def extract_xjtu_features(array: np.ndarray, channel: str = "horizontal") -> Dict[str, float]:
    """Collapse one XJTU snapshot ``(32768, 2)`` into a flat feature row.

    The modeling channel (default horizontal -> ``h_`` prefix) gets the full
    time-domain set; the other channel contributes only RMS as a reference.
    """
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"預期 snapshot 形狀為 (n_samples, 2)，實際為 {arr.shape}。")
    model_idx = _CHANNEL_INDEX[channel]
    other_idx = 1 - model_idx
    model_prefix = "h" if channel == "horizontal" else "v"
    other_prefix = "v" if channel == "horizontal" else "h"

    row: Dict[str, float] = {}
    for k, v in time_domain_features(arr[:, model_idx]).items():
        row[f"{model_prefix}_{k}"] = v
    row[f"{other_prefix}_rms"] = float(np.sqrt(np.mean(arr[:, other_idx] ** 2)))
    return row


def add_rul_health(df: pd.DataFrame, interval_min: float) -> pd.DataFrame:
    """Append ``rul_hours`` and ``health`` to a minute-indexed feature frame.

    Assumes ``df`` is sorted ascending by snapshot index and the last row is the
    failure point. ``health`` is a linear rescaling of RUL to 0..100 (Plan A).
    """
    out = df.copy()
    idx_last = out.index[-1]
    out["rul_hours"] = (idx_last - out.index) * interval_min / 60.0
    rul_max = out["rul_hours"].iloc[0]
    out["health"] = out["rul_hours"] / rul_max * 100.0
    return out


def build_bearing_table(bearing_dir: str | Path, channel: str, interval_min: float) -> pd.DataFrame:
    """Build one bearing's feature table (one row per snapshot) with labels."""
    files = list_xjtu_files(bearing_dir)
    rows, index = [], []
    for idx, path in files:
        rows.append(extract_xjtu_features(load_xjtu_file(path), channel))
        index.append(idx)
    df = pd.DataFrame(rows, index=pd.Index(index, name="minute"))
    return add_rul_health(df, interval_min)


def build_feature_table(raw_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Build the combined Condition-1 table (5 bearings stacked, ``bearing`` column)."""
    cfg = load_config()
    xj = cfg["xjtu"]
    root = resolve(raw_dir or xj["raw_dir"]) / xj["condition"]
    channel = xj["channel"]
    interval = xj["snapshot_interval_min"]
    frames = []
    for bearing in xj["bearings"]:
        t = build_bearing_table(root / bearing, channel, interval).reset_index()
        t.insert(0, "bearing", bearing)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    cfg = load_config()
    df = build_feature_table()
    out_path = resolve(cfg["xjtu"]["processed_features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print("=" * 70)
    print("XJTU-SY Condition 1 特徵表已建立")
    print("=" * 70)
    print(f"輸出：{out_path}")
    print(f"形狀：{df.shape}（{df.shape[0]} 列 × {df.shape[1]} 欄）")
    for bearing, g in df.groupby("bearing"):
        print(
            f"  {bearing}: {len(g)} 快照, "
            f"RUL {g['rul_hours'].min():.2f}–{g['rul_hours'].max():.2f} h, "
            f"h_rms {g['h_rms'].iloc[0]:.4f}→{g['h_rms'].iloc[-1]:.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
