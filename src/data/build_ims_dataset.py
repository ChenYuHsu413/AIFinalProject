"""Assemble the IMS Set 2 feature table with RUL / health labels (Module B).

Pipeline: list snapshots in chronological order -> extract one feature row per
snapshot -> attach the Remaining-Useful-Life label and a 0..100 health score.

Label scheme (Plan A — linear, see docs/MODULE_B_IMS_PLAN.md):
    RUL_hours = (t_last - t) / 3600           # last snapshot == failure -> RUL 0
    health    = RUL_hours / RUL_max * 100      # 100 at run start, 0 at failure

Run ``python -m src.data.build_ims_dataset`` to build and save the parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.load_ims import list_ims_files, load_ims_file
from src.features.vibration_features import extract_file_features
from src.utils.paths import load_config, resolve


def add_rul_health(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``rul_hours`` and ``health`` to a time-indexed feature frame.

    Assumes ``df`` is sorted ascending by its ``DatetimeIndex`` and that the last
    row coincides with failure. ``health`` is a linear rescaling of RUL to 0..100.
    """
    out = df.copy()
    t_last = out.index[-1]
    out["rul_hours"] = (t_last - out.index).total_seconds() / 3600.0
    rul_max = out["rul_hours"].iloc[0]
    out["health"] = out["rul_hours"] / rul_max * 100.0
    return out


def build_feature_table(raw_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Build the full IMS Set 2 feature table (one row per snapshot) with labels.

    Note: reads all 984 snapshots (~1.5 GB of text) — expect a few minutes.
    """
    cfg = load_config()
    files = list_ims_files(raw_dir)
    rows = []
    index = []
    for ts, path in files:
        rows.append(extract_file_features(load_ims_file(path), cfg))
        index.append(ts)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="timestamp"))
    return add_rul_health(df)


def main() -> None:
    cfg = load_config()
    df = build_feature_table()
    out_path = resolve(cfg["ims"]["processed_features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print("=" * 70)
    print("IMS Set 2 特徵表已建立")
    print("=" * 70)
    print(f"輸出：{out_path}")
    print(f"形狀：{df.shape}（{df.shape[0]} 個快照 × {df.shape[1]} 欄）")
    print(f"時間範圍：{df.index[0]} → {df.index[-1]}")
    print(f"RUL 範圍：{df['rul_hours'].min():.2f} → {df['rul_hours'].max():.2f} 小時")
    print(f"health 範圍：{df['health'].min():.1f} → {df['health'].max():.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
