"""Assemble the Paderborn feature table for fault classification (Module C).

Each measurement (one ``.mat`` file) collapses to a single row of time-domain
features over the vibration channel and both motor phase-current channels
(reusing ``vibration_features.time_domain_features``), plus the labels:

  * ``fault_class``   — healthy / outer / inner
  * ``damage_origin`` — healthy / artificial / real

The artificial-vs-real origin is what the headline experiment splits on (train on
healthy + artificial, test on real). Only time-domain features are extracted in
this MVP; motor-current spectral side-bands (true MCSA) are a later add.

Run ``python -m src.data.build_paderborn_dataset`` after placing the raw data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.load_paderborn import list_paderborn_files, load_paderborn_mat
from src.features.vibration_features import time_domain_features
from src.utils.paths import load_config, resolve

# Channel name -> short column prefix.  Falls back to a sanitised name otherwise.
_PREFIX = {
    "vibration_1": "vib",
    "phase_current_1": "cur1",
    "phase_current_2": "cur2",
}

# Bearing group key -> (fault_class, damage_origin).
_GROUP_LABEL = {
    "healthy": ("healthy", "healthy"),
    "artificial_outer": ("outer", "artificial"),
    "artificial_inner": ("inner", "artificial"),
    "real_outer": ("outer", "real"),
    "real_inner": ("inner", "real"),
}


def _prefix_for(channel: str) -> str:
    return _PREFIX.get(channel, channel.replace("phase_", "").replace("_", ""))


def bearing_label_map(bearings: Dict[str, Sequence[str]]) -> Dict[str, Tuple[str, str]]:
    """Flatten the config ``bearings`` groups into ``code -> (fault_class, origin)``."""
    out: Dict[str, Tuple[str, str]] = {}
    for group, codes in bearings.items():
        if group not in _GROUP_LABEL:
            raise KeyError(f"未知的軸承分組：{group}（可用：{sorted(_GROUP_LABEL)}）")
        label = _GROUP_LABEL[group]
        for code in codes:
            out[code] = label
    return out


def extract_paderborn_features(
    channels: Dict[str, np.ndarray], channel_names: Optional[Sequence[str]] = None
) -> Dict[str, float]:
    """Collapse one measurement's channels into a flat time-domain feature row.

    Each requested channel contributes the full 10-feature set under its prefix
    (e.g. ``vib_rms``, ``cur1_kurtosis``, ``cur2_crest_factor``).
    """
    names = list(channel_names) if channel_names is not None else list(channels)
    row: Dict[str, float] = {}
    for ch in names:
        prefix = _prefix_for(ch)
        for k, v in time_domain_features(channels[ch]).items():
            row[f"{prefix}_{k}"] = v
    return row


def build_feature_table(raw_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Build the combined feature table over all configured codes / conditions."""
    cfg = load_config()
    pb = cfg["paderborn"]
    root = resolve(raw_dir or pb["raw_dir"])
    conditions = pb["conditions"]
    channels = pb["channels"]
    labels = bearing_label_map(pb["bearings"])

    rows = []
    for code, (fault_class, origin) in labels.items():
        for cond, _code, idx, path in list_paderborn_files(root / code, conditions):
            sig = load_paderborn_mat(path, channels)
            feat = extract_paderborn_features(sig, channels)
            feat.update({
                "condition": cond,
                "bearing_code": code,
                "fault_class": fault_class,
                "damage_origin": origin,
                "measurement": idx,
            })
            rows.append(feat)
    df = pd.DataFrame(rows)
    label_cols = ["condition", "bearing_code", "fault_class", "damage_origin", "measurement"]
    feat_cols = [c for c in df.columns if c not in label_cols]
    return df[label_cols + feat_cols]


def main() -> None:
    cfg = load_config()
    df = build_feature_table()
    out_path = resolve(cfg["paderborn"]["processed_features"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print("=" * 70)
    print("Paderborn 故障診斷特徵表已建立")
    print("=" * 70)
    print(f"輸出：{out_path}")
    print(f"形狀：{df.shape}（{df.shape[0]} 列 × {df.shape[1]} 欄）")
    print("各 (origin, fault_class) 量測數：")
    counts = df.groupby(["damage_origin", "fault_class"]).size()
    for (origin, fault), n in counts.items():
        print(f"  {origin:>10} / {fault:<8}: {n}")
    print("=" * 70)


if __name__ == "__main__":
    main()
