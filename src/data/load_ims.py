"""Load IMS bearing run-to-failure snapshots (Module B).

The IMS Set 2 ("Test 2") distribution is a folder of 984 ASCII files, each one a
1-second vibration snapshot recorded every 10 minutes at 20 kHz.  Every file has
shape ``(20480, 4)`` — one accelerometer channel per bearing — and **no header**.
The file name *is* the timestamp (e.g. ``2004.02.12.10.32.39``), so the chronological
order of the run is recovered by sorting on the parsed timestamp.

See ``data/README.md`` for the download URL and placement, and
``docs/MODULE_B_IMS_PLAN.md`` for how this fits the dynamic-health track.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.utils.paths import load_config, resolve

# IMS file names are "YYYY.MM.DD.HH.MM.SS" with no extension.
_TIMESTAMP_FMT = "%Y.%m.%d.%H.%M.%S"


def parse_timestamp(name: str) -> datetime:
    """Parse an IMS file name (``2004.02.12.10.32.39``) into a ``datetime``."""
    return datetime.strptime(name, _TIMESTAMP_FMT)


def list_ims_files(raw_dir: Optional[str | Path] = None) -> List[Tuple[datetime, Path]]:
    """Return ``(timestamp, path)`` pairs for every snapshot, sorted by time.

    Parameters
    ----------
    raw_dir:
        Optional override of the snapshot folder.  Defaults to ``ims.raw_dir``
        in ``config.yaml``.
    """
    cfg = load_config()
    folder = resolve(raw_dir or cfg["ims"]["raw_dir"])
    if not folder.exists():
        raise FileNotFoundError(
            f"找不到 IMS Set 2 資料夾：{folder}\n"
            "請參考 data/README.md 的『模組 B — IMS 軸承資料集』下載與放置說明。"
        )
    pairs: List[Tuple[datetime, Path]] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        try:
            ts = parse_timestamp(p.name)
        except ValueError:
            # Skip anything that is not a timestamp-named snapshot.
            continue
        pairs.append((ts, p))
    if not pairs:
        raise FileNotFoundError(
            f"IMS 資料夾存在但找不到任何時間戳檔案：{folder}\n"
            "請確認解壓後的 984 個檔（檔名形如 2004.02.12.10.32.39）就在此資料夾下。"
        )
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_ims_file(path: str | Path) -> np.ndarray:
    """Load one snapshot as a ``(n_samples, n_bearings)`` float array.

    The files are whitespace/tab-delimited plain text; ``np.loadtxt`` handles
    both.  Set 2 yields ``(20480, 4)``.
    """
    return np.loadtxt(path)
