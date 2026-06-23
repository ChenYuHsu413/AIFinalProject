"""Load XJTU-SY bearing run-to-failure snapshots (Module B+).

The XJTU-SY distribution groups CSV snapshots by operating condition and bearing::

    data/raw/xjtu/
        35Hz12kN/            # condition 1: 2100 rpm, 12 kN
            Bearing1_1/
                1.csv        # one snapshot per minute, named by index
                2.csv
                ...
            Bearing1_2/ ...  # Bearing1_1 .. Bearing1_5
        37.5Hz11kN/          # condition 2: 2250 rpm, 11 kN (Bearing2_1 .. 2_5)
        40Hz10kN/            # condition 3: 2400 rpm, 10 kN (Bearing3_1 .. 3_5)

Each CSV holds ``32768`` rows x ``2`` columns sampled at 25.6 kHz (1.28 s):
column 0 = horizontal vibration, column 1 = vertical vibration.  Some
distributions prepend a header row
(``Horizontal_vibration_signals,Vertical_vibration_signals``); the loader
auto-detects and skips it.

The file name is an increasing integer (the minute index), so chronological
order is recovered by sorting on the parsed index.  Mirrors
``src/data/load_ims.py``.

See ``docs/MODULE_B_PLUS_XJTU_PLAN.md``.  Raw data is downloaded by the user
and never committed (see ``.gitignore``: ``data/raw/xjtu/``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.utils.paths import resolve

# Default download location; pass an explicit folder to override.
DEFAULT_XJTU_DIR = "data/raw/xjtu"


def parse_index(name: str) -> int:
    """Parse a snapshot file name (``1.csv``, ``2.csv`` ...) into its minute index."""
    return int(Path(name).stem)


def list_xjtu_files(bearing_dir: str | Path) -> List[Tuple[int, Path]]:
    """Return ``(index, path)`` pairs for one bearing folder, sorted by index.

    Parameters
    ----------
    bearing_dir:
        A single bearing folder, e.g.
        ``data/raw/xjtu/35Hz12kN/Bearing1_1``.
    """
    folder = resolve(bearing_dir)
    if not folder.exists():
        raise FileNotFoundError(
            f"找不到 XJTU 軸承資料夾：{folder}\n"
            "請參考 docs/MODULE_B_PLUS_XJTU_PLAN.md 的下載與放置說明，"
            "確認已下載 XJTU-SY 並解壓到 data/raw/xjtu/<工況>/<軸承>/。"
        )
    pairs: List[Tuple[int, Path]] = []
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        try:
            idx = parse_index(p.name)
        except ValueError:
            # Skip anything that is not an index-named snapshot.
            continue
        pairs.append((idx, p))
    if not pairs:
        raise FileNotFoundError(
            f"XJTU 軸承資料夾存在但找不到任何 CSV 快照：{folder}\n"
            "請確認資料夾下是 1.csv, 2.csv … 形式的逐分鐘快照。"
        )
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_xjtu_file(path: str | Path) -> np.ndarray:
    """Load one snapshot as a ``(32768, 2)`` float array (horizontal, vertical).

    The files are comma-delimited; a header row is auto-detected and skipped
    when present.
    """
    path = Path(path)
    with path.open("r") as fh:
        first_token = fh.readline().split(",")[0].strip()
    try:
        float(first_token)
        skip = 0  # first line is data
    except ValueError:
        skip = 1  # first line is a header
    return np.loadtxt(path, delimiter=",", skiprows=skip)
