"""Load Paderborn bearing-dataset measurements (Module C, current + vibration).

Paderborn (Lessmeier et al., 2016, KAt-DataCenter) ships one ZIP per bearing
code; each extracts to a folder named by the code (e.g. ``KA01/``) holding the
measurement ``.mat`` files for every operating condition.  A measurement file is
named ``{COND}_{CODE}_{idx}.mat`` (e.g. ``N15_M07_F10_KA01_1.mat``) and contains
a MATLAB struct whose ``Y`` field is an array of named signals (``vibration_1``,
``phase_current_1``, ``phase_current_2``, ``speed``, ``torque``, ...).

This module only parses files into numpy channel arrays; feature extraction and
labelling live in ``src/data/build_paderborn_dataset.py``.  See
``docs/MODULE_C_PADERBORN_PLAN.md`` for the download / placement instructions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.paths import resolve


def parse_measurement_name(name: str) -> Tuple[str, str, int]:
    """Split a measurement file name into ``(condition, bearing_code, index)``.

    ``"N15_M07_F10_KA01_3.mat"`` -> ``("N15_M07_F10", "KA01", 3)``.  The operating
    condition is the first three underscore tokens; the code is the fourth; the
    index is the trailing integer.
    """
    parts = Path(name).stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"無法解析 Paderborn 量測檔名：{name}")
    condition = "_".join(parts[:3])
    code = parts[3]
    index = int(parts[4])
    return condition, code, index


def list_paderborn_files(
    code_dir: str | Path, conditions: Optional[Sequence[str]] = None
) -> List[Tuple[str, str, int, Path]]:
    """Return ``(condition, code, index, path)`` for one bearing-code folder.

    ``conditions`` (if given) keeps only matching operating conditions.  Sorted by
    ``(condition, index)``.
    """
    folder = resolve(code_dir)
    if not folder.exists():
        raise FileNotFoundError(
            f"找不到 Paderborn 軸承碼資料夾：{folder}\n"
            "請參考 docs/MODULE_C_PADERBORN_PLAN.md 的下載與放置說明，"
            "確認已下載 Paderborn 並解壓到 data/raw/paderborn/<軸承碼>/。"
        )
    want = set(conditions) if conditions else None
    rows: List[Tuple[str, str, int, Path]] = []
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() != ".mat":
            continue
        try:
            cond, code, idx = parse_measurement_name(p.name)
        except ValueError:
            continue
        if want is not None and cond not in want:
            continue
        rows.append((cond, code, idx, p))
    if not rows:
        raise FileNotFoundError(
            f"Paderborn 資料夾存在但找不到符合的 .mat 量測：{folder}\n"
            f"（工況篩選={conditions}）。請確認檔名為 <工況>_<碼>_<序號>.mat。"
        )
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


def load_paderborn_mat(path: str | Path, channel_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load the requested named signals from one Paderborn ``.mat`` measurement.

    Returns ``{channel_name: 1-D float array}``.  Raises ``KeyError`` if a
    requested channel is absent.  ``scipy.io.loadmat`` is imported lazily so this
    module imports without touching the real dataset (e.g. during unit tests that
    build feature rows from synthetic arrays).
    """
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    key = next(k for k in mat if not k.startswith("__"))
    struct = mat[key]
    signals: Dict[str, np.ndarray] = {}
    for el in np.atleast_1d(struct.Y):
        nm = str(el.Name)
        if nm in channel_names:
            signals[nm] = np.asarray(el.Data, dtype=float).ravel()
    missing = [c for c in channel_names if c not in signals]
    if missing:
        raise KeyError(f"{Path(path).name} 缺少訊號通道：{missing}")
    return signals
