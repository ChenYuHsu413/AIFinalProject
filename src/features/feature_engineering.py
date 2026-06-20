"""Domain-motivated derived features for the AI4I dataset.

Each feature has a short engineering rationale documented in the README; the
docstrings here repeat the key idea for in-code reference.
"""
from __future__ import annotations

import pandas as pd

AIR_TEMP = "Air temperature [K]"
PROC_TEMP = "Process temperature [K]"
RPM = "Rotational speed [rpm]"
TORQUE = "Torque [Nm]"
WEAR = "Tool wear [min]"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the five engineered features appended.

    Features
    --------
    temp_diff
        Process minus air temperature. A growing gap usually means the cooling
        loop cannot keep up with the heat the process generates.
    power_proxy
        Torque x rotational speed: proportional to mechanical power. High
        sustained values stress the motor.
    wear_torque_interaction
        Tool wear x torque: a worn tool under heavy load is a known failure mode.
    wear_speed_interaction
        Tool wear x rotational speed: a worn tool spinning fast is also risky.
    temp_wear_interaction
        Process temperature x tool wear: combines thermal load and tool age.
    """
    out = df.copy()
    out["temp_diff"] = out[PROC_TEMP] - out[AIR_TEMP]
    out["power_proxy"] = out[TORQUE] * out[RPM]
    out["wear_torque_interaction"] = out[WEAR] * out[TORQUE]
    out["wear_speed_interaction"] = out[WEAR] * out[RPM]
    out["temp_wear_interaction"] = out[PROC_TEMP] * out[WEAR]
    return out


ENGINEERED_COLUMNS = [
    "temp_diff",
    "power_proxy",
    "wear_torque_interaction",
    "wear_speed_interaction",
    "temp_wear_interaction",
]
