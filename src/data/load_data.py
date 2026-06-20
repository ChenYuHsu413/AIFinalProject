"""Load the AI4I 2020 Predictive Maintenance dataset and print an EDA summary.

The CSV must be placed at ``data/raw/ai4i2020.csv`` (see ``data/README.md`` for
the download URL).  When invoked as a module (``python -m src.data.load_data``)
it prints a textual EDA summary so the user can quickly inspect the data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.paths import load_config, resolve


def load_raw(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw AI4I CSV.

    Parameters
    ----------
    path:
        Optional override of the CSV location.  Defaults to the path declared
        in ``config.yaml`` under ``paths.data_raw``.
    """
    cfg = load_config()
    csv_path = resolve(path or cfg["paths"]["data_raw"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 AI4I 2020 資料集：{csv_path}\n"
            "請參考 data/README.md 中的下載與放置說明。"
        )
    # ``utf-8-sig`` strips the byte-order mark that the UCI distribution ships
    # with — otherwise the first column name becomes ``﻿UDI`` and the
    # downstream "drop UDI" step silently fails.
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


def basic_eda(df: pd.DataFrame) -> dict:
    """Return a dictionary of EDA facts that downstream callers can render."""
    cfg = load_config()
    target = cfg["columns"]["target_primary"]
    failure_types = cfg["columns"]["failure_types"]
    numeric_cols = cfg["columns"]["numeric_raw"]

    eda = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "target_distribution": df[target].value_counts().to_dict(),
        "target_rate": float(df[target].mean()),
        "type_distribution": df["Type"].value_counts().to_dict(),
        "failure_type_counts": {ft: int(df[ft].sum()) for ft in failure_types},
        "numeric_describe": df[numeric_cols].describe().round(3).to_dict(),
    }
    return eda


def main() -> None:
    df = load_raw()
    eda = basic_eda(df)
    print("=" * 70)
    print("AI4I 2020 預測性維護資料集 — 基本 EDA 摘要")
    print("=" * 70)
    print(f"資料維度：{eda['shape']}")
    print(f"\n欄位（{len(eda['columns'])} 欄）：{eda['columns']}")
    print("\n-- 缺失值檢查 --")
    for col, n in eda["missing"].items():
        if n:
            print(f"  {col}：{n}")
    if not any(eda["missing"].values()):
        print("  （無缺失值）")
    print("\n-- 目標欄位 Machine failure 分布 --")
    for k, v in eda["target_distribution"].items():
        print(f"  {k}：{v}")
    print(f"  故障比例 = {eda['target_rate']:.4f}")
    print("\n-- 產品類別 Type 分布 --")
    for k, v in eda["type_distribution"].items():
        print(f"  {k}：{v}")
    print("\n-- 故障類型計數（TWF / HDF / PWF / OSF / RNF）--")
    for k, v in eda["failure_type_counts"].items():
        print(f"  {k}：{v}")
    print("\n-- 數值欄位描述統計 --")
    print(df[load_config()["columns"]["numeric_raw"]].describe().round(3))
    print("=" * 70)


if __name__ == "__main__":
    main()
