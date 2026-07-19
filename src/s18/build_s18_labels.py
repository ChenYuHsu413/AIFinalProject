"""情境 18 —— 建立**評估用**標註表(`DV` 與 `ylabel`)。

與特徵管線**完全分離的第二遍掃描**:本腳本只讀 `run_index`, `DV`, `ylabel`,
不碰任何訊號欄;特徵管線只讀訊號欄,不碰 DV/ylabel。兩者在分析階段才 join。
這個「兩遍掃描」的架構本身就是防洩漏鐵律 1 的結構性保證。

用法::

    python -m src.s18.build_s18_labels --split train
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import pandas as pd

from src.s18.build_s18_features import load_params

LABEL_COLS = ["run_index", "DV", "ylabel"]


def build(split: str, params: dict, outdir: Path) -> pd.DataFrame:
    files = params["data"][f"{split}_files"]
    zip_path = params["data"]["zip_path"]
    z = zipfile.ZipFile(zip_path)
    rows = []
    for fi, member in enumerate(files, 1):
        t0 = time.perf_counter()
        acc: dict = {}
        with z.open(member) as f:
            for chunk in pd.read_csv(f, usecols=LABEL_COLS, chunksize=1_000_000):
                g = chunk.groupby("run_index").agg(
                    dv_sum=("DV", "sum"), dv_n=("DV", "count"),
                    ylabel=("ylabel", "first"))
                for ri, r in g.iterrows():
                    a = acc.setdefault(int(ri), {"s": 0.0, "n": 0, "y": r["ylabel"]})
                    a["s"] += float(r["dv_sum"])
                    a["n"] += int(r["dv_n"])
        for ri, a in acc.items():
            rows.append({"run_index": ri, "source_file": member, "split": split,
                         "DV": a["s"] / a["n"] if a["n"] else float("nan"),
                         "ylabel": a["y"]})
        print(f"[{fi}/{len(files)}] {member} -> {len(acc)} runs in "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["source_file", "run_index"])
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"s18_labels_{split}.parquet"
    df.reset_index(drop=True).to_parquet(out, index=False)
    print(f"[done] {len(df)} runs -> {out}", flush=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--outdir", default="outputs/s18_experiment")
    a = p.parse_args()
    build(a.split, load_params(), Path(a.outdir))


if __name__ == "__main__":
    main()
