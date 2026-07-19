"""情境 18 —— 串流建立 per-run 特徵表(不解壓 zip)。

防洩漏:只讀 `config/s18_params.yaml` 的訊號欄白名單,**不含 DV / ylabel**。
標籤由檔名決定,寫成 `source_file` 欄供評估腳本 join,特徵管線本身不碰標註。

用法::

    python -m src.s18.build_s18_features --split train
    python -m src.s18.build_s18_features --split test    # Phase 2 之後才准跑
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pandas as pd
import yaml

from src.s18.s18_features import FEATURE_COLUMNS, META_COLUMNS, compute_run_features

# 特徵管線允許讀取的欄位 —— 刻意不含 DV / ylabel(防洩漏鐵律 1)
SIGNAL_COLS = [
    "time", "rod_demand_pos", "rod_actual_pos", "del_pos",
    "torque", "rotor_speed", "run_index", "transitions",
]
CONFIG = Path("config/s18_params.yaml")


def load_params(path: Path = CONFIG) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def iter_runs(zip_path: str, member: str,
              chunksize: int = 400_000) -> Iterator[Tuple[int, pd.DataFrame]]:
    """串流 yield 每個完整的 run,產出後即釋放記憶體。

    run 在檔案中連續出現,故看到新的 run_index 即可判定前一個已完整。
    """
    z = zipfile.ZipFile(zip_path)
    buf: list = []
    cur: int | None = None
    with z.open(member) as f:
        for chunk in pd.read_csv(f, usecols=SIGNAL_COLS, chunksize=chunksize):
            for ri, seg in chunk.groupby("run_index", sort=True):
                ri = int(ri)
                if cur is None:
                    cur = ri
                elif ri != cur:
                    yield cur, pd.concat(buf, ignore_index=True)
                    buf, cur = [], ri
                buf.append(seg)
    if buf and cur is not None:
        yield cur, pd.concat(buf, ignore_index=True)


def build(split: str, params: Dict, outdir: Path) -> pd.DataFrame:
    files = params["data"][f"{split}_files"]
    zip_path = params["data"]["zip_path"]
    rows = []
    for fi, member in enumerate(files, 1):
        t0 = time.perf_counter()
        n = 0
        for ri, run in iter_runs(zip_path, member):
            feat = compute_run_features(run, params)
            feat["run_index"] = ri
            feat["source_file"] = member
            feat["split"] = split
            feat["n_samples"] = len(run)
            rows.append(feat)
            n += 1
            if n % 25 == 0:
                el = time.perf_counter() - t0
                print(f"  [{fi}/{len(files)}] {member}: {n} runs, "
                      f"{el:.0f}s ({el/n:.1f}s/run)", flush=True)
            del run
        print(f"[{fi}/{len(files)}] {member} -> {n} runs in "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    cols = (["split", "source_file", "run_index", "n_samples"]
            + FEATURE_COLUMNS + META_COLUMNS)
    df = df[cols].sort_values(["source_file", "run_index"]).reset_index(drop=True)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"s18_features_{split}.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(outdir / f"s18_features_{split}.csv", index=False)
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
