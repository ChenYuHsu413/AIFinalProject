#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servo AI Dataset v4.1 Enterprise Streaming Generator

重點：
- 支援自行擴增生成 1000 萬筆（10,000,000 rows）或更多
- 使用 chunk 分批產生與寫入，避免記憶體爆掉
- 支援 CSV 單一大檔、CSV 分片、Parquet 分片
- 沿用 vendored `servo_v3_generator`（v4.1 物理）的 30 種 Scenario 與 142 欄
  Servo PLC/Drive Log；seed + chunk_index 可重現。

建議：
- 1000 萬筆以上建議使用 Parquet 分片，速度與檔案大小都比 CSV 好
- 若一定要 CSV，可使用 --split_files，避免產生超巨大單檔

範例（從 repo 根目錄，以模組方式執行）：

1) 生成 1000 萬筆 mixed CSV 單檔：
python -m src.monitor.servo_v3_streaming_generator --rows 10000000 --scenario mixed --format csv --output servo_10M.csv --chunk_rows 100000

2) 生成 1000 萬筆 Parquet 分片，推薦：
python -m src.monitor.servo_v3_streaming_generator --rows 10000000 --scenario mixed --format parquet --output_dir servo_10M_parquet --chunk_rows 100000 --split_files

3) 生成 1000 萬筆 CSV 分片：
python -m src.monitor.servo_v3_streaming_generator --rows 10000000 --scenario mixed --format csv --output_dir servo_10M_csv_parts --chunk_rows 100000 --split_files

4) 生成 Scenario 14 軸承磨耗 1000 萬筆：
python -m src.monitor.servo_v3_streaming_generator --rows 10000000 --scenario 14 --format parquet --output_dir bearing_wear_10M --split_files
"""

import argparse
from pathlib import Path
import json
import pandas as pd

# 匯入 vendored 生成器（v4.1 物理）的核心函式
from src.monitor.servo_v3_generator import generate_servo_data, SCENARIOS, safe_name


def scenario_for_chunk(mode: str, chunk_index: int) -> int:
    if mode == "mixed":
        return (chunk_index % 30) + 1
    return int(mode)


def write_csv_append(df: pd.DataFrame, output_path: Path, write_header: bool):
    df.to_csv(output_path, index=False, mode="a", header=write_header)


def write_parquet_part(df: pd.DataFrame, output_dir: Path, part_index: int, scenario_id: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"part_{part_index:05d}_scenario_{scenario_id:02d}.parquet"
    df.to_parquet(path, index=False)
    return path


def write_csv_part(df: pd.DataFrame, output_dir: Path, part_index: int, scenario_id: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"part_{part_index:05d}_scenario_{scenario_id:02d}.csv"
    df.to_csv(path, index=False)
    return path


def generate_streaming(
    total_rows: int,
    scenario: str,
    file_format: str,
    output: Path,
    output_dir: Path,
    chunk_rows: int,
    sampling_hz: int,
    seed: int,
    split_files: bool,
):
    if scenario != "mixed":
        sid = int(scenario)
        if sid < 1 or sid > 30:
            raise ValueError("--scenario must be 1~30 or mixed")

    if file_format == "parquet" and not split_files:
        raise ValueError("Parquet streaming requires --split_files. Use --format parquet --split_files --output_dir <dir>")

    if not split_files and file_format == "csv":
        if output.exists():
            output.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    rows_done = 0
    chunk_index = 0
    global_start_s = 0.0
    manifest_rows = []

    while rows_done < total_rows:
        n = min(chunk_rows, total_rows - rows_done)
        scenario_id = scenario_for_chunk(scenario, chunk_index)

        df = generate_servo_data(
            rows=n,
            scenario_id=scenario_id,
            sampling_hz=sampling_hz,
            seed=seed + chunk_index,
            global_start_s=global_start_s,
        )

        if split_files:
            if file_format == "csv":
                part_path = write_csv_part(df, output_dir, chunk_index + 1, scenario_id)
            else:
                part_path = write_parquet_part(df, output_dir, chunk_index + 1, scenario_id)
        else:
            part_path = output
            write_csv_append(df, output, write_header=(chunk_index == 0))

        name, category, alarm_code, root_cause = SCENARIOS[scenario_id - 1]
        manifest_rows.append({
            "part_index": chunk_index + 1,
            "file_name": str(part_path.name),
            "scenario_id": scenario_id,
            "scenario_name": name,
            "fault_category": category,
            "alarm_code": alarm_code,
            "rows": n,
            "global_start_s": global_start_s,
            "global_end_s": global_start_s + n / sampling_hz - 1 / sampling_hz,
        })

        rows_done += n
        global_start_s += n / sampling_hz
        chunk_index += 1

        print(f"[{rows_done:,}/{total_rows:,}] wrote {part_path.name} scenario={scenario_id:02d} rows={n:,}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "generation_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "total_rows": total_rows,
        "chunk_rows": chunk_rows,
        "chunks": chunk_index,
        "scenario": scenario,
        "format": file_format,
        "split_files": split_files,
        "sampling_hz": sampling_hz,
        "output": str(output) if not split_files else str(output_dir),
        "manifest": str(manifest_path),
        "note": "For 10M rows, Parquet split files are recommended."
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DONE")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Servo AI Dataset v3.1 10M Streaming Generator")
    parser.add_argument("--rows", type=int, default=10000000, help="total rows to generate, default=10,000,000")
    parser.add_argument("--scenario", default="mixed", help="1~30 or mixed, default=mixed")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet", help="csv or parquet")
    parser.add_argument("--output", default="servo_10M.csv", help="single CSV output path when not using --split_files")
    parser.add_argument("--output_dir", default="servo_10M_output", help="output directory for split files and manifest")
    parser.add_argument("--chunk_rows", type=int, default=100000, help="rows per chunk, default=100,000")
    parser.add_argument("--sampling_hz", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_files", action="store_true", help="write each chunk as separate CSV/Parquet file")
    args = parser.parse_args()

    generate_streaming(
        total_rows=args.rows,
        scenario=args.scenario,
        file_format=args.format,
        output=Path(args.output),
        output_dir=Path(args.output_dir),
        chunk_rows=args.chunk_rows,
        sampling_hz=args.sampling_hz,
        seed=args.seed,
        split_files=args.split_files,
    )


if __name__ == "__main__":
    main()
