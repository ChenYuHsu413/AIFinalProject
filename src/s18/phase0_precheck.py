"""情境 18 Phase 0 — 設計書 §3 前置檢核。

抽 3 個 LN run,回答三件事:
  1. run 內是否存在**方向反轉**(決定 BL_DeadZone / BL_ReversalErr 是否有定義)
  2. 取樣率與 ``dead_zone_width`` 內建窗參數(20 samples / 0.05 位移門檻)是否相容
  3. 單位(位置 / 扭矩 / 轉速)

防洩漏:本腳本**不讀 DV 也不讀 ylabel**。LN 段的身分由檔名決定
(``train_load0_*`` 即 LN),不靠標註欄,因此天然滿足鐵律 1。

用法::

    python -m src.s18.phase0_precheck --zip <FMCRD_Data.zip>
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SEED = 42
LN_TRAIN_FILE = "train_load0_1e_m15_200x5.csv"
# 特徵管線允許的欄位(設計書 §1)—— 刻意不含 DV / ylabel
SIGNAL_COLS = [
    "time", "rod_demand_pos", "rod_actual_pos", "del_pos",
    "torque", "rotor_speed", "run_index", "transitions",
]
N_RUNS = 3
DEFAULT_ZIP = "C:/Users/alung/Downloads/FMCRD_Data.zip"


def read_first_runs(zip_path: str, member: str = LN_TRAIN_FILE,
                    n_runs: int = N_RUNS,
                    chunksize: int = 500_000) -> Dict[int, pd.DataFrame]:
    """串流讀取,收齊前 ``n_runs`` 個完整 run 就停(不解壓、不整檔載入)。

    只有當某個 run_index 之後出現更大的 run_index,才認定它已完整。
    """
    parts: Dict[int, List[pd.DataFrame]] = {}
    complete: set = set()
    z = zipfile.ZipFile(zip_path)
    with z.open(member) as f:
        for chunk in pd.read_csv(f, usecols=SIGNAL_COLS, chunksize=chunksize):
            for ri, seg in chunk.groupby("run_index", sort=True):
                parts.setdefault(int(ri), []).append(seg)
            seen_max = max(parts)
            complete = {ri for ri in parts if ri < seen_max}
            if len(complete) >= n_runs:
                break
    keep = sorted(complete)[:n_runs]
    return {ri: pd.concat(parts[ri], ignore_index=True) for ri in keep}


def sampling_summary(run: pd.DataFrame) -> Dict:
    dt = np.diff(run["time"].to_numpy())
    dt_med = float(np.median(dt))
    return {
        "n_samples": int(len(run)),
        "dt_median_s": dt_med,
        "fs_hz": float(1.0 / dt_med) if dt_med > 0 else float("nan"),
        "duration_s": float(run["time"].iloc[-1] - run["time"].iloc[0]),
        "dt_is_uniform": bool(np.allclose(dt, dt_med, rtol=1e-6, atol=1e-12)),
    }


def reversal_summary(run: pd.DataFrame) -> Dict:
    """方向反轉檢核 —— 速度過零(dsp_analytics 用 v[i]*v[i-1] < 0 判定)。

    原始 rotor_speed 含量測雜訊,零附近會抖出大量假過零,因此同時報:
      * ``raw_sign_changes``:dsp_analytics 實際會看到的數字
      * ``significant_reversals``:速度先超過 +thr 再低於 -thr(或反之)的真實反轉
    """
    v = run["rotor_speed"].to_numpy()
    d = run["rod_demand_pos"].to_numpy()

    raw = int(np.sum(v[1:] * v[:-1] < 0))

    # 有意義的反轉:遲滯判定,只數真正換向的次數。
    # 門檻取「5% 全幅」與「雜訊地板」的大者 —— 只用全幅的話,整段沒動的 run
    # 會把門檻縮到雜訊尺度而數出上千次假換向。雜訊由**一階差分**估計:對平滑
    # 的真實運動趨近 0(不誤傷),只有在訊號本身就是雜訊時才會頂起門檻。
    amp = float(np.max(np.abs(v)))
    sigma_n = float(np.median(np.abs(np.diff(v)))) * 1.4826 / np.sqrt(2.0)
    thr = max(0.05 * amp, 4.0 * sigma_n)
    sig_reversals, state = 0, 0
    for x in v:
        if x > thr:
            if state == -1:
                sig_reversals += 1
            state = 1
        elif x < -thr:
            if state == 1:
                sig_reversals += 1
            state = -1

    dd = np.diff(d)
    demand_moves = dd[np.abs(dd) > 0]
    return {
        "raw_sign_changes": raw,
        "significant_reversals": int(sig_reversals),
        "speed_threshold_used": float(thr),
        "rotor_speed_min": float(np.min(v)),
        "rotor_speed_max": float(np.max(v)),
        "demand_monotonic": bool(np.all(demand_moves > 0) or np.all(demand_moves < 0)),
        "demand_step_count": int(np.sum(np.abs(dd) > 1e-9)),
        "demand_min": float(np.min(d)),
        "demand_max": float(np.max(d)),
        "transitions_unique": sorted(int(t) for t in run["transitions"].unique()),
    }


def unit_summary(run: pd.DataFrame) -> Dict:
    out = {}
    for col in ("rod_demand_pos", "rod_actual_pos", "del_pos", "torque", "rotor_speed"):
        a = run[col].to_numpy()
        out[col] = {
            "min": float(np.min(a)), "max": float(np.max(a)),
            "mean": float(np.mean(a)), "std": float(np.std(a)),
        }
    return out


def dead_zone_window_check(fs_hz: float) -> Dict:
    """``dead_zone_width`` 內建往後搜尋 20 samples、位移門檻 0.05。

    這兩個常數是在別的取樣率下訂的;在 50kHz 下 20 samples 只有 0.4 ms,
    可能還沒等到實際位置響應就放棄。此處計算換算結果供判斷。
    """
    return {
        "builtin_search_samples": 20,
        "builtin_move_threshold": 0.05,
        "window_duration_ms": 20.0 / fs_hz * 1e3 if fs_hz > 0 else float("nan"),
    }


def run_precheck(zip_path: str, outdir: Path) -> Dict:
    np.random.seed(SEED)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = read_first_runs(zip_path)

    report: Dict = {"seed": SEED, "source_file": LN_TRAIN_FILE,
                    "label_source": "filename (train_load0 == LN); DV/ylabel not read",
                    "runs": {}}
    for ri, run in runs.items():
        s = sampling_summary(run)
        report["runs"][str(ri)] = {
            "sampling": s,
            "reversal": reversal_summary(run),
            "units": unit_summary(run),
            "dead_zone_window": dead_zone_window_check(s["fs_hz"]),
        }
    _plot(runs, outdir / "fig_phase0_ln_runs.png")
    (outdir / "phase0_precheck.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _plot(runs: Dict[int, pd.DataFrame], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (ri, run) in zip(axes, sorted(runs.items())):
        t = run["time"].to_numpy()
        ax.plot(t, run["rod_demand_pos"], lw=1.0, label="rod_demand_pos", color="#1f77b4")
        ax.plot(t, run["rod_actual_pos"], lw=0.8, label="rod_actual_pos",
                color="#2ca02c", alpha=0.75)
        ax.set_ylabel("position")
        ax2 = ax.twinx()
        ax2.plot(t, run["rotor_speed"], lw=0.6, color="#d62728",
                 alpha=0.65, label="rotor_speed")
        ax2.axhline(0.0, color="k", lw=0.6, ls=":")
        ax2.set_ylabel("rotor_speed")
        ax.set_title(f"LN run_index={ri}  (train_load0)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--outdir", default="outputs/s18_experiment")
    a = p.parse_args()
    rep = run_precheck(a.zip, Path(a.outdir))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
