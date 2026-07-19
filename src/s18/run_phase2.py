"""情境 18 Phase 2 —— 設計書 §4.1–4.4 分析與圖 1–4。

執行紀律:
  * 基線只從 **train 的 LN 段**計算(鐵律 2)
  * primary_variant 已於 train 選定並鎖進 config(鐵律 3:test 不參與任何選擇)
  * `--split test` 只准跑一次

用法::

    python -m src.s18.run_phase2 --split train    # 開發驗證
    python -m src.s18.run_phase2 --split test     # 正式評估,只跑一次
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.s18.build_s18_features import load_params
from src.s18.s18_analysis import (
    within_noisy,
    LEVEL_ORDER,
    auc_with_ci,
    composite_score,
    ln_baseline,
    monotonicity,
    partial_spearman,
    spearman_permutation,
    trigger_rates,
    zscores,
)

EXPECTED_DIRECTION = {         # 設計書 §2 的預期方向
    "BL_ReversalErr_cmd": "up", "BL_ReversalErr_zc": "up",
    "BL_HystArea": "up", "BL_DirFE_Asym": "up",
    "STIFF_TorqueSlope": "down", "STIFF_ComplStd": "up",
    "FR_Coulomb": "up",
}


def load_table(split: str, outdir: Path, params: Dict) -> pd.DataFrame:
    feats = pd.read_parquet(outdir / f"s18_features_{split}.parquet")
    labs = pd.read_parquet(outdir / f"s18_labels_{split}.parquet")
    df = feats.merge(labs[["run_index", "source_file", "DV", "ylabel"]],
                     on=["run_index", "source_file"], how="left", validate="1:1")
    if df["ylabel"].isna().any():
        raise ValueError("特徵表與標註表 join 後出現缺標籤")
    return df


def analyse(df: pd.DataFrame, train_df: pd.DataFrame, params: Dict,
            outdir: Path, split: str) -> Dict:
    roster = params["feature_roster"]
    stat = params["statistics"]
    nperm, nboot = stat["permutation_n"], stat["auc_bootstrap_n"]
    primary = roster["primary"]
    analysed = primary + roster["exploratory"] + roster["control"]
    m: Dict = {"split": split, "n_runs": int(len(df)),
               "n_by_level": df["ylabel"].value_counts().reindex(LEVEL_ORDER)
                               .fillna(0).astype(int).to_dict(),
               "primary_variant": params["primary_variant_preregistration"]
                                        ["primary_variant"]}

    # ---- 4.1 單調性 ----
    mono = {}
    for f in analysed:
        r = monotonicity(df, f, n_perm=nperm)
        r["expected"] = EXPECTED_DIRECTION.get(f)
        r["direction_ok"] = bool(r["expected"] is None
                                 or r["direction"] == r["expected"])
        r["pass"] = bool(r["p_perm"] < 0.01 and r["direction_ok"]
                         and f in primary)
        r["low_coverage_flag"] = bool(r["coverage"]
                                      < params["nan_policy"]["coverage_flag_threshold"])
        mono[f] = r
    m["monotonicity"] = mono
    m["n_primary_passing"] = int(sum(v["pass"] for v in mono.values()))
    m["criterion_4_1"] = f"{m['n_primary_passing']}/7 主力特徵 p<0.01 且方向符合（門檻 ≥4）"

    # ---- 主要推論軌:within-noisy(無負載混淆)----
    m["within_noisy"] = within_noisy(df, analysed, n_perm=nperm, n_boot=nboot)
    m["inference_note"] = (
        "主要推論為 within-noisy（LO/MED/HI，負載條件一致）；"
        "LN 錨定的 AUC 與觸發率為操作性偵測指標，含退化與負載的混合效應。")

    # ---- 4.2 與 DV 的相關 ----
    m["dv_spearman"] = {f: spearman_permutation(df[f], df["DV"], n_perm=nperm)
                        for f in analysed}

    # ---- 4.3 可分性 ----
    ln = df[df["ylabel"] == "LN"]
    sep = {}
    for f in analysed:
        sep[f] = {lvl: auc_with_ci(df[df["ylabel"] == lvl][f], ln[f],
                                   n_boot=nboot)
                  for lvl in ["LO", "MED", "HI"] if (df["ylabel"] == lvl).any()}
    m["separability"] = sep

    # ---- _zc 混淆控制(config 要求)----
    if params["primary_variant_preregistration"]["zc_confound_controls"]["partial_correlation"]:
        ordinal = df["ylabel"].map({l: i for i, l in enumerate(LEVEL_ORDER)})
        d2 = df.assign(_ord=ordinal)
        m["zc_partial_correlation"] = {
            f: partial_spearman(d2, f, "_ord", "n_zero_crossings", n_perm=nperm)
            for f in ["BL_ReversalErr_zc", "BL_ReversalErr_cmd"]}

    # ---- 4.4 基線 + 複合異常分數 ----
    base = ln_baseline(train_df, primary)
    z = zscores(df, base, primary)
    comp = composite_score(z, top_k=3,
                           min_available=params["nan_policy"]["min_available_features"])
    z_train = zscores(train_df[train_df["ylabel"] == "LN"], base, primary)
    comp_train_ln = composite_score(z_train, top_k=3,
                                    min_available=params["nan_policy"]["min_available_features"])
    thr = float(np.nanpercentile(comp_train_ln, 95))
    m["baseline"] = {"source": "train LN only",
                     "threshold_p95": thr,
                     "stats": base.to_dict()}
    m["trigger_rates"] = trigger_rates(comp, df["ylabel"], thr).to_dict()

    _figures(df, z, comp, base, thr, outdir, split, params)
    return m


def _fetch_run(zip_path: str, member: str, target_ri: int):
    """回讀指定 run 的原始波形(圖 1 用;分析本身不需要)。"""
    from src.s18.build_s18_features import iter_runs
    for ri, run in iter_runs(zip_path, member):
        if ri == target_ri:
            return run
    return None


def _setup_fonts() -> None:
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _figures(df, z, comp, base, thr, outdir: Path, split: str,
             params: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    _setup_fonts()
    import matplotlib.pyplot as plt

    levels = [l for l in LEVEL_ORDER if (df["ylabel"] == l).any()]
    colors = {"LN": "#2ca02c", "LO": "#ffbb33", "MED": "#ff7f0e", "HI": "#d62728"}

    zip_path = params["data"]["zip_path"]

    def _rep_run(lvl: str, by: str):
        """取該等級中 `by` 特徵最接近中位數的代表性 run,並回讀原始波形。"""
        sub = df[df["ylabel"] == lvl]
        if sub.empty:
            return None, None
        med = sub[by].median()
        row = sub.iloc[int((sub[by] - med).abs().to_numpy().argmin())]
        return row, _fetch_run(zip_path, row["source_file"], int(row["run_index"]))

    # ---- 圖 1(主圖):Stribeck 平面 —— 讓人肉眼看見獲勝特徵在量什麼 ----
    reps = {lvl: _rep_run(lvl, "FR_Coulomb") for lvl in ["LN", "HI"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cb = float(params["friction"]["coulomb_speed_max"])
    for ax, lvl in zip(axes, ["LN", "HI"]):
        row, run = reps[lvl]
        if run is None:
            continue
        v = run["rotor_speed"].to_numpy()[::20]
        tq = run["torque"].to_numpy()[::20]
        ax.axvspan(-cb, cb, color="#cccccc", alpha=0.45,
                   label=f"庫倫帶 |v|<{cb:g}")
        ax.scatter(v, tq, s=2, alpha=0.25, color=colors[lvl])
        lo = np.abs(v) < cb
        ax.axhline(float(np.mean(np.abs(tq[lo]))), color="k", ls="--", lw=1.2,
                   label=f"庫倫帶 mean|torque| = {row['FR_Coulomb']:.3f}")
        ax.set_xlabel("rotor_speed")
        ax.set_ylabel("torque")
        ax.set_title(f"{lvl} run {int(row['run_index'])}  "
                     f"FR_Coulomb={row['FR_Coulomb']:.3f}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    ylim = (min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes))
    for ax in axes:
        ax.set_ylim(ylim)
    fig.suptitle(f"圖 1:Stribeck 平面（torque–velocity），LN vs HI（{split}）"
                 f"—— 低速帶 |torque| 整體抬升即為 FR_Coulomb 所量")
    fig.tight_layout()
    fig.savefig(outdir / f"fig1_stribeck_{split}.png", dpi=130)
    plt.close(fig)

    # ---- 圖 1b(副面板):FE–torque 斜率 —— STIFF_TorqueSlope 的視覺化 ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for lvl in ["LN", "HI"]:
        row, run = reps[lvl]
        if run is None:
            continue
        fe = (run["rod_actual_pos"] - run["rod_demand_pos"]).to_numpy()[::20]
        tq = run["torque"].to_numpy()[::20]
        ax.scatter(fe, tq, s=2, alpha=0.18, color=colors[lvl])
        k, b = np.polyfit(fe, tq, 1)
        xs = np.linspace(fe.min(), fe.max(), 50)
        ax.plot(xs, k * xs + b, color=colors[lvl], lw=2.2,
                label=f"{lvl} run {int(row['run_index'])}：斜率 {k:.3f}")
    ax.set_xlabel("following error (actual − demand)")
    ax.set_ylabel("torque")
    ax.set_title(f"圖 1b:FE–torque 斜率對照（{split}）—— STIFF_TorqueSlope 所量")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"fig1b_fe_torque_slope_{split}.png", dpi=130)
    plt.close(fig)

    # ---- 附錄圖:demand–actual 相圖(階梯激勵,無遲滯迴圈)----
    # 保留為「為何迴圈類特徵不適用」的直接證據,與 BL_DeadZone 結構性 N/A 同節。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, lvl in zip(axes, ["LN", "HI"]):
        row, run = _rep_run(lvl, "BL_HystArea")
        if run is None:
            continue
        d = run["rod_demand_pos"].to_numpy()
        a = run["rod_actual_pos"].to_numpy()
        ax.plot(d - d.mean(), a - a.mean(), lw=0.5, color=colors[lvl], alpha=0.8)
        ax.set_xlabel("rod_demand_pos（去均值）")
        ax.set_ylabel("rod_actual_pos（去均值）")
        ax.set_title(f"{lvl} run {int(row['run_index'])}  "
                     f"HystArea={row['BL_HystArea']:.5f}")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"附錄圖:demand–actual 相圖（階梯激勵，無遲滯迴圈）（{split}）\n"
                 f"指令為分段常數階梯，軌跡呈階梯狀而非迴圈；"
                 f"迴圈類特徵需連續軌跡激勵，此即其不適用之直接證據")
    fig.tight_layout()
    fig.savefig(outdir / f"figA_phase_plot_{split}.png", dpi=130)
    plt.close(fig)

    # 圖 2:箱型圖
    feats = list(EXPECTED_DIRECTION) + ["FE_RMS"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, f in zip(axes.ravel(), feats):
        data = [df[df["ylabel"] == l][f].dropna() for l in levels]
        bp = ax.boxplot(data, labels=[f"{l}\nn={len(d)}" for l, d in zip(levels, data)],
                        patch_artist=True, showfliers=False)
        for patch, l in zip(bp["boxes"], levels):
            patch.set_facecolor(colors[l])
            patch.set_alpha(0.65)
        ax.set_title(f, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle(f"圖 2:特徵 × 退化等級箱型圖({split})")
    fig.tight_layout()
    fig.savefig(outdir / f"fig2_boxplots_{split}.png", dpi=130)
    plt.close(fig)

    # 圖 3:特徵 vs DV 散佈
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, f in zip(axes.ravel(), feats):
        for l in levels:
            s = df[df["ylabel"] == l]
            ax.scatter(s["DV"], s[f], s=8, alpha=0.5, c=colors[l], label=l)
        r = spearman_permutation(df[f], df["DV"], n_perm=200)
        ax.set_title(f"{f}  ρ={r['rho']:.3f} (n={r['n']})", fontsize=9)
        ax.set_xlabel("DV")
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle(f"圖 3:特徵 vs DV({split})")
    fig.tight_layout()
    fig.savefig(outdir / f"fig3_dv_scatter_{split}.png", dpi=130)
    plt.close(fig)

    # 圖 4:某個 HI run 的 z-score 根因排名
    hi = df[df["ylabel"] == "HI"]
    fig, ax = plt.subplots(figsize=(9, 5))
    if not hi.empty:
        i = comp[hi.index].idxmax()
        row = z.loc[i].dropna().sort_values(key=np.abs, ascending=True)
        ax.barh(row.index, row.values,
                color=["#d62728" if v > 0 else "#1f77b4" for v in row.values])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"圖 4:root-cause z-score 排名 — HI run "
                     f"{int(df.loc[i,'run_index'])}(複合分數 {comp[i]:.2f},門檻 {thr:.2f})")
        ax.set_xlabel("z-score (vs train LN 基線)")
    fig.tight_layout()
    fig.savefig(outdir / f"fig4_zscore_ranking_{split}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--outdir", default="outputs/s18_experiment")
    a = p.parse_args()
    params = load_params()
    outdir = Path(a.outdir)

    train_df = load_table("train", outdir, params)
    df = train_df if a.split == "train" else load_table(a.split, outdir, params)
    m = analyse(df, train_df, params, outdir, a.split)

    out = outdir / (f"metrics_s18.json" if a.split == "test"
                    else f"metrics_s18_{a.split}.json")
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(json.dumps({k: m[k] for k in
                      ["split", "n_runs", "n_by_level", "primary_variant",
                       "criterion_4_1"]}, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
