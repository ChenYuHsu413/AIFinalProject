"""情境 18 Phase 3 —— 增量資訊消融（設計書 §4.5）。

**不改動 repo 既有模組**:沿用 `train_servo.run(out_dir=...)`(寫進候選目錄,
不碰 active)與 `validation_gate.run_gate`;變體以**執行期 config 覆寫**實現
(monkeypatch 該模組匯入的 `load_config`),`FEATURE_SETS` 只新增鍵、不改既有鍵。

三個回合(全部預先定義於 `config/s18_params.yaml` `ablation` 節):
  * `base` —— repo 現有 full(21 維),用於確認能複現 0.819 基準
  * `A`    —— 預註冊原案,全部候選特徵入列(31 維)
  * `B`    —— train 診斷通過者(23 維);選擇依據為 train Phase 2 診斷,非 test 結果

NaN 處理:train 中位數填補 + `<feature>_isna` 指示欄。填補統計量只從 train 計算。

用法::

    python -m src.s18.run_ablation
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from src.s18.build_s18_features import load_params
from src.utils.paths import load_config, resolve

S18_DIR = Path("outputs/s18_experiment")
AUG_PARQUET = S18_DIR / "servo_features_s18_augmented.parquet"


def build_augmented_table(params: Dict) -> pd.DataFrame:
    """repo 特徵表 ⨝ s18 特徵表,含 NaN 填補與指示欄。

    **join key 陷阱**:repo 表的 `run_index` 並非原始 CSV 的 run 編號,而是
    `build_servo_from_zip` 以 `sort_values(["split","ylabel","__ri"])` 排序後
    `reset_index` 產生的**全域列號 0..N-1**(原始編號 `__ri` 未保留)。
    直接用 run_index 對接會錯位。此處重建同一套排序鍵,並以
    `FE_RMS` vs `position_error_rms`(兩表對同一物理量的獨立計算)交叉驗證。
    """
    cfg = load_config()
    repo = pd.read_parquet(resolve(cfg["servo"]["processed_features"]))
    label_map = {**params["data"]["train_files"], **params["data"]["test_files"]}
    s18 = pd.concat([
        pd.read_parquet(S18_DIR / f"s18_features_{sp}.parquet") for sp in ("train", "test")
    ], ignore_index=True)
    s18["ylabel"] = s18["source_file"].map(label_map)

    # 重建 repo 的全域列號
    s18 = (s18.sort_values(["split", "ylabel", "run_index"])
              .reset_index(drop=True)
              .rename(columns={"run_index": "orig_run_index"}))
    s18["run_index"] = s18.index

    extra = sorted({f for v in params["ablation"]["variants"].values()
                    for f in v["extra_features"]})
    merged = repo.merge(s18[["split", "run_index", "ylabel", "FE_RMS"] + extra],
                        on=["split", "run_index"], how="left",
                        suffixes=("", "_s18"), validate="1:1")
    if len(merged) != len(repo):
        raise ValueError(f"join 後列數改變:{len(repo)} -> {len(merged)}")

    # --- join 正確性驗證(必須全對,否則整個消融無效)---
    if (merged["ylabel"] != merged["ylabel_s18"]).any():
        raise ValueError("join 後 ylabel 不一致 -> 對應鍵重建失敗")
    dev = (merged["FE_RMS"] - merged["position_error_rms"]).abs()
    if not (dev < 1e-6).all():
        raise ValueError(
            f"join 驗證失敗:FE_RMS 與 position_error_rms 最大偏差 {dev.max():.3g}"
            "（兩表對同一物理量的獨立計算應一致）")
    print(f"[aug] join 驗證通過:{len(merged)} 段 ylabel 全對、"
          f"FE_RMS≡position_error_rms（最大偏差 {dev.max():.2e}）", flush=True)
    merged = merged.drop(columns=["ylabel_s18", "FE_RMS"])

    # NaN:train 中位數填補 + 指示欄(填補統計量只從 train 計算)
    tr = merged["split"] == "train"
    added: List[str] = []
    for f in extra:
        if merged[f].isna().any():
            merged[f"{f}_isna"] = merged[f].isna().astype(int)
            added.append(f"{f}_isna")
            merged[f] = merged[f].fillna(float(merged.loc[tr, f].median()))
    if merged[extra].isna().any().any():
        raise ValueError("填補後仍有 NaN")

    S18_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(AUG_PARQUET, index=False)
    print(f"[aug] {len(merged)} 段、{len(extra)} 個 s18 特徵"
          f"（指示欄 {added or '無'}）-> {AUG_PARQUET}", flush=True)
    return merged


def register_feature_sets(params: Dict, augmented: pd.DataFrame) -> Dict[str, int]:
    """把變體特徵組**新增**進 FEATURE_SETS(不改既有鍵)。"""
    from src.features import servo_features as SF
    full = SF.feature_set_columns("full")
    dims = {"base": len(full)}
    for key, v in params["ablation"]["variants"].items():
        cols = list(full)
        for f in v["extra_features"]:
            cols.append(f)
            if f"{f}_isna" in augmented.columns:
                cols.append(f"{f}_isna")
        name = f"s18_{key}"
        if name in SF.FEATURE_SETS:
            raise KeyError(f"{name} 已存在,拒絕覆寫既有特徵組")
        SF.FEATURE_SETS[name] = {"label": v["name"], "desc": v["name"],
                                 "columns": cols}
        dims[key] = len(cols)
    return dims


def run_variant(key: str, feature_set: str, outdir: Path) -> Dict:
    """以覆寫後的 config 跑一次 train_servo，再過驗證閘門。"""
    from src.models import train_servo
    from src.pipeline.validation_gate import run_gate

    base_cfg = load_config()
    variant_cfg = yaml.safe_load(yaml.safe_dump(base_cfg))       # deep copy
    variant_cfg["servo"]["processed_features"] = str(AUG_PARQUET)
    variant_cfg["servo"]["reference_feature_set"] = feature_set

    cand = outdir / f"candidate_{key}"
    if cand.exists():
        shutil.rmtree(cand)
    cand.mkdir(parents=True)

    original = train_servo.load_config
    train_servo.load_config = lambda *a, **k: variant_cfg    # 執行期覆寫
    try:
        # 設計書 §4.5 的基準是「LR + balanced + full」——三個變體必須釘死同一個
        # 模型家族,否則比較的是模型而非特徵(首次執行未釘,變體 B 的 CV 選到 MLP,
        # 使比較失效;此為協定修正,非依結果調整)。
        train_servo.run(out_dir=cand,
                        data_config={"fixed_clf_model": "logistic_regression"})
    finally:
        train_servo.load_config = original

    metrics = json.loads((cand / "metrics.json").read_text(encoding="utf-8"))

    # 閘門有兩種讀法,都報:
    #  (1) production —— 以線上 config 的特徵契約驗;變體宣告了不同特徵組,
    #      completeness 會 FAIL。這不是缺陷而是實情:要上線就得改契約。
    #  (2) contract-matched —— 以變體自身的 config 驗,才測得到 smoke/指標/AE。
    gate_prod = run_gate(cand, write_report=True)

    import src.pipeline.validation_gate as VG
    orig_vg = VG.load_config
    VG.load_config = lambda *a, **k: variant_cfg
    try:
        gate_var = run_gate(cand, write_report=False)
    except Exception as exc:                    # smoke 固件(feature_demo)未含 s18 欄位
        gate_var = {"passed": False, "checks": [],
                    "error": f"{type(exc).__name__}: {exc}"}
    finally:
        VG.load_config = orig_vg

    return {"feature_set": feature_set, "metrics": metrics,
            "gate_production": {
                "passed": bool(gate_prod["passed"]),
                "checks": {c["name"]: c["status"] for c in gate_prod["checks"]},
                "detail": {c["name"]: c.get("detail") for c in gate_prod["checks"]}},
            "gate_contract_matched": {
                "passed": bool(gate_var["passed"]),
                "checks": {c["name"]: c["status"] for c in gate_var.get("checks", [])},
                "detail": {c["name"]: c.get("detail") for c in gate_var.get("checks", [])},
                "error": gate_var.get("error")},
            "candidate_dir": str(cand)}


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(S18_DIR / "ablation"))
    a = p.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = load_params()
    aug = build_augmented_table(params)
    dims = register_feature_sets(params, aug)

    results = {}
    for key in ["base", "A", "B"]:
        fs = "full" if key == "base" else f"s18_{key}"
        print(f"\n===== 變體 {key}（{fs}，{dims[key]} 維）=====", flush=True)
        results[key] = run_variant(key, fs, outdir)
        results[key]["dims"] = dims[key]

    out = outdir / "ablation_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
