"""情境 18 Phase 4 —— 產出 diagnosis JSON（合約格式 × 實證內容）。

結構依 `docs/contracts/DATA_CONTRACT_v1.0.md` §2 的範例（該範例即事實上的規範——
附錄 §6 對 Diagnosis 只寫「見 api/schemas.py」，該檔未隨合約交付，故無正式 schema）。

**內容一律反映實證結果，不反填合約範例：**
  * `root_cause_ranking` 用實際特徵陣容與真實 z-score；**絕不出現 `BL_DeadZone`**
    （Phase 1 證實其在階梯激勵下結構性不可估計）
  * `drive_adjustments` 只出摩擦/剛性路徑；合約範例的 P2-60 背隙補償
    （`set_from_feature_value: 0.12mm`）**不產出**——本工況無反轉激勵，背隙量不可估計
  * `meaning` 依最終措辭紀律撰寫，torque 基特徵一律附負載混淆註記
  * 新增 `data_provenance_warning`（v1.0 schema 外的擴充，已列入 v1.1 提案）

用法::

    python -m src.s18.build_diagnosis_json
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.s18.build_s18_features import load_params
from src.s18.run_phase2 import load_table
from src.s18.s18_analysis import composite_score, ln_baseline, zscores

OUTDIR = Path("outputs/s18_experiment")
SCENARIO_ID = "18_Ball_Screw"          # 合約 §6 L1 enum 原字串

# 措辭紀律:torque 基特徵一律附負載混淆註記（Phase 2 §4 裁決）
MEANING = {
    "STIFF_TorqueSlope": "扭矩對追隨誤差的斜率下降 — 等效剛性降低"
                         "（註：與負載條件存在混合效應）",
    "FR_Coulomb": "低速帶扭矩包絡抬升 — 摩擦增加"
                  "（註：與負載條件存在混合效應）",
    "STIFF_ComplStd": "順應性波動變化 — 剛性穩定度（within-noisy 軌無顯著趨勢，"
                      "僅供排名參考）",
    "BL_ReversalErr_cmd": "指令換向處的位置誤差（within-noisy 軌無訊號；"
                          "背隙路徑於本工況未獲驗證）",
    "BL_ReversalErr_zc": "速度過零處的位置誤差（全類別檢定之顯著性經 within-noisy "
                         "分軌判定為負載混淆假陽性）",
    "BL_HystArea": "demand–actual 包絡面積（階梯激勵下無遲滯迴圈，within-noisy 無訊號）",
    "BL_DirFE_Asym": "正反向追隨誤差不對稱（within-noisy 軌無訊號）",
}


def severity_of(score: float, th: Dict) -> Dict[str, str]:
    """合約 §2 的巢狀 severity。§1 的 HMI 顏色對應:綠/黃/橙/紅。"""
    if score > th["critical"]:
        # 合約 §1 規定 critical = 降速運行 + 通知工程師,但 §2 未列舉 action 的
        # 合法值(範例僅見 adjust_parameters)。此處依 §1 語意命名,並列入 v1.1 提案。
        return {"label": "critical", "action": "reduce_speed_and_notify",
                "hmi_color": "red"}
    if score > th["warning"]:
        return {"label": "warning", "action": "adjust_parameters",
                "hmi_color": "orange"}
    if score > th["watch"]:
        return {"label": "watch", "action": "record_trend", "hmi_color": "yellow"}
    return {"label": "normal", "action": "none", "hmi_color": "green"}


def _drive_adjustments(ranking: List[Dict]) -> List[Dict]:
    """只出摩擦/剛性路徑。通用參數群 + advisory 表述,不寫死可執行的量值。"""
    top = {r["feature"] for r in ranking[:3]}
    out: List[Dict] = []
    if "STIFF_TorqueSlope" in top:
        out.append({
            "param_group": "position_loop_gain", "action": "advisory_review",
            "direction": "increase", "value": None,
            "label": "Position Loop Gain（通用參數群）",
            "note": "等效剛性下降之對應方向;實際參數編號與量值需依驅動器型號與"
                    "現場整定決定,本分析不輸出可直接寫入的數值。"})
    if "FR_Coulomb" in top:
        out.append({
            "param_group": "torque_limit_and_friction_comp",
            "action": "advisory_review", "direction": "review", "value": None,
            "label": "Torque Limit / 摩擦補償（通用參數群）",
            "note": "摩擦增加之對應方向;需先以機械保養排除潤滑因素再考慮參數補償。"})
    return out


def build_one(row: pd.Series, z_row: pd.Series, score: float, params: Dict,
              th: Dict) -> Dict:
    sev = severity_of(score, th)
    ranked = z_row.dropna().reindex(z_row.dropna().abs().sort_values(
        ascending=False).index)
    ranking = [{"rank": i + 1, "feature": f, "z_score": round(float(v), 3),
                "meaning": MEANING.get(f, f)}
               for i, (f, v) in enumerate(ranked.items())]

    drive = _drive_adjustments(ranking)
    return {
        "type": "diagnosis",
        "scenario_id": SCENARIO_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                             .replace("+00:00", "Z"),
        "anomaly_score": round(float(score), 4),
        "is_anomaly": bool(score > th["warning"]),   # 綁 P95,與 Phase 2 同一條線
        "severity": sev,
        "source_run": {"split": "test", "run_index": int(row["run_index"]),
                       "source_file": row["source_file"],
                       "ground_truth_ylabel": row["ylabel"]},
        "root_cause_ranking": ranking,
        "drive_adjustments": drive,
        "plc_adjustments": [],
        "mechanical_checks": [
            "潤滑滾珠螺桿、檢查潤滑週期與油品劣化（對應 FR_Coulomb 上升）",
            "檢查螺帽預壓與支撐軸承預壓（對應等效剛性下降）",
            "檢查滾珠螺桿磨損狀況",
        ],
        "not_estimable": [
            {"quantity": "backlash_compensation",
             "contract_reference": "§2 範例之 P2-60 / D2000（set_from_feature_value）",
             "reason": "本工況（階梯定位循環）無連續反轉激勵,背隙量不可估計;"
                       "BL_DeadZone 於 Phase 1 證實結構上恆為 0,已標不適用。"}
        ],
        "hmi_display": {
            "color": sev["hmi_color"],
            "alarm_lines": [f"{r['feature']} (z={r['z_score']}): "
                            f"{r['meaning'].split('（')[0]}" for r in ranking[:2]],
            "action": sev["action"],
        },
        "fallback": {
            "level": sev["label"],
            "chain": ["model_output"],
            "trigger_alert": False,
            "note": "離線批次分析,無即時 fallback 鏈語意;此欄依 schema 出但不具"
                    "線上意義。",
        },
        "data_provenance_warning": [
            "健康基線取自**零負載** LN 檔案（*_load0）,而 LO/MED/HI 取自 *_noisy;"
            "本 anomaly_score 與 severity 門檻之偵測力為**退化與負載條件的混合效應**,"
            "非純退化指標。",
            "退化推論以 within-noisy 軌（LO/MED/HI,負載條件一致）為準:"
            "FR_Coulomb ρ=0.794、STIFF_TorqueSlope ρ=−0.496（p<0.001）。",
            "背隙（BL）族於本工況無訊號,歸因為激勵條件不支持而非訊號微弱;"
            "情境 18 的背隙路徑**未獲驗證**。",
            "FMCRD 為高擬真物理模擬資料集,非真實工廠/產線伺服馬達遙測。",
        ],
        "provenance": {
            "dataset": "PHM FMCRD (high-fidelity simulation, not factory telemetry)",
            "baseline": "train LN only (n=200, zero-load)",
            "threshold_semantics": params["severity_thresholds"]["provenance_note"].strip(),
            "contract": "docs/contracts/DATA_CONTRACT_v1.0.md §2（範例即事實規範;"
                        "正式 Diagnosis schema 未隨合約交付）",
        },
    }


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    params = load_params()
    th = params["severity_thresholds"]
    prim = params["feature_roster"]["primary"]

    train = load_table("train", OUTDIR, params)
    test = load_table("test", OUTDIR, params)
    base = ln_baseline(train, prim)
    z = zscores(test, base, prim)
    comp = composite_score(z, 3, params["nan_policy"]["min_available_features"])

    # (1) HI / critical —— 取複合分數最高者(即 Phase 2 圖 4 那筆)
    hi = test[test["ylabel"] == "HI"].index
    i_hi = comp[hi].idxmax()
    # (2) MED / warning —— 取落在 warning 帶內、最接近該帶中位數者
    med = test[test["ylabel"] == "MED"].index
    band = comp[med][(comp[med] > th["warning"]) & (comp[med] <= th["critical"])]
    i_med = (band - band.median()).abs().idxmin()

    docs = [build_one(test.loc[i], z.loc[i], float(comp[i]), params, th)
            for i in (i_hi, i_med)]
    out = OUTDIR / "diagnosis_sample_18.json"
    out.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")

    for d in docs:
        print(f"[{d['severity']['label']:>8}] run {d['source_run']['run_index']} "
              f"({d['source_run']['ground_truth_ylabel']}) "
              f"score={d['anomaly_score']} top={d['root_cause_ranking'][0]['feature']}"
              f" z={d['root_cause_ranking'][0]['z_score']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
