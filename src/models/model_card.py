"""Auto-generate a Model Card markdown for the persisted best model.

Inspired by the Google "Model Cards for Model Reporting" paper. This writes
``outputs/models/MODEL_CARD.md`` with the headline metrics, training data
summary, intended use / out-of-scope statements and known limitations so that
the artefact stays in sync with what is actually deployed.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from src.utils.paths import load_config, resolve


def _safe_metric(metrics: Dict[str, Any], key: str) -> str:
    if key in metrics and metrics[key] is not None:
        try:
            return f"{float(metrics[key]):.4f}"
        except (TypeError, ValueError):
            return str(metrics[key])
    return "—"


def _load_bundle() -> Dict[str, Any]:
    cfg = load_config()
    p = resolve(cfg["paths"]["best_model"])
    return joblib.load(p)


def _confusion_block(cm: List[List[int]] | None) -> str:
    if not cm:
        return "（尚未產生混淆矩陣，請執行 `python -m src.models.evaluate`）"
    tn, fp = cm[0]
    fn, tp = cm[1]
    return (
        "| | pred 0 (健康) | pred 1 (故障) |\n"
        "| --- | ---: | ---: |\n"
        f"| **true 0 (健康)** | {tn} | {fp} |\n"
        f"| **true 1 (故障)** | {fn} | {tp} |"
    )


def _params_block(clf) -> str:
    params = {
        k: v for k, v in clf.get_params(deep=False).items()
        if not k.startswith("_") and not callable(v)
    }
    rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(params.items()))
    return "| 參數 | 值 |\n| --- | --- |\n" + rows


def _feature_importance_block(metrics_dir: Path) -> str:
    fi_path = metrics_dir / "feature_importance_permutation.csv"
    if not fi_path.exists():
        return "（尚未產生 permutation importance — 請執行 `python -m src.models.evaluate`）"
    df = pd.read_csv(fi_path).head(8)
    if "importance_mean" not in df.columns:
        return "（permutation importance 檔案格式不符）"
    rows = "\n".join(
        f"| {r['feature']} | {r['importance_mean']:+.4f} |"
        for _, r in df.iterrows()
    )
    return "| 特徵 | importance_mean (F1) |\n| --- | ---: |\n" + rows


def _tuning_block(cfg: dict, meta: Dict[str, Any]) -> str:
    tuned_path = resolve(cfg["paths"]["tuned_params"])
    if not tuned_path.exists():
        return "（尚未執行 `python -m src.models.tune`）"
    try:
        tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
    except Exception:
        return "（無法解析 tuned_params.json）"
    if not tuned:
        return "（tuned_params.json 為空）"
    is_tuned = bool(meta.get("tuned"))
    header = (
        "本模型為**調參後**的結果。" if is_tuned
        else "已執行調參但**未勝過**原始最佳模型，仍採用原始版本。"
    )
    rows = "\n".join(
        f"| {t['model_name']} | {t['feature_set']} | "
        f"{t['best_cv_score']:.4f} | "
        f"{t['test_metrics'].get('f1', 0):.4f} | "
        f"`{json.dumps(t['best_params'], ensure_ascii=False)}` |"
        for t in tuned
    )
    return (
        f"{header}\n\n"
        "| 模型 | 特徵組合 | best_cv_f1 | test_f1 | best_params |\n"
        "| --- | --- | ---: | ---: | --- |\n" + rows
    )


def generate() -> Path:
    cfg = load_config()
    bundle = _load_bundle()
    meta_path = resolve(cfg["paths"]["best_model_meta"])
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    eval_path = resolve(cfg["paths"]["outputs_metrics"]) / "best_model_eval.json"
    eval_payload: Dict[str, Any] = {}
    if eval_path.exists():
        try:
            eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            eval_payload = {}

    cm = eval_payload.get("confusion_matrix")
    metrics = bundle.get("metrics", {})
    clf = bundle["pipeline"].named_steps["clf"]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_path = resolve(cfg["paths"]["model_card"])

    md = f"""# Model Card — AI 伺服馬達預測性維護原型

> 本文件由 `src/models/model_card.py` 在執行 `python -m src.models.evaluate` 時
> 自動產生，避免手寫文件與實際部署模型脫節。

## 1. 模型基本資訊

| 項目 | 值 |
| --- | --- |
| 模型名稱 | **{bundle.get('model_name', '—')}** |
| 模型類別 | `{type(clf).__name__}` |
| 特徵組合 | **{bundle.get('feature_set', '—')}** |
| 訓練資料 | UCI AI4I 2020（合成資料集） |
| 主要目標 | `Machine failure`（二元分類） |
| 是否經過 Optuna 調參 | {"是" if meta.get("tuned") else "否（採用原始最佳）"} |
| 模型卡產生時間 | {now} |

## 2. 預期用途與範圍

### ✅ 預期用途
- 在「**目前運轉條件**」下估計伺服馬達故障風險，輸出機率、健康分數、
  風險等級與規則式維護建議。
- 配合第二階段故障類型分類器，提示**可能的故障模式**
  （TWF / HDF / PWF / OSF / RNF）。
- 作為**維護工程師的決策輔助**，協助排程巡檢與保養。

### ❌ 不適用情境
- 直接控制馬達運轉或安全互鎖（本系統**不**發送控制命令）。
- 精準預測剩餘壽命（RUL）：AI4I 2020 不含時間序列／run-to-failure 資料。
- 未經實際工廠資料驗證的高風險場域部署。

## 3. 訓練細節

### 特徵欄位（{len(bundle.get('feature_columns', []))} 個）
{", ".join(f"`{c}`" for c in bundle.get('feature_columns', []))}

### 模型超參數
{_params_block(clf)}

## 4. 測試集評估結果

### 主要指標
| 指標 | 值 |
| --- | ---: |
| Accuracy | {_safe_metric(metrics, 'accuracy')} |
| Precision | {_safe_metric(metrics, 'precision')} |
| Recall | {_safe_metric(metrics, 'recall')} |
| F1 | {_safe_metric(metrics, 'f1')} |
| ROC-AUC | {_safe_metric(metrics, 'roc_auc')} |
| PR-AUC | {_safe_metric(metrics, 'pr_auc')} |

> **為什麼不只看 Accuracy**：AI4I 故障比例 ~3.4%，「全部猜健康」就有 ~97%
> Accuracy 但完全沒有實用價值。預測性維護情境通常以 Recall + PR-AUC
> 為主軸。

### 混淆矩陣
{_confusion_block(cm)}

### Permutation Importance（前 8）
{_feature_importance_block(resolve(cfg["paths"]["outputs_metrics"]))}

## 5. 超參數調整摘要

{_tuning_block(cfg, meta)}

## 6. 已知限制

- **合成資料**。AI4I 2020 來自參數化模型，絕對的指標數值不能直接外推到
  真實工廠的伺服馬達。
- **無時間序列**。缺乏 run-to-failure 軌跡，無法做嚴謹 RUL 預測。
- **決策門檻**預設為 0.5。在實際成本模型出現前，可在 Streamlit 的
  「互動式決策門檻調整」頁面探索較低門檻（提高 Recall）。
- **訓練樣本不平衡**。正樣本約 3.4%，所有指標需與基準率一同解讀。
- **RNF 類別**：依資料集設計屬隨機事件，目前模型在此類別上接近隨機猜測，
  屬於合理結果而非模型瑕疵。

## 7. 倫理與安全考量

- 本系統提供**維護決策輔助**；最終派工、停機、換刀的決策由維護工程師
  做出。建議在公司流程中明確標示「AI 建議」與「人類決定」的界線。
- 模型偏差（如對某些 Type 類別表現差異）可在 `outputs/metrics/`
  進一步分析；正式上線前建議在實際機台分群檢驗。

## 8. 維護與重訓建議

- 若取得實際感測資料（電流、電壓、震動、溫度、警報碼、維修紀錄），
  建議**重新訓練**並改以 survival analysis 或 RUL 模型替代本原型。
- 建議每月或每次製程變更後重跑：
  - `python -m src.models.train`
  - `python -m src.models.evaluate`
  - `python -m src.models.train_failure_types`
  - `python -m src.models.tune`（如需要重新調參）

## 9. 相關檔案

- 模型 binary：`outputs/models/best_model.joblib`
- 模型 metadata：`outputs/models/best_model_meta.json`
- 評估報告：`outputs/metrics/best_model_eval.json`
- 跨模型比較：`outputs/metrics/model_comparison.csv`
- 調參 trial 記錄：`outputs/metrics/tuning_history.csv`
- 二階段故障類型指標：`outputs/metrics/failure_type_comparison.csv`
- 圖表：`outputs/figures/`
"""
    out_path.write_text(md, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = generate()
    print(f"模型卡寫入 {path}")
