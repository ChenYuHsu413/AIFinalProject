# 模組 C 規劃 — Paderborn 馬達電流 + 振動故障診斷

> **定位**：補上 A/B/B+ 三軌都缺的**馬達電流（MCSA）模態**，最貼近「馬達」主題。
> Paderborn（Lessmeier et al., 2016, KAt-DataCenter）是**故障分類**資料集（非 RUL），
> 含**電流 + 振動 + 溫度/轉速/扭矩**，且同時有**人工**（EDM/雕刻）與**真實**（加速壽命）損傷。
> 結果回寫 [`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)；資料取捨見 [`DATASET_EVALUATION.md`](DATASET_EVALUATION.md)。
>
> **狀態（2026-06-24）**：**步 0–5 全部完成，並已於真實資料上跑通**。
> 程式：`load_paderborn.py` / `build_paderborn_dataset.py` / `train_paderborn.py` + 8 項單元測試（全綠）；
> Dashboard 模組 C 頁與四軌文件回寫已完成。**真實資料結果**（1 工況 N15_M07_F10、22 軸承碼、440 量測）：
> **baseline 5 折 CV macro-F1 = 1.00**（RF/SVM）、**人工→真實泛化 macro-F1 = 0.20**（acc 0.24）、
> **落差 0.80** —— 人工故障訓練在真實損傷上幾乎泛化失敗（與 Paderborn 文獻一致的 domain shift）。

---

## 0. 一句話定位

> 用「健康 + **人工**故障」軸承訓練分類器，測在「**真實**加速壽命損傷」軸承上 ——
> 量化「人工故障訓練能否泛化到真實故障」（domain shift），與 B+ 的 LOBO vs LOCO 同構。

---

## 1. 資料事實（子集 MVP）

| 項目 | 值 |
| --- | --- |
| 檔案 | `.mat`（MATLAB struct），檔名 `{COND}_{CODE}_{idx}.mat`，每軸承每工況 20 筆 |
| 通道（MVP 取用） | `vibration_1`、`phase_current_1`、`phase_current_2`，皆 64 kHz |
| 軸承碼 | 健康 `K001–K006`；人工 `KA*`(外環)/`KI*`(內環)；真實 `KA04/KA15/KB23/KI04…` |
| 工況 | 4 種（`N15_M07_F10` 等）；**MVP 先用 1 工況** |
| 標籤 | `fault_class ∈ {healthy, outer, inner}`、`damage_origin ∈ {healthy, artificial, real}` |
| 解析 | `scipy.io.loadmat(squeeze_me, struct_as_record=False)`，訊號在 `Y` 具名陣列 |

> `scipy` 已是相依，**無需新增套件**。

---

## 2. 範圍決策

| 決策 | 取捨 |
| --- | --- |
| 任務＝**分類**（非 RUL） | Paderborn 非 run-to-failure；不做 RUL 回歸（延續紅線） |
| 模態＝**振動 + 電流（時域特徵）** | 賣點是電流；MCSA 頻譜邊帶（真正的電流診斷特徵）列**後續加值** |
| 切分＝**人工→真實**（非隨機） | 隨機切分會高估；人工→真實才是有研究價值的 domain-shift 量測 |
| 規模＝**1 工況子集 MVP** | 先跑通，碼/工況由 config 增減；raw 不進 git、附下載說明 |
| 服務＝**Dashboard 讀 artifacts** | FastAPI 即時預測端點列後續，不在本次 |

---

## 3. 可重用 vs 需新增

**可直接重用（不改）**
- `src/features/vibration_features.py::time_domain_features`（對振動與每相電流各抽一組時域特徵）。
- `src/models/model_registry.py`（`build` / `available_models` / `NEEDS_SCALING`）、
  `src/data/preprocess.py::make_column_transformer`（純數值、依模型決定是否標準化）。
- `src/utils/paths.py`、`src/ui/charts.py::confusion_heatmap`、`style.*`。

**需新增（已完成步 0–3）**
- `src/data/load_paderborn.py`：`parse_measurement_name` / `list_paderborn_files` / `load_paderborn_mat`。
- `src/data/build_paderborn_dataset.py`：`bearing_label_map` / `extract_paderborn_features` /
  `build_feature_table` / `main`（→ `paderborn_features.parquet`）。
- `src/models/train_paderborn.py`：`feature_columns` / `split_artificial_real` / `run`
  （baseline 分層 CV 選最佳模型 + 人工→真實泛化；輸出對照 JSON / 模型 / 預測）。
- `tests/test_paderborn_features.py`：檔名解析、標籤映射、特徵欄位/值、artificial/real 切分（合成資料）。

**待辦（步 4–5）**
- `app/streamlit_app.py`：模組 C 一頁「馬達電流故障診斷」（KPI、baseline vs 泛化對照、兩張混淆矩陣、誠實聲明）+ 導覽接線。
- 回寫 `DATASET_EVALUATION.md`（★下一步→進行中/完成）、`README.md`（四軌、頁數、§19）、`CLAUDE.md` 紅線、關於頁。

---

## 4. 核心結果（實驗設計）

`src/models/train_paderborn.py` 在同一特徵表上做兩種評估（仿 B+ LOBO vs LOCO）：

1. **baseline**：健康 + 人工故障 上分層 CV（`StratifiedKFold`），逐模型比 macro-F1、選最佳；輸出 accuracy / macro-F1 / 混淆矩陣。
2. **artificial→real**：以最佳模型訓練健康+人工、測真實損傷；輸出 accuracy / macro-F1 / 混淆矩陣。
3. **落差 = baseline − 泛化**：量化人工→真實 domain shift。

**真實資料結果（2026-06-24，N15_M07_F10、22 碼、440 量測）**：

| 評估 | accuracy | macro-F1 |
| --- | --- | --- |
| baseline（健康+人工 · 5 折 CV，最佳 RandomForest） | 1.00 | **1.00** |
| 人工→真實泛化（測真實損傷） | 0.24 | **0.20** |
| **落差** | | **0.80** |

- **結論**：模型在人工故障上**完美**分類，但對**真實加速壽命損傷幾乎泛化失敗**（near-chance）；真實混淆矩陣顯示
  約 **90/220** 真實受損量測被誤判為「健康」。與 Paderborn 文獻一致：人工(EDM/雕刻)故障訊號特徵不同於真實疲勞損傷。
- **誠實細節**：真實測試集**全為受損軸承（無健康類）**，三類 macro-F1 含一個 0 分的 healthy（無樣本）會機械性拉低；
  即使只看 outer/inner 兩類，鑑別力亦僅 ~0.3。落差為真，magnitude 受此影響需如實說明。

對照 JSON（`outputs/metrics/paderborn_eval.json`）結構：`results.{baseline, artificial_to_real}` + `summary.{best_model, baseline_macro_f1, generalization_macro_f1, gap}`。

---

## 5. config / 測試 / 邊界

- **config**：`config.yaml` `paderborn:`（raw_dir、conditions、bearings 分組、channels、enabled_models、cv_folds、輸出路徑）。
- **測試**：`tests/test_paderborn_features.py` 全用合成資料，不需下載；不碰真實 `.mat`。
- **邊界**：完全不改 A/B/B+ 主線與既有產物，皆為疊加；raw 不進 git。

---

## 6. 分階段交付（每步附驗收）

| 步 | 工作 | 驗收 | 狀態 |
| --- | --- | --- | --- |
| 0 | plan 文件 + config + .gitignore + data/README 區段 | 文件成段、config 可載入、附下載說明 | ✅ |
| 1 | `load_paderborn.py` + `build_paderborn_dataset.py` | `.mat` 解析出通道；parquet 含標籤與 `vib_*`/`cur*_*` | ✅（真實資料已跑：440 量測 × 35 欄） |
| 2 | `tests/test_paderborn_features.py` | 特徵/標籤/切分正確；`pytest` 綠 | ✅（8 項全綠） |
| 3 | `train_paderborn.py`：baseline CV + 人工→真實 | 對照 JSON（兩組 acc/macro-F1/混淆矩陣）+ 模型 + 預測 | ✅（真實：baseline 1.00 → 真實 0.20、落差 0.80） |
| 4 | Dashboard 模組 C 頁 + 導覽 | 頁面顯示 KPI/對照/混淆矩陣/誠實聲明 | ✅ |
| 5 | 回寫 DATASET_EVALUATION / README / CLAUDE 紅線 / 關於頁 | 各文件補日期戳、數字一致 | ✅（四軌、頁數 12） |

> **真實資料驗證（已完成）**：`build_paderborn_dataset` → `train_paderborn` 跑通；baseline macro-F1=1.00、
> 人工→真實 macro-F1=0.20、混淆矩陣顯示真實損傷大量被誤判為健康 —— 落差如實呈現。

---

## 7. 報告故事線與誠實聲明

> 模組 C：補上電流模態（MCSA）→ 用人工故障訓練、測真實損傷 → **實測結果：落差 0.80**
> （baseline 1.00 → 真實 0.20），如實指出「**人工故障不足以代表真實損傷分布**」——人工 EDM/雕刻
> 故障的訊號特徵與真實疲勞損傷不同，需領域自適應或納入真實樣本才能泛化。與 B+ 跨工況 RUL 受限同調的誠實結論。

**誠實聲明（務必寫進報告與頁面）**：
- 電流為**真實 PMSM 試驗台**馬達電流，MCSA 模態主張成立；但屬**軸承試驗台、非產線伺服馬達**，需脈絡化。
- 資料含**人工 + 真實**兩種損傷；頭條實驗明確「訓練人工、測真實」並**如實呈現泛化落差**，不過度宣稱。
- 屬**故障分類**非 RUL，不宣稱 RUL。
- MVP 為**子集**（列出所用碼/工況），結果限於該子集。

---

> **延伸規劃**：後續四條延伸軌（CE1 領域自適應救人工→真實 / CE2 MCSA 頻譜邊帶 /
> CE3 全 4 工況跨工況泛化 / CE4 即時預測端點）見
> [`MODULE_C_PADERBORN_EXTENSIONS_PLAN.md`](MODULE_C_PADERBORN_EXTENSIONS_PLAN.md)。

> **交叉連結**：[`DATASET_EVALUATION.md`](DATASET_EVALUATION.md)（資料取捨與遷移路線）、
> [`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)（結果回寫處）、
> [`MODULE_B_PLUS_XJTU_PLAN.md`](MODULE_B_PLUS_XJTU_PLAN.md)（同構的 domain-shift 主題）。
