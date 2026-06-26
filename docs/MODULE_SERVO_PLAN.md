# 模組 Servo — 伺服馬達健康狀態估測（專案主線）

> **狀態（2026-06-26）**：主線重構完成並通過 PR 自我審查修正。以 PHM Servomotor-Driven
> Ballscrew 退化資料為主線；Reference Model（健康分類 + DV 回歸）、AI 訓練模擬器、馬達欄位
> 解釋、LLM 維護助理、維修知識庫（TF-IDF RAG）、深度學習離線 baseline 皆已可運作。
> LLM 維護助理改為**多供應商**（Groq / OpenRouter / Gemini / Anthropic 依序嘗試，
> 全失敗才用離線範本）；側邊欄補充模組（A/B/B+/C）改為**可收合**（預設收合）。
> 結構化輸出新增 `consistency_warning`（分類器狀態與 DV 風險矛盾時提醒）；維修問答與維護報告
> 改用獨立 prompt（問答不再吐整份報告）。**目前以 placeholder 合成資料訓練**，待下載真實
> PHM 資料後重訓。Model A / B / B+ / C 保留為對照與歷史補充模組。

本文件相對連結：[`README.md`](../README.md)、[`data/README.md`](../data/README.md)、
[`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)、[`MODULE_C_PADERBORN_PLAN.md`](MODULE_C_PADERBORN_PLAN.md)。

## 1. 定位與誠實性紅線

- 主線資料：**PHM Society Servomotor-Driven Ballscrew Mechanism Degradation Dataset**。
- 這是**模擬資料**，**不是**真實工廠伺服馬達 log；比軸承資料更接近伺服系統，但仍須如實揭露。
- 任務為 **health state estimation（分類）+ degradation value 回歸**。
- `run_index` 為運轉段索引，**不**等於剩餘壽命（RUL）；本模組**不宣稱 RUL**。
- 目前產物以 **placeholder 合成資料**訓練（`config.yaml::servo.placeholder: true`），僅供流程展示。

## 2. 任務定義

| 任務 | 目標欄位 | 類別 / 範圍 | 中文顯示 |
| --- | --- | --- | --- |
| 健康狀態分類 | `ylabel` | LN / LO / MED / HI | 健康 / 輕度退化 / 中度退化 / 高度退化 |
| 退化程度回歸 | `DV`（0..1） | 連續值 | degradation_score → health_score / risk_level |

DV 風險帶（`config.yaml::servo.dv_risk`，placeholder 校準，需以真實分布重校）：
`<0.33` Low、`0.33–0.66` Medium、`≥0.66` High。

## 3. 資料管線與特徵

原始 PHM 為逐時序 CSV（欄位見 `src/features/servo_features.py::RAW_COLUMNS`）。
**伺服器不吃原始大 CSV**：以 `run_index` 為單位把每段聚合成一列特徵
（`build_feature_table`），輸出 `data/processed/servo_features.parquet`。

特徵組（`FEATURE_SETS`，使用者可在訓練模擬器選擇）：

| 名稱 | 內容 |
| --- | --- |
| `basic_motion` | 轉速 / 扭矩 / 位移增量的 mean/std/rms |
| `current` | 三相電流、D/Q 軸電流的 rms/std |
| `position_tracking` | 目標 / 實際位置、位置誤差（含 max/std） |
| `full` | 上述三組聯集 |
| `engineered` | 退化最敏感精選：current_rms、torque_std、rotor_speed_std、position_error_mean/max、quadrature_rms、direct_rms |

## 4. 正式模型 vs 訓練模擬器

- **Reference Model（離線、完整資料）**：`src/models/train_servo.py`。分類器以分層 CV
  macro-F1 自 `servo.enabled_models` 選最佳；DV 以 RandomForest 回歸。匯出
  `servo_clf.joblib`、`servo_reg.joblib`、`servo_feature_config.json`（含特徵欄位、
  label 映射、健康基線、DV 風險帶）、`servo_clf_eval.json`、`servo_reg_eval.json`。
- **Training Simulator（伺服器、小資料、教學）**：`src/models/servo_simulator.py` +
  `src/ui/servo_views.py::render_simulator`。可選資料量（100/500/1000/5000）、特徵組、
  演算法（LR/Ridge、決策樹、隨機森林、梯度提升、MLP，皆 sklearn），顯示訓練時間、
  Accuracy/F1/混淆矩陣（分類）或 MAE/RMSE/R²（回歸），並對照 Reference Model 與真實標籤、
  附文字解釋為何資料量 / 特徵 / 模型會影響結果。

## 5. 推論與結構化輸出

`src/models/servo_predict.py::predict_servo` 將一列聚合特徵轉成結構化輸出：
`predicted_health_state`、`health_state_zh`、`health_state_proba`、`model_confidence`、
`degradation_score`（DV）、`health_score`、`risk_level`、`consistency_warning`（分類器健康狀態與 DV
風險差 ≥2 級時的矛盾提醒，否則為 `null`）、`top_features`（vs 健康基線的 z 偏離 + 白話提示）、
`maintenance_advice`、`placeholder`。FastAPI：`GET /servo/model_info`、`POST /servo/predict`。

## 6. 應用層

- **馬達欄位解釋**：`src/servo/field_glossary.py`（欄位中文名 / 說明 / 對伺服意義 / 異常徵兆）+
  特徵組說明，頁面見 `render_glossary`。
- **LLM 維護助理**：`src/llm/maintenance_assistant.py`。接收結構化輸出 + 檢索片段，生成
  「結果說明 / 可能原因 / 建議檢查 / 維修優先級 / 工單草稿 / 報告摘要」。保守措辭（可能 / 建議檢查 /
  需由現場人員確認）。**多供應商**：依 `config.yaml::llm.providers` 順序嘗試
  **Groq / OpenRouter / Gemini**（皆 OpenAI 相容、有免費額度，用標準庫呼叫、**不增加 runtime 依賴**）
  與 **Anthropic**（SDK）；**全部不可用時退回離線 fallback 範本**。對應金鑰：`GROQ_API_KEY` /
  `OPENROUTER_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`。
- **維修知識庫 / RAG**：`src/knowledge/`。離線優先——`data/knowledge/*.md` 種子文件 + TF-IDF
  字元 n-gram 檢索（`retriever.py`，sklearn，無新依賴）；`crawler.py` 為選用白名單爬蟲
  （`sources.yaml`，預設 `enabled: false`，尊重 robots.txt）。

## 7. 深度學習（第二部分，離線唯讀）

`src/models/servo_dl.py`：sklearn MLP baseline（分類 + DV 回歸）+ 以健康資料擬合的 PCA
重建誤差（隨退化等級上升）。寫入 `outputs/metrics/servo_dl_results.json`，Dashboard 唯讀
顯示（訓練模擬器頁的「深度學習離線結果」展開區）。**真正的 1D-CNN / Autoencoder 需離線
torch 與真實時序資料，列為後續工作；雲端 runtime 不跑 DL。**

## 8. 部署策略

- 原始大資料不進 git（`data/raw/servo/` 已忽略）。
- 伺服器只放：Reference Model、scaler/encoder/feature_config、demo feature dataset、樣本筆、
  metrics、知識庫小型資料（皆已在 `.gitignore` 白名單，隨 repo 提交）。
- 訓練模擬器只用 `servo_feature_demo.csv`。LLM 用 API 但有 fallback。爬蟲離線可跳過。

## 9. 重建步驟

```bash
python -m src.data.build_servo_dataset   # 無原始資料時自動產生 placeholder 特徵表
python -m src.models.train_servo         # 訓練 Reference Model（分類 + 回歸）
python -m src.models.servo_dl            # （選用）離線 DL baseline
streamlit run app/streamlit_app.py       # 首頁主線 = 模組 Servo
```

## 10. 待真實資料下載後

1. 將原始 CSV 放入 `data/raw/servo/`，重跑 `build_servo_dataset`（改走真實聚合路徑）。
2. 重跑 `train_servo`、`servo_dl`；把 `config.yaml::servo.placeholder` 設為 `false`。
3. 依真實 DV 分布重校 `servo.dv_risk` 風險帶。
4. 視資料調整聚合粒度（`run_index` / `transitions`）。
5. 取得真實時序後再做 1D-CNN / Autoencoder（離線 torch）。
