# 模組 Servo — 伺服馬達健康狀態估測（專案主線）

> **狀態（2026-06-27）**：主線重構完成並通過 PR 自我審查修正。以 PHM Servomotor-Driven
> Ballscrew 退化資料為主線；Reference Model（健康分類 + DV 回歸）、AI 訓練模擬器、馬達欄位
> 解釋、LLM 維護助理、維修知識庫（TF-IDF RAG）、深度學習離線 baseline 皆已可運作。
> LLM 維護助理改為**多供應商**（Groq / OpenRouter / Gemini / Anthropic 依序嘗試，
> 全失敗才用離線範本）；側邊欄補充模組（A/B/B+/C）改為**可收合**（預設收合）。
> 結構化輸出新增 `consistency_warning`（分類器狀態與 DV 風險矛盾時提醒）；維修問答與維護報告
> 改用獨立 prompt（問答不再吐整份報告）。
> **已導入真實 PHM 資料並重訓（2026-06-27，`placeholder=false`）**：原始 FMCRD 8 檔 106 GB 以
> 串流聚合（`build_servo_from_zip.py`，不解壓不爆記憶體、線上統計與 `aggregate_run` 比對誤差 8.8e-13）
> 產出 **1,465 段特徵**（train 665 / 留出 test 800）；DV 由物理單位（max≈5012）正規化 0..1、依真實分布
> 重校 `dv_risk`（0.20 / 0.48）。**留出測試結果**：分類 logistic_regression macro-F1 **0.757**、
> DV 回歸 RandomForest **R²=0.937 / MAE=0.047**；**PyTorch** DL 離線 baseline MLP macro-F1 0.714、
> 神經 autoencoder 重建誤差隨退化單調上升（LN 0.33→HI 2.15）。**資料特性如實揭露**：`train_noisy_LO` 原始檔僅含 65 段
> （非 200，下載偏少），故 train LO 類別偏少；測試集各類 200 段完整。
> **已補真實資料載入路徑防護**：欄位 schema 驗證、`ylabel` 數值碼對應
> （`servo.ylabel_map`）、多檔 `run_index` 不再互相合併、DV 超出 0..1 警告（見 §3、§10）。
> **另補小資料/類別不均的穩健性**：`train_servo` 單樣本類別清楚報錯、`servo_dl` 在無 LN 段或
> 單樣本時不再崩、訓練模擬器與儀表板頁的非預期例外改為優雅降級（不噴 traceback）、模擬器改用
> committed demo CSV（對齊 §8）、機隊 API 帶 `placeholder` 旗標。
> Model A / B / B+ / C 保留為對照與歷史補充模組。

本文件相對連結：[`README.md`](../README.md)、[`data/README.md`](../data/README.md)、
[`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)、[`MODULE_C_PADERBORN_PLAN.md`](MODULE_C_PADERBORN_PLAN.md)、
[`WEB_REVAMP_PLAN.md`](WEB_REVAMP_PLAN.md)（前後端分離改版規劃）。

## 1. 定位與誠實性紅線

- 主線資料：**PHM Society Servomotor-Driven Ballscrew Mechanism Degradation Dataset**。
- 這是**模擬資料**，**不是**真實工廠伺服馬達 log；比軸承資料更接近伺服系統，但仍須如實揭露。
- 任務為 **health state estimation（分類）+ degradation value 回歸**。
- `run_index` 為運轉段索引，**不**等於剩餘壽命（RUL）；本模組**不宣稱 RUL**。
- 目前產物已以**完整真實 PHM FMCRD 資料集**訓練（`config.yaml::servo.placeholder: false`，2026-06-27 導入）；
  FMCRD 為高擬真**模擬**資料集，非真實工廠 log（見上方狀態戳與 §3）。

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

真實資料載入路徑防護（`build_feature_table` / `load_raw_servo`）：

- **Schema 驗證**（`validate_raw_columns`）：缺必要欄位或某欄整欄空值 → 清楚報錯，
  不再在聚合迴圈深處丟難懂的 `KeyError`，也不會把缺失訊號靜默當成 0.0。
- **`ylabel` 對應**：真實標籤若為數值碼，於 `config.yaml::servo.ylabel_map` 設
  `{0: LN, 1: LO, 2: MED, 3: HI}`；未對應到 LN/LO/MED/HI 會直接報錯（不猜測編碼）。
  某段 `ylabel` 全空也會明確報錯而非 `IndexError`。
- **多檔分段**：`load_raw_servo` 為每檔加 `__source_file__`，聚合改以
  `(檔名, run_index)` 分組——多個實驗檔 `run_index` 各自從 0 起算時不會被合併成同一列；
  輸出 `run_index` 重編為全域唯一段索引。
- **DV 範圍檢查**：真實 DV 超出 0..1 時 `build_servo_dataset` 會警告（風險帶以 0..1 校準）。

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

`src/models/servo_dl.py`（**PyTorch**）：MLP 分類 + DV 回歸 + 以健康資料擬合的**神經 autoencoder**
重建誤差（隨退化等級上升，取代先前的 PCA 替身）。寫入 `outputs/metrics/servo_dl_results.json`，
Dashboard 唯讀顯示（訓練模擬器頁的「深度學習離線結果」展開區）。torch 只在 `requirements-dl.txt`、
**離線訓練**；雲端 / Docker 映像裝 `requirements-dev.txt`（**不含 torch**），runtime 只讀 JSON、不跑 DL。**真正的 1D-CNN 需在
原始 PHM 時序波形上開窗（非此處逐段聚合特徵）；原始 FMCRD 已備齊，開窗管線 + 1D-CNN 列為 Phase B。**

## 8. 部署策略

- 原始大資料不進 git（`data/raw/servo/` 已忽略）。
- 伺服器只放：Reference Model、scaler/encoder/feature_config、demo feature dataset、樣本筆、
  metrics、知識庫小型資料（皆已在 `.gitignore` 白名單，隨 repo 提交）。
- 訓練模擬器只用 `servo_feature_demo.csv`。LLM 用 API 但有 fallback。爬蟲離線可跳過。

## 9. 重建步驟

```bash
# 真實 PHM（FMCRD zip，106 GB）—— 串流聚合、不解壓、每檔 checkpoint 可續跑
python -m src.data.build_servo_from_zip --zip <FMCRD_Data.zip 路徑>
#   （無原始資料時改用：python -m src.data.build_servo_dataset 產生 placeholder）
python -m src.models.train_servo         # 訓練 Reference Model（分類 + 回歸；有 split 則留出評估）
python -m src.models.servo_dl            # （選用）離線 DL baseline
streamlit run app/streamlit_app.py       # 首頁主線 = 模組 Servo
```

> Windows 主控台若為 cp950，跑訓練請加 `PYTHONUTF8=1`（log 內含 `R²` 等字元）。

## 10. 真實資料導入（已完成，2026-06-27）

1. ✅ 原始 FMCRD `*.csv`（8 檔、欄位與 `RAW_COLUMNS` 完全吻合、`ylabel` 已是 LN/LO/MED/HI）以
   `build_servo_from_zip.py` **串流聚合**（不解壓、線上統計、每檔 checkpoint）。途中發現
   `train_noisy_LO` 的 `i_3p_c` 有少量非數值雜訊 → `pd.to_numeric(errors="coerce")` 容錯。
2. ✅ DV 為物理單位（max≈5012）→ 正規化 0..1；依真實分布重校 `servo.dv_risk` = 0.20 / 0.48。
3. ✅ `placeholder=false`；`train_servo` / `servo_dl` 走 **split-aware**（train_* 訓練、test_* 留出）。
4. ⬜（後續）視資料調整聚合粒度（`run_index` / `transitions`）、或對 train LO 段數偏少補資料。
5. ✅（Phase A）PyTorch DL 離線 baseline：MLP 分類/回歸 + 神經 autoencoder（取代 PCA 替身）；torch 僅 `requirements-dl.txt`、離線訓練（雲端映像裝 dev、不含 torch）。
6. ⬜（Phase B）原始時序 1D-CNN：開窗 builder（從原始 FMCRD zip 串流抽時間窗）+ 1D-CNN 分類 / conv-AE。原始 FMCRD 已備齊（`Downloads/FMCRD_Data.zip`，CRC 對齊溯源指紋）。
