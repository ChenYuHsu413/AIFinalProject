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
> 重校 `dv_risk`（0.20 / 0.48）。**留出測試結果**：分類 logistic_regression macro-F1 **0.757**
> （engineered；**2026-07-10 特徵組轉 full → 0.819**，DV R²=0.937→**0.944**，見 §11）、
> DV 回歸 RandomForest **R²=0.937 / MAE=0.047**；**PyTorch** DL：Phase A（MLP macro-F1 0.714
> →**0.782**（full，見 §7）+ 神經 autoencoder 留出 LN 0.36→HI 2.20）＋ **Phase B 真 1D-CNN**（原始波形能量包絡，80 runs/檔、
> 留出 macro-F1 **0.692**（n=320，seed 敏感 ±0.03）、conv-AE 留出 LN 0.26→HI 0.36，見 §7）。**資料特性如實揭露**：`train_noisy_LO` 原始檔僅含 65 段
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
**離線訓練**；雲端 / Docker 映像裝 `requirements-dev.txt`（**不含 torch**），runtime 只讀 JSON、不跑 DL。

> **狀態（2026-07-10）— DL 特徵組與參考模型解耦**：參考模型轉 full（§11）後，DL 不再直接跟隨
> `reference_feature_set`，改用兩個獨立 config 欄位——`dl_classifier_feature_set: full`（MLP 分類
> macro-F1 **0.714→0.782**、DV 回歸 R²=0.981，收下 full 的增益）與 `dl_ae_feature_set: engineered`
> （autoencoder 維持 engineered）。**理由**：AE 重建誤差健康→退化的**單調性**（LN 0.36 ≤ LO 0.38 ≤
> MED 0.65 ≤ HI 2.20）是它作為健康指標的核心承諾；在 full 空間 AE 於 **LN/LO 反轉**（LN 0.897 >
> LO 0.868），與 FLAML 留出崩盤（§11）同為 train/test **LO domain shift** 的旁證，故 AE 不為跟隨
> full 而放寬單調測試。1D-CNN（Phase B）吃原始波形、不讀特徵組，不受影響。

**Phase B（已完成）— 真 1D-CNN on 原始波形**：`src/data/build_servo_windows.py` 從原始 FMCRD zip
**串流**每段 run、降為原始波形**能量包絡**（逐塊 std，8 通道 × 256 時間塊；MVP 子集、split 依檔分離無洩漏）→
`src/models/servo_cnn.py`（PyTorch）訓練 **1D-CNN 分類 + 1D conv-autoencoder**，寫
`outputs/metrics/servo_cnn_results.json`，後端 `GET /servo/cnn_results`、報表頁「1D-CNN（原始波形）」唯讀顯示。
留出測試（80 runs/檔、n=320）**Accuracy 0.709 / macro-F1 0.692**；conv-AE 留出重建誤差 LN 0.26→HI 0.36 單調。
windowed `.npz` 為 gitignored 暫存（雲端只讀 JSON）。**深化結論（誠實）**：資料由 40→80 runs（更大、更具代表性的
測試集）後，多 seed 掃描顯示加寬+dropout、每塊 std+mean 等均**未穩健勝出**（平均約 0.64–0.70、seed 敏感 ±0.03），
故保留最簡 narrow conv；先前 40-run 單 seed 的 0.729 為樂觀抽樣，已如實下修。**後續**：原始逐點 CNN、頻譜輸入。

## 8. 部署策略

- 原始大資料不進 git（`data/raw/servo/` 已忽略）。
- 伺服器只放：Reference Model、scaler/encoder/feature_config、demo feature dataset、樣本筆、
  metrics、知識庫小型資料（皆已在 `.gitignore` 白名單，隨 repo 提交）。
- 訓練模擬器只用 `servo_feature_demo.csv`。LLM 用 API 但有 fallback。爬蟲離線可跳過。
- **部署單一真相源（S3，2026-07-11）**：正式推論**一律經模型版本 registry 載入**（`servo_model_registry.load_active()`，
  見 §14）。`outputs/models/servo_*.joblib` **僅為 `train_servo` 的預設工作輸出**，不是部署來源；它與 active 版本
  可能因單獨重訓而分岔，但**不影響服務**（載入只讀 registry）。後端啟動會比對兩者 CRC32、不一致時**記警告**
  （非致命）。收斂 `outputs/models/`（改由 registry 全權管理）列為 future work。

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
6. ✅（Phase B 完成）原始波形 1D-CNN：`build_servo_windows.py`（串流→能量包絡）+ `servo_cnn.py`（1D-CNN 分類 + conv-AE）；80 runs/檔留出 macro-F1 0.692（n=320，seed 敏感）、後端 `/servo/cnn_results`、報表頁顯示。原始 FMCRD 用 `Downloads/FMCRD_Data.zip`（CRC 對齊溯源指紋）。

## 11. 類別不平衡實驗（Phase 0 → E1 → E2/E3 → E4 → E5 → 轉正）

> **狀態（2026-07-10）**：完成「診斷 → 加權 → 過採樣 → 天花板確認 → 特徵/AutoML 突破 → 生成式擴充
> → Reference Model 轉正」一整段實驗。E1–E5 為**唯讀腳本**（`scripts/`），沿用 `train_servo.run()`
> 的載入與 split（train 665 / 留出 test 800），**不重新切分、不動測試集**。核心結論：engineered 7 維
> 空間內的 ~0.757 天花板**不是樣本量問題**（加權、SMOTE 家族、CTGAN 都補不動 LN↔LO 互相誤判）；
> **瓶頸是特徵可分性**——換 full(21 維) 後留出 macro-F1 0.757→**0.819**。full 空間裡 SMOTE 略降、
> CTGAN 有害、FLAML 即使加權仍只 0.69，**贏家是最樸素的 LR + balanced + full**。已據此**轉正**：
> `reference_feature_set` engineered→full，重訓後留出 macro-F1 **0.819**、DV R²=0.944，下游全綠
> （engineered 0.757 留為歷史對照）。DL baseline 特徵組隨之解耦（MLP→full、AE→engineered，見 §7）。

**問題**：train 的 LO（輕度退化）僅 **65 段**，其餘三類各 ~200（不平衡比 3.08）；test 四類均衡各
200。留出混淆矩陣顯示 LN（健康）與 LO 大量互相誤判。

| 階段 | 腳本 | 產物 |
| --- | --- | --- |
| Phase 0 診斷 | [`scripts/diagnose_class_imbalance.py`](../scripts/diagnose_class_imbalance.py) | `outputs/figures/servo_train_class_dist.png`、`servo_confusion_heatmap.png` |
| E1 加權 | [`scripts/run_e1_class_weight.py`](../scripts/run_e1_class_weight.py) | [`outputs/metrics/e1_class_weight.json`](../outputs/metrics/e1_class_weight.json) |
| E2/E3 過採樣 | [`scripts/run_e2_e3_resampling.py`](../scripts/run_e2_e3_resampling.py) | [`outputs/metrics/e2_e3_resampling.json`](../outputs/metrics/e2_e3_resampling.json)、`outputs/figures/e2_e3_test_macro_f1.png` |
| E4 特徵/AutoML | [`scripts/run_e4_automl_and_features.py`](../scripts/run_e4_automl_and_features.py) | [`outputs/metrics/e4_automl_features.json`](../outputs/metrics/e4_automl_features.json) |

**Phase 0（診斷）**：現況 Reference Model（LR + `class_weight='balanced'`）留出 per-class F1 =
LN 0.632 / LO 0.531 / MED 0.905 / HI 0.959；LO recall 僅 0.500（200 段 LO 有 86 段被誤判成 LN），
LN precision 0.606。退化明顯的 MED/HI 幾乎全對，問題集中在 LN↔LO 邊界。

**E1（class_weight 加權）**：只切換 `class_weight`，其餘超參沿用工廠（LR `max_iter=2000`；
RF `n_estimators=200`），`random_state=42`。

| 模型 | 版本 | 留出 macro-F1 | LO 的 F1 |
| --- | --- | --- | --- |
| LogReg | default | 0.636 | **0.029** |
| LogReg | **balanced** | **0.757** | 0.531 |
| RandForest | default | 0.667 | 0.195 |
| RandForest | balanced | 0.687 | 0.312 |

→ 加權對 LR 帶來 **+0.121** macro-F1，增益幾乎全來自把 LO 從「形同放棄」（F1 0.029）救回
（+0.501）；LR 明顯優於 RF，後續實驗只用 LR。

**E2/E3（過採樣）**：`imbalanced-learn` 的 SMOTE / BorderlineSMOTE / ADASYN / SMOTETomek，各配
`class_weight` ∈ {None, 'balanced'}，共 10 組。**防洩漏**：整條 `imblearn.Pipeline`
（StandardScaler → 重採樣 → LR）交給 5-fold `StratifiedKFold`，重採樣只發生在每個 fold 的訓練分割。

- 兩個對照組精確重現（E0_default 0.636、E0_balanced 0.757），驗證管線正確。
- 四種過採樣全部落在留出 macro-F1 **0.755–0.759**，與單純加權**等效**、無顯著超越；LO recall 天花板
  約 0.535。重採樣「疊加」加權（+balanced）無額外好處，多屬持平或略降（符合過度矯正預期）。

**⚠ 自我修正註記（誠實揭露）——SMOTE 絕對數量在 CV fold 內的失真**：E2/E3 用
`sampling_strategy={'LO': 200}`（**絕對數量**）。在最終「完整 train」擬合時 LO 是 65→200
（合成含量 ≈3.1×）；但在 5-fold CV 內，每個 fold 的訓練分割只有約 52 段真實 LO，卻仍被補到**同一個
絕對值 200**（≈3.85×）——**CV fold 內的過採樣強度其實比最終部署的模型更激進**。因此
`e2_e3_resampling.json` 裡的 **CV macro-F1 並非與留出、或跨組完全等價的比較基準**（改用比例式
`sampling_strategy`（float / 'auto'）才會讓強度隨 fold 真實少數類數量等比縮放）。這不影響本實驗結論
——**留出 test 800 段才是判準**，且十組全部收斂在 ~0.757——但此失真必須如實記錄。

**E2/E3 天花板結論**：加權已逼近 **engineered 7 維空間**的可分上限，過採樣（在同一空間內插補）補不出
新的判別資訊。訓練集 LO 全部來自 `train_noisy_LO` 單一原始檔（下載偏少、僅 65 段），與 test LO 可能
存在**分布差異（domain shift）**；重採樣只在特徵空間內插值，無法跨越分布差異，僅緩解樣本量不足。

**E4（特徵組對照 + FLAML AutoML）—— 假設證實：瓶頸在特徵可分性，不在樣本量**

Part A（LR + `class_weight='balanced'`，其餘不變，只換特徵組）：

| 特徵組 | 維度 | 留出 macro-F1 | LO recall | LN precision | LO 的 F1 |
| --- | --- | --- | --- | --- | --- |
| engineered | 7 | 0.757 | 0.500 | 0.606 | 0.531 |
| **full** | 21 | **0.819** | 0.595 | 0.675 | 0.643 |

→ 僅把特徵組換成 full，留出 macro-F1 **+0.062（0.757→0.819）**，LN↔LO 誤判同步下降（混淆矩陣：
LN→LO 68→44、LO→LN 86→75）。**證實 E2/E3 的天花板是 engineered 特徵空間的可分性上限、而非樣本量**
——補進 `basic_motion` / `position_tracking` / 原始電流等真實維度，給了線性模型原本缺乏的 LN/LO 判別資訊。

Part B（FLAML AutoML，`task=classification, metric=macro_f1, time_budget=600, seed=42`，只餵 train、
內部 CV 自驗）：

| 組別 | 特徵組 | best | 內部驗證 macro-F1 | 留出 macro-F1 | LO recall |
| --- | --- | --- | --- | --- | --- |
| FLAML_engineered | 7 | rf | 0.728 | 0.666 | 0.125 |
| FLAML_full | 21 | rf | 0.727 | 0.662 | 0.100 |

→ FLAML 兩次都選 rf，**內部 CV 看似 0.727，留出卻僅 ~0.66**，比單純 LR+balanced 還差、更遠低於
LR_full 0.819；LO recall 崩到 0.10（未加權樹集成在 domain-shift 的 test LO 上幾乎全放棄）。**教訓**：
AutoML 以內部 CV 最佳化又未套 class_weight，挑到的模型在 in-distribution 好看、在偏移的少數類上失守；
**內部 CV 0.727 對留出 0.66 的落差，本身即 domain shift 的證據**。簡單、可解釋的 LR + balanced +
full 特徵完勝。

**修正「天花板」定性**：E2/E3 的 ~0.757 是**在 engineered 7 維空間內**的上限，非固定值；E4 證明換特徵
空間即突破至 0.819。domain shift 依然存在（LO recall 仍不及 MED/HI、FLAML 的 CV−留出落差為證），但
天花板由**特徵可分性 × 模型歸納偏誤**共同決定，而非樣本量。

**E5（full 空間擴充方法對決 + 特徵組選擇驗證）** — 腳本
[`scripts/run_e5_and_validation.py`](../scripts/run_e5_and_validation.py)、產物
[`outputs/metrics/e5_validation.json`](../outputs/metrics/e5_validation.json)、
[`outputs/figures/e5_ctgan_kde_full.png`](../outputs/figures/e5_ctgan_kde_full.png)。

*Task 1 — 特徵組選擇只依訓練集*：train 665 段 5-fold CV（LR+balanced），full **0.740 ± 0.028** >
engineered **0.730 ± 0.036**（+0.010）。**選 full 可只依訓練集 CV 決定，留出 0.819 僅為事後驗證**，
排除 test-set selection 疑慮。

*Task 2 — full 空間三組對決*（留出 test 800 段）：

| 組別 | 特徵組 | 留出 macro-F1 | LO recall | LN precision |
| --- | --- | --- | --- | --- |
| engineered（LR+bal，歷史） | engineered | 0.757 | 0.500 | 0.606 |
| **E0_full**（無擴充） | full | **0.819** | 0.595 | 0.675 |
| SMOTE_full（LO→200） | full | 0.811 | 0.585 | 0.678 |
| CTGAN_full（+135 合成 LO） | full | 0.686 | 0.100 | 0.554 |
| FLAML_weighted_full（sample_weight=balanced） | full | 0.690 | 0.335 | 0.544 |

→ **贏家是最樸素的 LR + balanced + full = 0.819**：full 空間裡 SMOTE **無助反略降**（0.811），
CTGAN **明顯有害**（0.686）。**CTGAN TSTR：純合成 LO 訓練、測真實 LO 的 F1 = 0.010**——合成樣本
幾乎無法辨識真實 LO；KDE 品質圖顯示 65 筆小樣本在 21 維下多個特徵（torque_rms / direct_rms /
quadrature_rms 等）分布錯位。**E4c（給 FLAML 公平一戰）**：即使傳入 `sample_weight='balanced'`，
FLAML 仍選 lgbm、留出僅 0.690——**推翻「FLAML 輸在沒加權」的質疑**：問題不在加權，而是樹集成在
domain-shift 的 test LO 上過擬合，樸素正則化線性模型泛化更好。

*誠實揭露*：所有擴充僅作用訓練資料，test 800 段全為真實樣本；CTGAN 僅以 65 筆 LO 訓練屬小樣本、
21 維生成品質有限（見 KDE 與 TSTR），仍侷限既有特徵分布、無法跨越 train/test 的 LO domain shift。

**Reference Model 轉正（2026-07-10，engineered → full）**：Task 1 CV 確認 full 較優後，
`config.yaml::servo.reference_feature_set` 改為 `full`，`python -m src.models.train_servo` 正式重訓——
CV 從 `enabled_models` 選出 **logistic_regression**（CV 0.740，勝 mlp 0.732 / rf 0.697），留出
**macro-F1 0.819**、DV 回歸 RandomForest **R²=0.944 / MAE=0.044**（皆優於 engineered 的 0.757 /
0.937）。**engineered 0.757 保留為歷史對照**。下游全數驗證：`servo_feature_config.json`（21 欄 +
健康基線）、`servo_predict` 的 `top_features`、訓練模擬器、demo CSV、`/servo/predict` 與
`/servo/model_info` 均隨新特徵組正常運作；servo 測試 38 passed。DL baseline 的特徵組解耦見 §7。

## 12. 即時串流 demo（S1：FMCRD replay → SSE → 視窗聚合 → 參考模型）

> **狀態（2026-07-11）**：S1 + S1b 完成。沿用即時監控的 SSE 骨架（`data: {json}` 串流、`StreamingResponse`
> `text/event-stream`；範本見 [`src/monitor/live_stream.py`](../src/monitor/live_stream.py)），把資料源換成
> **真實 FMCRD 測試資料**、模型接**參考模型 `predict_servo`**。端到端跑通：健康狀態隨 replay 段落
> **LN → LO → HI** 演進（degradation_score 0.05→0.26→0.75、health 95→25、風險 Low→High）。**S1b 驗證抽稀無害、
> W/S 鎖定 run 循環對齊（見 §12.1）；完整儀表板留 S2。**

**Task A — replay 素材抽取**（[`scripts/extract_replay_segments.py`](../scripts/extract_replay_segments.py)）：
從 FMCRD zip 的**測試分割**抽 LN / LO / HI 三段到 `data/demo/replay/`（`config.yaml::servo_replay`；加入 git 白名單、
各約 3 MB）。**關鍵粒度**：參考模型以**整段 run（`build_feature_table` 依 `run_index` 聚合）**訓練，每個 run 是一個
**6 s / 5 階梯定位循環**；只取 run 頭段（單一階梯）會落在訓練分布外、預測失真。故每段取**數個完整 run 並均勻抽稀**
（保留欄位與時間順序，只放寬取樣間隔），使「一個 run 循環」的視窗重現 per-run 聚合。FMCRD 測試檔為**單一類別/檔**，
退化全程是**跨檔**拼接（各取數 run），manifest 記錄各段來源檔 / `run_indexes` / `ylabel` / 列數（溯源）。無 zip 時腳本清楚報錯。

**Task B — 發布端**（[`scripts/servo_replay_publisher.py`](../scripts/servo_replay_publisher.py) +
核心 [`src/monitor/servo_replay_stream.py`](../src/monitor/servo_replay_stream.py)）：逐列讀 replay CSV，以可設定頻率
（`servo_replay.emit_hz`，預設加速重播）發布 `GET /servo/stream` SSE，**訊息 schema = `RAW_COLUMNS`**；支援
**多段串播**（LN→LO→HI）。另保留 `--mode fake` 合成模式——**僅供管線連通性測試，資料不在訓練分布內、預測無效**
（程式與 README 均標明；接收端標記 `⚠ 假數據，預測無效`）。**demo 一律用 replay。**

**Task C — 視窗聚合接收端**（[`scripts/servo_replay_consumer.py`](../scripts/servo_replay_consumer.py)）：SSE 接收 →
滑動視窗（W 秒 / S 秒，`servo_replay.window` 可調；以 row `time` 建**單調串流時鐘**、跨 run 的 time 重置也不亂）→
**直接複用 `servo_features.aggregate_run`**（與訓練同一套統計，避免特徵定義漂移）算 21 維 full 特徵 → `predict_servo` →
逐視窗打印 `predicted_health_state` + `degradation_score`（另附視窗真實 `ylabel` 對照）。

**誠實揭露**：LN↔LO 邊界仍見互相誤判（如某 LN 窗判 LO），為 §11 記錄之模型**既有**弱點（0.819 天花板），非管線 bug；
**DV / 風險**呈乾淨單調，是最穩健的退化指標。全程**唯讀既有模型、未改訓練程式**。

啟動與參數見 [README §5.1](../README.md)；串流監控為 [README §9 「MLOps 閉環（已完成）」](../README.md) 的起點。

### 12.1 S1b — replay 視窗驗證（唯讀）

> **狀態（2026-07-11）**：完成。腳本 [`scripts/validate_replay_windows.py`](../scripts/validate_replay_windows.py)（唯讀），
> 產物 [`outputs/metrics/replay_window_validation.json`](../outputs/metrics/replay_window_validation.json) +
> 偏差圖 `outputs/figures/replay_window_decimation_bias.png`。**結論：抽稀對模型無害，補救不需要；W/S 鎖定 run 循環對齊。**

S1 已定視窗粒度 = 一個完整 run 循環（OOD 教訓）。本實驗驗兩件事：

**(1) 抽稀偏差量化**——對 replay 各 run 的「抽稀視窗特徵」vs「原始完整 run 聚合特徵」逐特徵比相對偏差，分兩組：

| 組別 | 範圍 | median | p90 | max |
| --- | --- | --- | --- | --- |
| 分布型（mean/std/rms） | 模型 full 21 特徵 | 0.83% | 1.9% | 70.2%\* |
| 極值型（`*_max`/`*_min`） | 模型 full 21 特徵 | 0.0% | 0.0% | **0.001%** |

\* 70% 出現在 **`torque_mean`≈0**（整段淨扭矩對稱抵銷）等**近零分母**特徵：絕對差僅 ~0.003 N·m、物理可忽略、非退化判別欄位。
真正被擔心的**極值型 `*_max` 偏差在模型欄位（僅 `position_error_max`）幾乎為零（0.001%）**。全 56 聚合特徵中 `*_max`/`*_min`
確有抽稀縮水（最大 ~62%，如 `direct_max`），但**這些欄位不在模型 full 特徵組內**，不影響預測。
→ **補救（peak-preserving decimation / 提高保留率）不需要**；僅留待未來若特徵組納入 `*_max/*_min` 再啟用。

**(2) 端到端一致性**——「串流管線預測（抽稀 run → `predict_servo`）」vs「離線整段聚合預測」，逐 run 比預測狀態：

- **replay 素材 9 runs：一致率 100%**（LN/LO/HI 各 3/3，目標 ≥95% 達標）。
- **佐證（更大樣本 24 runs，8/類）：91.7%**；2 筆不一致**全落在 LN↔LO 邊界、皆近門檻**（§11 記錄之模型既有弱點，
  如 test LO run 被判 LN↔LO 互翻），**疑似管線偏差 0 筆**。抽稀本身不製造 MED/HI 誤判。

**決策**：`config.yaml::servo_replay.window` 的 **W=6s（一個 run 循環）、S=3s（50% 重疊）鎖定**、註解記錄依據與本驗證連結。
S2 儀表板可直接在此粒度上疊加，不需重訓或改抽稀策略。

## 13. 告警引擎 + 即時監控頁（S2）

> **狀態（2026-07-11）**：完成。在 S1/S1b 的串流管線上加**告警遲滯引擎**
> （[`src/monitor/alert_engine.py`](../src/monitor/alert_engine.py)）與 **Streamlit 即時監控頁**
> （[`src/ui/servo_views.py`](../src/ui/servo_views.py)::`render_live_monitor`）。收/窗/推論邏輯抽成共用模組
> [`src/monitor/servo_replay_client.py`](../src/monitor/servo_replay_client.py)（S1 CLI consumer、告警引擎、監控頁三者共用，單一真相源）。
> 端到端驗收（headless AppTest + CLI）：播 LN→LO→HI，狀態燈依序變色、告警於 HI 段**觸發一次不重複**、工單自動生成；
> 強制 LLM 全供應商失敗仍完整走 fallback、且**告警不被阻塞**。

**Task A — 告警引擎**：消費逐窗結構化預測，兩條獨立事件流：

- **主告警（遲滯）**：`risk_level` 連續 **N=3** 窗達 High 才**觸發**、連續 **M=3** 窗回落 ≤Medium 才**解除**
  （`config.yaml::servo_alert`，註解引用 S1b 的 LN/LO 閃爍證據）。**依據不是教科書**——S1b 實測 LN↔LO 預測會在門檻
  附近閃爍（24 runs 2 筆近門檻互判），無遲滯會抖動誤報。
- **矛盾提示（獨立）**：`consistency_warning` 非 null（分類器狀態與 DV 風險差 ≥2 級）時發獨立事件,**不**觸發主告警。
- **觸發時**：事件**立即**寫 `outputs/alerts/<date>.jsonl`（時間、視窗特徵/模型輸出快照、`model_version`——先填
  `"v1"`，S3 接手後由模型 metadata 自動帶入），**再於背景執行緒**呼叫既有多供應商 LLM 助理生成工單草稿並補寫
  連結事件——**LLM 任何延遲/失敗都不阻塞或延後告警本身**（事件先寫、工單走 `try/except` + 非同步；全供應商失敗
  自動走離線 fallback 範本）。

**Task B — Streamlit 即時監控頁**（消費**同一條** SSE `/servo/stream`，不另起通道）：

- **頂部健康狀態燈**（綠/黃/橙/紅 = LN/LO/MED/HI），以**近 3 窗多數決**平滑避免 LN/LO 邊界閃爍；
  **但同屏保留未平滑的逐窗原始預測**（副欄小字列出近 K 窗原始狀態）。**誠實性**：平滑是**顯示層決策、非竄改模型輸出**——
  模型真實的逐窗不確定性同屏可見，答辯時可指著螢幕說明 S1b 的閃爍發現與此顯示決策。
- **中部**：`degradation_score` 與 `model_confidence` 滾動時序圖。
- **底部事件流**：主告警／告警解除／矛盾提示分色（最新在上），主告警可展開見**工單草稿**（背景生成、稍後補上）。
- 頁面標示**數據源**（讀 manifest 的 replay 段落名）與**模擬性質**，需先啟動發布端。

啟動見 [README §5.2](../README.md)（告警為 [README §9 「MLOps 閉環（已完成）」](../README.md) 的一環）。W/S 沿用 §12.1 鎖定值；N/M/平滑窗數見 `config.yaml::servo_alert`。全程唯讀既有模型。

## 14. 模型版本管理 + 驗證閘門 + 重訓管線（S3）

> **狀態（2026-07-11）**：完成。把「重訓 → 驗證 → 部署/擋下」做成一條自動管線，改動面涵蓋模型載入路徑、
> `/servo/model_info`、版本目錄結構——**但 `predict_servo` 對外介面與輸出 schema 不變、下游零感知，全套測試維持綠**
> （147 passed）。三條驗收（正常轉正 / 爛候選擋下 / 回滾）皆實跑通過。

**Registry 結構**（[`src/models/servo_model_registry.py`](../src/models/servo_model_registry.py)，命名避開既有
`model_registry` 演算法工廠）：

```
models/registry/
  registry.json          # {active_version, versions:{v1:{摘要}, ...}}（純欄位切換，無 symlink、Windows 相容）
  v1/  servo_clf.joblib  servo_reg.joblib  servo_feature_config.json  metrics.json
  candidate_<ts>/        # 重訓管線輸出，轉正前的暫存（gitignore，不提交）
```

每版 `metrics.json` 記錄部署決策所需證據：留出 macro-F1 / DV R²、訓練資料摘要、特徵組、**模型檔 CRC32**（完整性）、
訓練時 config 快照、**FMCRD 溯源指紋**、時間戳。**遷移**：現行部署模型登記為 **v1**（回填 macro-F1 **0.8187** /
DV R² **0.9444** + CRC32 + 溯源）；`registry.json` 與 `v*/` 進 git 白名單（`models/` 未被忽略，`candidate_*/` 忽略）。
**載入集中化**：`load_active()` / `load_version(v)` 為唯一載入來源，`predict_servo` 與 FastAPI 啟動經此載入、
`/servo/model_info` 回傳 `model_version`、告警引擎的 `model_version` 由 active 版本帶入（不再硬編）。

**驗證閘門**（[`src/pipeline/validation_gate.py`](../src/pipeline/validation_gate.py)）——依序執行，任一 FAIL 即整體
FAIL，結果寫候選目錄的 `gate_report.json`：

| 檢查 | 內容 |
| --- | --- |
| 1 完整性 | 必要檔案齊全；`feature_config` 欄位與其 feature_set 定義、且與 `config.reference_feature_set` 一致；模型檔 CRC32 與 metrics 記錄相符 |
| 2 煙霧測試 | 載入候選對 demo 列跑 `predict_servo`，輸出 schema 與值域合法 |
| 3 留出指標 | **重新計算**候選在 committed test split 的 macro-F1 / DV R²（不採信自報），要求 ≥ active − 容忍帶 |
| 4 AE 單調性 | 候選含 DL/AE 才執行（重建誤差 LN→HI 非遞減），否則 SKIP 記錄 |

**容忍帶依據**（`config.yaml::servo_gate`，預設 macro-F1 / DV R² 各 **0.005**）：訓練隨機性（seed、CV fold 洗牌）會讓
留出 macro-F1 浮動約 ±0.01；容忍帶**吸收此隨機性、但不放行實質退化**——候選最多可比 active 低 `tolerance`，不得更低。
閘門本身有攔截測試（[`tests/test_validation_gate.py`](../tests/test_validation_gate.py)）：偽造缺檔 / CRC 竄改 / 特徵
不符 / 退化模型（常數類別）各驗證真的被擋。

**重訓管線**（[`scripts/retrain_pipeline.py`](../scripts/retrain_pipeline.py)）：訓練 → `candidate_<ts>/`（不碰 active）
→ 閘門 → PASS 轉正 `v<n+1>` + 切 active + log 對比；FAIL 保留候選與 gate_report、active 不動、exit 1。
`--dry-run`（驗證不切換）、`--data-config`（訓練資料子集，如 `{"train_frac":0.1}`；S4 漂移劇本用此參數口）。
`train_servo.run(out_dir, data_config)` 為最小侵入新增、預設行為不變。

**回滾程序**：編輯 `models/registry/registry.json` 的 `active_version` 改回目標版本（或 `python -c
"from src.models import servo_model_registry as r; r.set_active('v1')"`），**服務重啟/重載**後全鏈路恢復——
`predict_servo` / `/servo/model_info` / 告警事件 `model_version` 皆隨之切回。已實測 v2→v1 回滾全鏈路恢復。

指令與參數見 [README §5.3](../README.md)；本節（registry + 驗證閘門 + 重訓）為 [README §9 「MLOps 閉環（已完成）」](../README.md) 的一環。

## 15. 漂移偵測閉環（S4）

> **狀態（2026-07-11）**：完成。閉環 = 漂移偵測 → 自動觸發重訓 → 驗證閘門 → 版本切換。`scripts/run_drift_demo.py`
> 一鍵全程無人工介入、可重複（結尾 reset 回 v1）。三條結構保證實跑通過。**核心設計原則：退化 ≠ 漂移。**

**兩個 autoencoder，職責相反（勿混淆）**：

| AE | 擬合資料 | 誤差意義 | 用途 |
| --- | --- | --- | --- |
| `servo_dl` 神經 AE（§7） | **僅健康 LN** | 隨退化**上升**（LN 低→HI 高） | **健康指標**（上升是好訊號） |
| S4 漂移 AE（PCA 線性） | 該版本**全類別**訓練資料 | 退化**在分布內**（低），只有真 off-manifold 才高 | **漂移偵測**（低是好訊號） |

漂移 AE（[`src/monitor/drift_detector.py`](../src/monitor/drift_detector.py)）在 engineered 空間、n_components 由**累積解釋
變異量 95%** 決定（記錄於每版 `drift_baseline.json`）。**版本綁定**：每版存自己的 `drift_ae.joblib` + `drift_baseline.json`
（隨版本走）。**觸發**：滾動重建誤差 > 訓練 P95 持續 N 窗。**PSI 為診斷、非觸發**——同質退化段（如純 HI）的邊際分布
本就異於混合訓練分布，PSI 會對正常退化誤觸，違反「HI 不誤觸」硬要求，故只回報不觸發。

**核心研究發現：重建式漂移偵測的盲區**（[`scripts/validate_drift_blindspot.py`](../scripts/validate_drift_blindspot.py)、
`outputs/metrics/drift_blindspot.json` + 圖）：資料集自身的 **noisy LO domain shift 並非可重建式偵測的漂移**——noisy LO
在全類別 PCA 空間**內插於 LN↔MED**（兩者都在訓練內），recon error（~0.05）與分布內類別無異、遠低於 P95（0.15）。
即**無監督重建式偵測的盲區 = 類別條件 / 流形內偏移**；這與 §11 的 domain shift 三重證據（train-LO n=65 不均、SMOTE/CTGAN
補不動、FLAML 留出崩盤）互相印證——那是**分類邊界**難度、非分布層級位移。**這是發現，不是失敗**：它精確界定了偵測器
的適用範圍（感測器/工況層級的分布位移），並說明實務上為何需要多訊號並用。

**漂移劇本用注入式感測器漂移**（誠實標註）：對 replay 段的電流通道施以 **gain×1.3**（`config.yaml::servo_drift.injection`，
`publisher` 端注入）。**感測器增益漂移為真實世界典型故障模式，此處為注入模擬**，給 demo 一個真正 off-manifold 的分布位移：
recon error 5.8 » P95 0.15 → DRIFT；而 HI 退化 0.05 « P95 → 不誤觸。

**閉環**（[`src/monitor/closed_loop.py`](../src/monitor/closed_loop.py)）：DRIFT → **背景執行緒**（不阻塞監控串流）跑
重訓管線（`--data-config inject_drift` 把漂移工況併入訓練資料）→ 閘門 → 預設 dry-run（`servo_drift.auto_retrain` 可切
全自動轉正）。因果鏈回寫事件流（`drift_detected → retrain_started → retrain_finished(gate, version)`），儀表板分色顯示。
**重訓再擬合既有部署模型家族**（`fixed_clf_model`），不在每次漂移重新架構搜尋（那需完整重驗證）；也避免 append 增強列的
近重複在 CV 洩漏而選到過擬合模型。

**驗收（實跑，`drift_demo.json`）**：(a) HI 退化不誤觸 ✅；(b) 注入漂移觸發 ✅；(c) 重訓後 v2 消化（recon 回落 < P95）✅。
漂移資料上分類 macro-F1 **v1 0.440 → v2 0.834**。

**誠實揭露**：
- **互補訊號（信心）**：注入漂移段 `model_confidence` **不降反升**（0.79→1.00）——模型對 OOD 資料「自信地錯」。故信心**非**
  可靠漂移訊號、重建式偵測才是必要；儀表板保留信心滾動圖供對照。
- **標籤假設**：demo 重訓納入漂移段時標籤現成（replay 自帶 `ylabel`）；**真實工廠中漂移後新資料需標註流程**（人工檢修
  記錄回填等），這是模擬簡化。
- 結構保證有單元測試（[`tests/test_drift_detector.py`](../tests/test_drift_detector.py)）：(a) HI 不誤觸、(b) 注入段對
  v1 觸發、(c) v2 消化。

一鍵劇本啟動與預期輸出見 [README §5.4](../README.md)；此閉環完整鏈路（串流→告警→漂移→重訓→閘門→registry）
總覽見 [README §9 「MLOps 閉環（已完成）」](../README.md)。

## 16. 方法論：資料洩漏的三種形態與防護

> **狀態（2026-07-11）**：本專題在三個不同階段各抓到一次資料洩漏，形態不同、機制同源——**評估時看見了不該看見的
> 資訊**。並列於此，作為方法論成熟度的證據（比任何單一指標更能說明防護意識）。

| # | 階段 | 洩漏形態 | 症狀 | 防護 |
| --- | --- | --- | --- | --- |
| 1 | 類別不均實驗（§11） | **重採樣洩漏** | 在 CV 外先 SMOTE，合成樣本的鄰居跨 fold 出現在訓練與驗證兩側 | 整條 `imblearn.Pipeline`（Scaler→重採樣→分類器）交給 CV，**重採樣只在每個 fold 的訓練分割內**發生 |
| 2 | 特徵組 / 模型選擇（§11） | **測試集選擇洩漏** | 用 test 表現挑特徵組 / 模型 = 拿測試集調選擇 | 選擇一律以 **train 的分層 CV**（`reference_feature_set` 依 train CV 選 full；留出 test 只評估一次） |
| 3 | 漂移重訓（§15） | **增強近重複洩漏** | append 只縮放電流的複製列，非電流特徵完全相同 → 近重複跨 fold → CV 選到過擬合的 GB（留出 0.671） | 重訓**再擬合既有部署模型家族**（`fixed_clf_model`）、不重新架構搜尋；固定 LR 不受近重複影響 → 留出 0.834 |

**共通教訓**：洩漏不是一種 bug，是一類 bug——任何讓評估集資訊在訓練 / 選擇 / 增強階段「提前現身」的路徑。防護的共同原則
是**把邊界劃在對的地方**：重採樣的邊界在 fold 內、選擇的邊界在 train、增強的邊界在「不製造評估看得到的近重複」。

## 17. Command Center 主前端接入（S5：後端聚合端點 → Next.js 監控頁）

> **狀態（2026-07-11）**：Task 0（後端自包含聚合串流端點）＋ P1（Next.js Servo 即時監控頁）完成。把 S1–S4 的
> MLOps 閉環成果接進 Next.js Command Center 主前端，架構原則**後端算、前端只畫**——視窗聚合、推論、告警判定、
> 漂移判定全留後端，前端僅渲染。舊「即時監控雷達」（`/monitor`，合成 v3）移入側欄「補充 / 歷史」區、續標 Legacy · 合成 demo。

**前後端職責劃分（後端算、前端畫）**：新監控頁的一切決策（近 3 窗多數決平滑、告警遲滯狀態、漂移是否觸發）
都在後端算好、以 enriched JSON 逐視窗發布，前端只負責顯示。這與舊「即時監控雷達」（`/monitor/stream`，合成 v3
資料、前端做部分計算）分屬兩條軌：舊雷達定位為 **legacy / 合成 demo**，保留不動作為 fallback；新 Servo 監控頁走
真實 FMCRD replay + 參考模型 + S2/S4 引擎。

**Task 0 — 後端聚合串流端點**（[`src/monitor/monitor_stream.py`](../src/monitor/monitor_stream.py)，端點於
[`app/backend/main.py`](../app/backend/main.py)）：

- `GET /servo/monitor/stream?segment=normal|drift`（SSE）：**自包含設計**——後端進程內直接復用 replay 素材
  （`data/demo/replay/`）→ 視窗管線（`iter_window_predictions_from_rows`）→ 參考模型推論 → 告警遲滯引擎
  （`AlertEngine`）→ 漂移偵測器（`DriftDetector`），逐視窗發布 enriched 事件：`window_ts`、
  `predicted_health_state`（原始）、`smoothed_state`（近 3 窗多數決）、`health_state_proba`、`degradation_score`、
  `model_confidence`、`risk_level`、`alert_state`（引擎當前狀態）、`drift_status`（重建誤差／P95 閾值／是否觸發）、
  `model_version`、`replay_segment`（數據源標注）。逐視窗推論為阻塞運算，端點以 `run_in_executor` 卸載到工作
  執行緒、不卡事件迴圈。素材缺失回 503。
- `GET /servo/monitor/events?limit&offset`（輪詢式）：回傳最近 N 筆告警／矛盾提示／DRIFT 事件（讀
  `outputs/alerts/*.jsonl`），最新在上、支援分頁；`work_orders` 另附供前端把工單草稿掛回告警。
- `smoothed_state` 是**顯示層決策**（近 K 窗多數決），沿用 Streamlit 版「原始 vs 平滑」的誠實設計——前端同屏顯示
  逐窗原始預測，狀態燈用平滑值。單元測試 [`tests/test_monitor_stream.py`](../tests/test_monitor_stream.py)：enriched
  事件 schema、`drift` 段注入、未知段落拒絕、事件端點分頁。

**P1 — Servo 即時監控頁**（[`web/src/app/servo/monitor/page.tsx`](../web/src/app/servo/monitor/page.tsx)，
新路由 `/servo/monitor`）：`"use client"` 元件以原生 `EventSource` 消費 `/servo/monitor/stream`，前端**只渲染**後端算好的
enriched 事件——

- **頂部**：狀態燈（顏色由 `smoothed_state` 決定）＋同屏顯示逐窗原始預測（沿用 Streamlit「原始 vs 平滑」誠實設計）
  ＋ `model_version` 徽章 ＋ 數據源標注（replay 段落名、模擬性質、是否注入漂移）。
- **中部**：`degradation_score` / `model_confidence` 滾動時序圖（recharts），漂移重建誤差獨立小圖含 P95 閾值線。
- **底部**：事件流（主告警／矛盾提示／DRIFT 分色、最新在上），`alert_triggered` 可展開工單草稿——草稿由後端 LLM
  背景執行緒非同步生成，前端以輪詢 `/servo/monitor/events` 取回並依 `alert_id` 掛回（不阻塞串流）。
- **側欄**：本頁列入「伺服馬達健康（主線）」；舊「即時監控雷達」移入「補充 / 歷史」collapsible 區、續標 Legacy · 合成 demo。
- **韌性**：後端不可用 / 斷線時顯示離線橫幅並保留頁面骨架（不白屏），`EventSource` 自動重試；本輪回放結束顯示完成態。

驗收（本地 FastAPI:8000 + Next.js:3000）：`normal` 劇本完整呈現 LN→LO→HI 演進、DV 升到 ~0.59、風險轉高、
`alert-0001` 遲滯觸發並掛上 LLM 工單草稿；關閉後端後頁面優雅降級為離線態。CI web 三關（eslint 0 新警告 / tsc / build）全綠。

**publisher vs 進程內聚合的定位差異**：[`scripts/servo_replay_publisher.py`](../scripts/servo_replay_publisher.py)
（發布 `GET /servo/stream` 原始列 SSE）保留為**本機開發工具**——供 CLI 消費端（`servo_replay_consumer.py`）與
Streamlit 監控頁（消費同一條 `/servo/stream`）使用。新的 `/servo/monitor/stream` **不依賴**該服務：它在後端進程內
直接讀素材、完成視窗聚合與推論，對外只發布已算好的 enriched 事件。兩者共用同一套底層管線函式（單一真相來源），
差別只在「原始列 over HTTP」vs「enriched 事件、進程內」。
