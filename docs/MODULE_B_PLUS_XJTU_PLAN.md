# 模組 B+ 規劃 — XJTU-SY 多軌跡 RUL 泛化驗證

> **定位**：用**多條獨立退化軌跡**，把 IMS Set 2「單軌跡禁止做的跨設備驗證」**誠實地
> 做出來**，作為 [`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md) 單軌跡結論的泛化補強。
> 資料來源評估與取捨理由見 [`DATASET_EVALUATION.md`](DATASET_EVALUATION.md)。
>
> **狀態（2026-06-23）**：步驟 1 起步——`src/data/load_xjtu.py` 骨架已建立、`.gitignore`
> 已排除 `data/raw/xjtu/`；待使用者下載 XJTU-SY（Condition 1：Bearing1_1~1_5）後驗證讀檔
> shape `[32768, 2]`。其餘步驟尚未實作。風格對齊 [`MODULE_B_DL_PLAN.md`](MODULE_B_DL_PLAN.md)。

---

## 0. 一句話定位

> 以**同一套固定參數**的健康指標／FPT 管線套用到多顆軸承，並（進階）做
> **leave-one-bearing-out（LOBO）** 監督式 RUL，證明 Module B「單軌跡下趨勢外推優於
> 監督式」的論點是**資料設定問題、非方法優劣**。

---

## 1. 資料事實（以下載後官方 README 為準）

- XJTU-SY：**15 顆軸承**，3 種工況各 5 顆。工況（轉速／徑向力）約為：
  C1 ≈ 2100 rpm / 12 kN、C2 ≈ 2250 rpm / 11 kN、C3 ≈ 2400 rpm / 10 kN。
- 每筆 = **25.6 kHz** 取樣、1.28 s 快照（32768 點）、**每 1 分鐘**錄一筆；
  **2 通道**（水平 H＋垂直 V）。
- 真實 run-to-failure，末端有標注故障型（外圈／內圈／保持架等）。
- 與 IMS 同為振動模態 → **管線高度可重用**。

---

## 2. 範圍決策（控制發散）

| 項目 | MVP 決策 | 理由 |
| --- | --- | --- |
| 工況 | **只取 1 個工況（建議 C1）的 5 顆軸承** | 先證明「同工況多軌跡」泛化即可，避免一次扛跨工況 |
| 通道 | 先用**水平通道（H）** | 與 IMS 單通道建模一致；垂直留對照 |
| 任務 | ① 固定參數健康指標／FPT 跨 5 軌跡 ② （進階）LOBO 監督式 RUL | ① 是 MVP；② 是報告頭條 |
| 產物模式 | 沿用「離線跑 + commit 小 CSV/JSON + Dashboard 只讀」 | 與 parquet/joblib/AE 規劃一致，雲端零負擔 |

---

## 3. 可重用 vs 需新增

**可直接重用（不改）：**
- `src/features/vibration_features.py` 的 `time_domain_features(signal)`、`band_energy`、
  `freq_domain_features`——皆吃單一 1D 訊號，與資料來源無關。
- `src/models/rul_extrapolation.py` 的 `build_health_indicator`、`detect_fpt`、
  `extrapolate_rul`——皆為通用單序列函式，**逐軸承呼叫即可**。

**需新增（薄適配層）：**
- `src/data/load_xjtu.py`：讀 XJTU CSV（32768×2）、解析檔名序號→時間順序、
  回傳 `(index, array)`。對齊 `src/data/load_ims.py` 介面。
- `src/data/build_xjtu_dataset.py`：對單一工況的每顆軸承逐筆抽特徵→每軸承一張特徵表
  （含 RUL/health 標籤，標籤法同方案 A 線性 100→0）。
- `src/models/eval_xjtu_generalization.py`：跑下面第 4 節兩個結果，輸出小產物。

---

## 4. 兩個結果（核心賣點）

### 結果①（MVP，便宜）：固定參數跨軌跡健康監測
- 用**與 IMS 完全相同的 config 規則**（`hi_smooth_window`、`fpt_n_sigma=3`、
  `fpt_consecutive=5`、`fit_window` 等，不逐軸承調參）套到 C1 的 5 顆軸承。
- 輸出每顆軸承的 **FPT 提前量** 與**退化區 MAE/RMSE** → 一張對照表。
- 敘事：「**一套不調參的方法，在 5 條獨立軌跡上都偵測到退化起點**」——這正是 IMS
  單軌跡無法主張的泛化證據（並如實標注哪幾顆收斂好、哪幾顆是突發型誤差大）。

### 結果②（進階，頭條）：leave-one-bearing-out 監督式 RUL
- 因為現在**有多條軌跡**，可合法地用 4 顆訓練、留 1 顆測試（輪流），做監督式 RUL 回歸。
- 把它與 Module B 的「IMS 單軌跡監督式失敗（MAE 120h、R² −76）」並列。
- 誠實敘事升級：「監督式 RUL **不是不能做，而是需要多軌跡**；在 XJTU 多軌跡下做 LOBO，
  結果為 ___（如實填）——印證 Module B『單軌跡下趨勢外推優於監督式』的論點是**資料
  設定問題、非方法優劣**。」

---

## 5. config / 測試 / 邊界

- `config.yaml` 新增 `xjtu:` 區段：raw 路徑、`sampling_rate_hz: 25600`、選用工況與軸承
  清單、通道、沿用的 FPT/外推參數、輸出路徑。
- `tests/`：合成小陣列驗 `load_xjtu` 形狀（32768×2）、health 單調遞減、LOBO 切分不洩漏
  （測試軸承不進訓練）。
- **完全不改動 AI4I 軌與 Module B 既有主線**；本模組純疊加。

---

## 6. 分階段交付（每步附驗收）

| 步 | 工作 | 驗收 |
| --- | --- | --- |
| 1 | `load_xjtu` + 下載 C1 5 顆 | 能讀任一筆得 `[32768, 2]`、依序排序 |
| 2 | `build_xjtu_dataset`（重用特徵函式） | 每軸承一表、峭度/RMS 後段上升 |
| 3 | 結果① 固定參數跨 5 軌跡 FPT/RUL | 5 顆 FPT 提前量＋MAE 對照表 ✅ |
| 4 | （進階）結果② LOBO 監督式 RUL | 輪流留一驗證、MAE 如實報告 ✅ |
| 5 | Dashboard 新增「多軌跡泛化」小頁（讀 CSV/JSON） | 5 軌跡健康曲線疊圖＋LOBO 對照 |
| 6 | 報告段落：單軌跡 → 多軌跡泛化 | 文字成段，併入 `MODULE_B_RESULTS.md` |

> **MVP 邊界**：時間緊可只交付**步 1–3 + 步 6**（結果①）。光是結果① 就足以把「無泛化
> 證據」這個最大缺口補上；步 4 的 LOBO 列為加分。

---

## 7. 報告故事線

> 模組 B（IMS，單軌跡）：誠實揭露單軌跡無法泛化、監督式 RUL 失敗 →
> 模組 B+（XJTU-SY，多軌跡）：用固定參數管線跨 5 條軌跡驗證泛化，並以 LOBO 證明
> 監督式 RUL 的成敗取決於資料設定而非方法。

**誠實聲明（務必寫進報告）**：
- 本模組 MVP 只取**單一工況**，不宣稱跨工況泛化；跨工況列為延伸。
- LOBO 結果如實呈現，不因加分而過度配適。
- Paderborn（電流診斷）、多感測器融合、ESP32 實場接入列為未來工作。
