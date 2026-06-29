# 模組 B 規劃 — IMS 軸承健康度 / RUL 資料流程

> **定位**：在現有 AI4I「靜態製程風險分類」之外，新增一條獨立的
> **動態健康度 / 剩餘壽命（RUL）** 軌，將專案從「靜態風險評估」升級為
> 「靜態風險 + 動態健康度」雙模組。
>
> **狀態（2026-06-29）**：步驟 1–5 已於 2026-06-22 實作完成（見第 5 節交付表）；步驟 6（1D-CNN AE）**不納入本專案**——DL 對照已由 Servo 主線兩階段深度學習達成，IMS 單軌跡 AE 取消（見 [`MODULE_B_DL_PLAN.md`](MODULE_B_DL_PLAN.md)）。

---

## 0. 已定案的範圍決策

| 項目 | 決策 | 理由 |
| --- | --- | --- |
| 使用資料集 | **IMS / NASA Bearing Dataset，只用 Set 2（Test 2）** | NASA 與 IMS 是同一份 run-to-failure 資料；Set 2 通道單純（4 通道）、外圈故障是教科書範例、檔案最少（984），原型最適合 |
| 健康分數方案 | **方案 A：線性 RUL → health（100→0）** | 簡單、誠實、可解釋；拐點法（方案 B）寫進未來 refinement |
| 建模對象 | **只對故障軸承 B1 建退化曲線**，其餘 3 個軸承只留 RMS 當對照 | 聚焦真正退化的軸承，避免無效工作 |
| 進階模型 | **1D-CNN Autoencoder 異常偵測**（取代原「1D-CNN / LSTM 做 RUL 回歸」），放最後 | DL 做 RUL 回歸會撞單軌跡外推牆；改無監督異常偵測。詳見 [`MODULE_B_DL_PLAN.md`](MODULE_B_DL_PLAN.md) |

> 資料以官方下載的 IMS Set 2 原始檔為準；984 檔已於 2026-06-22 完成特徵抽取（見第 5 節交付表）。

---

## 1. IMS Set 2 資料事實

- 984 個無副檔名檔案，檔名即時間戳（例：`2004.02.12.10.32.39`）。
- 每檔 = 1 秒快照 @ 20 kHz，形狀 `20480 × 4`（4 個軸承各 1 個加速度通道），ASCII、tab 分隔、無表頭。
- 每 10 分鐘錄一檔，整個實驗連續運轉至 **B1 外圈剝落（outer race failure）**。
- 無現成 RUL 標籤 —— 由「最後一檔 = 故障」反推。
- 解壓後約 1.5 GB，**不進 git**（`.gitignore` 已排除 `data/`）。

軸承幾何（Rexnord ZA-2115，用於 FFT 故障頻帶特徵）：

- 16 滾子、轉速 2000 rpm（33.3 Hz）。
- 故障特徵頻率：**BPFO ≈ 236 Hz、BPFI ≈ 297 Hz、BSF ≈ 140 Hz、FTF ≈ 15 Hz**。
- B1 為外圈故障 → 重點觀察 **BPFO（236 Hz）** 頻帶能量。

---

## 2. 端到端流程（對齊現有 repo 慣例）

```
data/raw/ims/2nd_test/                    ← 984 個時間戳檔（不進 git）
        │
        ▼  src/data/load_ims.py            解析單檔 + 從檔名取時間戳
(timestamp, ndarray[20480, 4])
        │
        ▼  src/features/vibration_features.py   每檔每通道 → ~20 個時域+頻域特徵
        │
        ▼  src/data/build_ims_dataset.py   依時間排序組表 + 建 RUL / health 標籤（方案 A）
data/processed/ims_set2_features.parquet  ← 984 列 × ~20 特徵 + RUL + health
        │
        ▼  src/models/train_rul.py         特徵(+滑動視窗) → RUL 回歸器
outputs/models/ims_rul.joblib  +  outputs/metrics/ims_rul.json
        │
        ▼  app/streamlit_app.py            新增「動態健康度」頁籤（時間軸播放 100→0）
        │
        ▼（進階，最後做）src/models/train_rul_dl.py   1D-CNN / LSTM 對照
```

---

## 3. 各階段細節

### ① 解析 — `src/data/load_ims.py`
- 讀單檔為 `20480 × 4` ndarray；ASCII、tab 分隔、無表頭。
- 解析檔名時間戳 → `datetime`，作為時間索引。
- 回傳 `(timestamp, array)`；提供「列出資料夾內所有檔並依時間排序」的輔助函式。

### ② 特徵抽取 — `src/features/vibration_features.py`
每檔每通道輸出約 20 個特徵：

- **時域**：RMS、峰值、峰峰值、**峭度 Kurtosis**、偏度、**Crest factor**、Shape factor、Impulse factor、std、mean。
  （峭度跳變 = 外圈早期點蝕的關鍵早警訊號。）
- **頻域（FFT）**：總頻譜能量、各故障特徵頻率附近的頻帶能量，**B1 重點抽 BPFO（236 Hz）頻帶能量**。
- 對 **B1 抽全套**；B2~B4 只留 RMS 當對照。

### ③ 標籤 + 組表 — `src/data/build_ims_dataset.py`（方案 A）
- 依時間排序所有檔。
- `RUL = 最後一檔時間 − 當前時間`（小時）。
- `health = RUL / RUL_max × 100`，線性 100 → 0。
- 輸出 `data/processed/ims_set2_features.parquet`（時間索引）。

### ④ RUL 預測 — 兩種做法（實作後的關鍵發現）

**先試了監督式回歸（`src/models/train_rul.py`），證實在本資料上行不通：**
- RandomForest / GradientBoosting + 滑動視窗特徵 + 前 70%/後 30% 時間切分。
- 結果 **MAE ≈ 120 h、R² ≈ −76**（爛掉）。原因：單一退化軌跡下，測試段 RUL
  範圍（0–48.5 h）完全落在訓練段（50–164 h）之外，而**樹模型無法外推**單調目標。
  這不是調參能解決的，是任務框架問題。此模組保留作為「為何不能這樣做」的對照教材。

**改用趨勢外推法（`src/models/rul_extrapolation.py`，PHM 標準做法）：**
- 由振動特徵（預設 `b1_rms`）建**資料驅動健康指標**並平滑。
- 偵測**退化起點 FPT**：指標連續數點超過 `baseline_mean + n·std`。
- FPT 後以**指數模型 + 滾動視窗**擬合趨勢，外推到失效門檻反推 RUL；
  並對預測值做**物理上限裁切**（RUL 不可超過設計壽命）。
- 評估只在退化區（post-FPT），另報「近失效區」MAE 看收斂。
- 實測：退化提前量 **3.1 天**、退化區 **MAE ≈ 25 h / RMSE ≈ 37 h**。
  本軸承為**突發型失效**（指標末 ~2% 壽命才暴衝），RUL 早期偏粗、近末期收斂，
  屬該失效模式的固有限制，如實呈現。

### ⑤ Dashboard 頁籤 — `app/streamlit_app.py`（＋ `src/ui/charts.py`）
- 新增「動態健康度」分頁，沿時間軸播放 B1：
  - health 100 → 0 漸進下滑；
  - 峭度 / BPFO 能量曲線在外殼發熱前數週跳起；
  - 標出**告警門檻穿越點**（早警提前量，週為單位）。

### ⑥（進階，最後做）深度模型對照

> ⚠️ **方向已更新** —— 原規劃的「1D-CNN / LSTM 做 RUL 回歸」經真實數據評估後**捨棄**：
> 單一退化軌跡下會撞與 ④ 監督式回歸相同的「無法外推」牆，且 DL 更嚴重、報告會露破綻。
> 改為 **1D-CNN Autoencoder 異常偵測**（無監督，與統計 FPT 互相印證）。
> 完整分析與規劃見 [`MODULE_B_DL_PLAN.md`](MODULE_B_DL_PLAN.md)。

---

## 4. config / 測試 / 邊界

- `config.yaml` 新增 `ims:` 區段：raw 路徑、取樣率 `20000`、滑動視窗 `N`、告警門檻、軸承幾何與故障頻率。
- `tests/`：
  - 合成小陣列測特徵函式（脈衝訊號 → 峭度上升）；
  - RUL 非負且隨時間遞減；
  - 輸出表列數 = 檔案數。
- **完全不改動 AI4I 軌**；模組 B 全程平行獨立。

---

## 5. 分階段交付（每步附驗收點）

| 步 | 工作 | 驗收 |
| --- | --- | --- |
| 1 | 下載 Set 2 + `load_ims` | 能讀任一檔得到 `[20480, 4]` |
| 2 | 特徵抽取 → parquet | 984 列輸出，峭度後段明顯上升 |
| 3 | 標籤 + 組表（方案 A） | RUL 單調遞減、health ∈ [0,100] ✅ |
| 4 | RUL：監督式回歸（對照，已知失敗）→ **趨勢外推法（採用）** | 退化區 MAE ≈ 25 h、FPT 提前 3.1 天 ✅ |
| 5 | Dashboard 頁籤 | 健康曲線 100→0 + FPT/告警 + RUL 預測圖 ✅ |
| 6 | 1D-CNN Autoencoder 異常偵測（進階，見 [`MODULE_B_DL_PLAN.md`](MODULE_B_DL_PLAN.md)） | 重建誤差 FPT 與統計 FPT 並列印證（**不採用，2026-06-29**） |

---

## 6. 報告故事線

> 模組 A（AI4I）：靜態製程風險分類 → 模組 B（IMS）：動態健康度 / RUL 預測
> ＝ 從「靜態風險評估」躍升至「動態健康度預測」。

Paderborn / MCSA 免感測器電流診斷寫進「未來工作」當亮點，不在本次範圍。
