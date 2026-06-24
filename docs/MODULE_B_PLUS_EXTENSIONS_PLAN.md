# 模組 B+ 延伸規劃 — 從「檢視頁」升級為「會做事的系統」

> **定位**：[`MODULE_B_PLUS_XJTU_PLAN.md`](MODULE_B_PLUS_XJTU_PLAN.md) 已把多軌跡 / 多工況的
> **泛化驗證**做完（FPT 跨工況成立、監督式絕對 RUL 受 domain shift 限制）。本文規劃三條
> **延伸軌**，把 B+ 從「攤開結果的分頁」推進成「有研究結論、能給維護建議、能現場 demo」的系統。
> 結果與限制最終回寫 [`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)。
>
> **狀態（2026-06-24）**：**E1（跨工況自適應 RUL）已完成**——baseline 重現 −1.22，三手段
> （壽命比例 / transductive z-score / CORAL）最佳把 LOCO R² 抬到 **−0.92**、壽命比例 oracle 上界
> **+0.15**（診斷出形狀可泛化、瓶頸在壽命尺度）；產物 `outputs/metrics/xjtu_domain_adapt.json`、
> 測試 `tests/test_domain_adapt.py`、Dashboard 消融對照表。**E2（維護建議決策層）已完成**。
> **E3（即時串流回放）已完成**（Plotly 瀏覽器端 frames 動畫逐快照重播、可即時切速、播完停在末幀、
> 零重跑不閃爍；重用既有函式、無新產物）。E1/E2/E3 已抽出至獨立「**B+ 延伸應用**」頁、以三 tab 呈現。
> **三軌延伸 E1/E2/E3 全數完成。** 回寫見 [`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)。

---

## 0. 一句話定位

> B+ 已證明「健康監測能泛化、絕對 RUL 跨工況會崩」。延伸要回答三個「然後呢」：
> **(E1)** 跨工況崩掉的 RUL 能不能用領域自適應/壽命正規化救回來？
> **(E2)** 這些數字怎麼變成「何時該維護」的建議？
> **(E3)** 怎麼讓它看起來像一台在跑的監測台，而不是靜態圖？

---

## 1. 現況基線（延伸從這裡接上，數字來自既有產物）

| 指標 | 值 | 來源 |
| --- | --- | --- |
| 固定參數 FPT 跨工況偵測 | 15/15 軸承、3 工況全數偵測；平均提前 7.18 h | `outputs/metrics/xjtu_generalization.json` |
| LOBO 監督式 RUL（工況內留一軸承） | pooled R² ≈ **−0.62**、MAE ≈ 11.2 h | `outputs/metrics/xjtu_lobo.json` |
| LOCO 監督式 RUL（留一工況） | pooled R² ≈ **−1.22**、MAE ≈ 14.2 h | `outputs/metrics/xjtu_loco.json` |
| 既有特徵 | 11 維時域+H/V（`h_rms`、`h_kurtosis`… `v_rms`） | `src/data/build_xjtu_dataset.py` |

> **E1 的攻擊目標就是把 LOCO 的 −1.22 往上抬**；E2/E3 不改模型，純加值。

---

## 2. 三條延伸軌（控制發散）

| 軌 | 一句話 | 賣點 | 改動面 | 風險 |
| --- | --- | --- | --- | --- |
| **E1 跨工況自適應 RUL**（頭條） | 用領域自適應 / 壽命正規化把 LOCO 救回來 | 把被動的負面結果變成「診斷出 domain shift 並嘗試修復」的研究貢獻 | 新增評估腳本，重用特徵與切分 | 可能修不好——但「試了仍不足」也是更強的結論 |
| **E2 維護建議決策層** | RUL/FPT → 風險等級 + 建議維護時間窗 | 呼應專案名「**預測性維護建議**系統」，把指標變成可行動輸出 | 純後處理 + Dashboard 卡片 | 低；需誠實標註成本參數為示意 |
| **E3 即時串流回放** | 逐快照重播 HI 上升 / FPT 觸發 / RUL 更新 | demo 現場最吸睛，靜態圖→會動的監測台 | 純 UI 層，重用既有函式 | 低；需講清楚是離線資料重播非真即時 |

> **MVP 邊界**：時間只夠一條 → 做 **E1**（報告分數天花板最高）。要 demo 效果 → 加 **E3**。
> E2 介於兩者之間、最快做完。三條沒有相依，可任意組合。

---

## 3. 可重用 vs 需新增

**可直接重用（不改）：**
- `src/models/rul_extrapolation.py`：`build_health_indicator` / `detect_fpt` / `extrapolate_rul`（E3 逐前綴呼叫即可）。
- `src/models/train_rul_lobo.py` / `train_rul_loco.py` 的特徵組與切分邏輯（E1 在其上加自適應層）。
- `src/data/build_xjtu_dataset.py` 產出的每軸承特徵表（三軌共用）。
- `app/streamlit_app.py` 既有 `xjtu_health_overlay`、KPI strip、`style.*` 元件。

**需新增（薄層）：**
- E1：`src/models/eval_xjtu_domain_adapt.py`（在 LOCO 流程外掛特徵對齊 / 壽命比例目標，輸出對照 JSON）。
- E2：`src/models/maintenance_advice.py`（純函式：`(health, fpt, rul, cost_params) → {risk, window, rationale}`）。
- E3：`app/streamlit_app.py` 多軌跡泛化頁新增「回放」區塊（play/step 控制 + 漸進重算），不新增資料。

---

## 4. 各軌詳細

### E1（頭條）：跨工況自適應 RUL —— 攻擊 LOCO 的 −1.22

**問題**：LOCO 失敗的主因是**壽命尺度 + 運轉條件漂移**（C1≈2h、C3 可達 42h；特徵分布隨轉速/負載偏移）。

**三種候選手段（由簡到繁，建議依序試、做成消融表）：**
1. **壽命比例目標（lifetime normalization）**：把標籤從「絕對剩餘小時」改成「剩餘壽命比例 0–1」，
   預測後再乘該軸承總壽命還原小時數。直接針對「尺度差異」這個主因，最便宜。
2. **工況感知特徵正規化（transductive z-score）**：來源各工況用自身統計標準化、目標工況用其
   **未標註特徵**統計標準化（只用 target 的 X、不用 RUL，屬合法無監督領域自適應）。
3. **CORAL 特徵對齊**：把來源特徵協方差對齊到目標（純 numpy 幾十行），再訓練同一回歸器。

**敘事升級**：「跨工況崩潰 = domain shift；我們診斷出主因是壽命尺度，並以 ___ 把 LOCO pooled R²
從 −1.22 提升到 ___（如實填）。」**就算只部分改善或無效**，也比現況的被動結論強。

**風險/誠實**：手段 2/3 使用 **target 的未標註特徵**（推論期可得、無 RUL 標籤），須在報告寫明屬
transductive / 無監督 DA，非偷看答案；改善幅度如實呈現，不過度宣稱「解決」。

### E2：維護建議決策層 —— 把數字變成行動

**輸入**：單軌跡的健康指標曲線、FPT、RUL 估計（皆既有）。
**輸出**：`{風險等級(綠/黃/紅), 建議維護時間窗, 理由}`。
**邏輯（最小可解釋版）**：
- 風險等級：健康指標相對 baseline/告警門檻的位置 + 是否已過 FPT。
- 建議時間窗：`現在 + RUL × (1 − 安全裕度)`，安全裕度為可調參數（如 0.3）。
- 成本對照（選配）：給定「非預期停機成本 vs 計畫維護成本」，比較「現在修 vs 等到 T 修」的期望成本，標示建議點。

**Dashboard**：每顆軸承一張卡片「剩餘 ≈ X h｜風險：黃｜建議於 T 前安排維護｜依據：HI 已過 FPT、距告警 Y h」。
**誠實**：屬決策支援**啟發式**，成本參數為示意值，未對真實維護結果做驗證；定位與既有 sidebar
「DECISION SUPPORT · NOT CONTROL」一致。

### E3：即時串流回放 —— 會動的監測台

**做法**：選一顆軸承，沿時間軸**逐快照**餵入，畫面即時更新：HI 曲線一格格長、跨過 FPT 門檻時亮燈、
RUL 以當前可見前綴重新外推。play / pause / step 控制 + 速度調整。
**重用**：`build_health_indicator` / `detect_fpt` / `extrapolate_rul` 對「前 k 筆」呼叫即可。
**誠實**：是**離線資料重播**的視覺化，非真實即時感測串流；ESP32 真即時接入仍列未來工作。

---

## 5. config / 測試 / 邊界

- E1：`config.yaml` `xjtu:` 區段新增 `domain_adapt:` 子段（手段開關、安全裕度、是否用壽命比例）。
  測試：合成兩組不同尺度的序列，驗壽命比例還原正確、transductive 標準化不洩漏 target 標籤。
- E2：`maintenance_advice.py` 為純函式 → 單元測試覆蓋綠/黃/紅邊界與時間窗計算；不碰既有主線。
- E3：純 UI；以既有單軸承資料跑通，無新產物、無新依賴。
- 三軌**完全不改 AI4I（模組 A）與 IMS（模組 B）主線**，亦不改既有 B+ 產物，皆為疊加。

---

## 6. 分階段交付（每步附驗收）

### E1 — 跨工況自適應 RUL
| 步 | 工作 | 驗收 |
| --- | --- | --- |
| 1 | 壽命比例目標版 LOCO | 還原小時數正確；輸出 LOCO-ratio pooled / per-condition R² 對照表 |
| 2 | 工況感知正規化 + CORAL | 兩手段各跑一輪，無 target 標籤洩漏（測試通過） |
| 3 | 消融對照表（baseline vs 三手段） | 一張表並列 pooled R² / MAE，標示最佳手段 |
| 4 | 回寫 `MODULE_B_RESULTS.md` + Dashboard 顯示對照 | 文字成段、誠實標註 DA 性質與改善幅度 |

### E2 — 維護建議決策層
| 步 | 工作 | 驗收 |
| --- | --- | --- |
| 1 | `maintenance_advice()` 純函式 + 單元測試 | 綠/黃/紅與時間窗在合成輸入下符合預期 |
| 2 | Dashboard 每軸承建議卡片 | B/B+ 頁顯示風險等級 + 建議維護時間窗 + 理由 |
| 3 | （選配）成本對照 | 給定成本參數標出「建議維護點」，參數可調 |

### E3 — 即時串流回放
| 步 | 工作 | 驗收 |
| --- | --- | --- |
| 1 | 逐前綴重算 HI/FPT/RUL 的回放核心 | step 一次畫面前進一格、數值正確 |
| 2 | play/pause/速度控制 + FPT 亮燈 | 連續播放至失效；FPT 在正確索引觸發 |

> **最小交付**：E1 步 1–3 + 回寫即可成為報告頭條；E2 步 1–2 即有「建議系統」雛形；E3 步 1 即可 demo。

---

## 7. 報告故事線與誠實聲明

> 模組 B+（現況）：固定參數 FPT 跨工況泛化成立；絕對 RUL 跨工況崩潰（domain shift）→
> **E1**：診斷主因為壽命尺度，嘗試壽命正規化 / 領域自適應，LOCO R² 由 −1.22 改善至 ___（如實）→
> **E2**：把 RUL/FPT 轉成風險等級與維護時間窗，補上「建議」這一段 →
> **E3**：以串流回放把整套監測流程演成會動的儀表台。

**誠實聲明（務必寫進報告）**：
- E1 的領域自適應使用 target 的**未標註特徵**（推論期可得），屬 transductive / 無監督，非偷看標籤；
  改善幅度如實呈現，修不好就誠實說「需更多工況或更強 DA」。
- E2 為決策支援啟發式，成本參數為示意，未對真實維護結果驗證。
- E3 為離線資料重播之視覺化，非真實即時串流；ESP32 真場接入仍列未來工作。
- 不宣稱任何單一工況/單軌跡結果可無條件泛化（延續既有紅線）。

---

> **交叉連結**：[`MODULE_B_PLUS_XJTU_PLAN.md`](MODULE_B_PLUS_XJTU_PLAN.md)（B+ 主線）、
> [`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)（結果回寫處）、
> [`DATASET_EVALUATION.md`](DATASET_EVALUATION.md)（資料取捨）。
