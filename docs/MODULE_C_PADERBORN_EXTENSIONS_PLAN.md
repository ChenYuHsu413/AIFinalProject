# 模組 C 延伸規劃 — 從「MVP 分類」升級為「會診斷電流、能跨工況、可修泛化」的系統

> **定位**：[`MODULE_C_PADERBORN_PLAN.md`](MODULE_C_PADERBORN_PLAN.md) 已把模組 C 的 MVP 做完
> （馬達電流 MCSA + 振動的**時域特徵故障分類**；人工→真實泛化實測 **baseline 1.00 → 真實 0.20、
> 落差 0.80**）。本文規劃四條**延伸軌**，把模組 C 從「MVP 分類」推進成「有電流診斷深度、能跨工況、
> 並嘗試修復人工→真實落差、可即時預測」的系統。結果回寫 [`MODULE_C_PADERBORN_PLAN.md`](MODULE_C_PADERBORN_PLAN.md) §4。
>
> **狀態（2026-06-28）**：**CE1（領域自適應）已完成**——新增 `src/models/adapt_paderborn.py`，
> 在同一「人工→真實」切分上跑 baseline / CORAL / 工況感知標準化 / few-shot 四評估，輸出
> `outputs/metrics/paderborn_domain_adapt.json`；FastAPI `GET /paderborn/domain_adapt`；Next.js 模組 C 頁加
> 「CE1 領域自適應」消融卡（無監督對齊長條 + few-shot 學習曲線）。測試 `tests/test_adapt_paderborn.py`（5 項）。
> **核心結論（誠實）**：baseline macro-F1 **0.200**（自洽重現 MVP）；**兩種無監督仿射對齊未能改善**
> （CORAL **0.038**、z-score **0.189**，皆 ≤ baseline；CORAL 跨 reg 0.01–100 全 < 0.20）——人工 EDM/雕刻
> 故障與真實疲勞劣化的差異**不是單純協方差/平移位移**，線性對齊修不動；**few-shot 有效且可量化**：每類
> k=1/3/5/10 真實標籤 → macro-F1 **0.288 / 0.380 / 0.416 / 0.541**（±~0.04，5 次抽樣平均）。
> 因真實測試集無 healthy 類，另報 outer/inner 二類 `binary_f1`。
> **CE4（即時預測端點）先前已完成**（`predict_paderborn.py` + `/paderborn/samples`、`/paderborn/predict`）。
> CE2/CE3 仍為規劃草稿、尚未動工（需重抓 Paderborn 原始 `.mat`；優先序 **CE2 > CE3**）。

---

## 0. 一句話定位

> MVP 已證明「人工故障訓練在真實損傷上幾乎泛化失敗（落差 0.80）」。延伸要回答四個「然後呢」：
> **(CE1)** 這落差能不能用領域自適應 / 少量真實樣本救回來？
> **(CE2)** 只用時域特徵不夠「電流診斷」——加上 MCSA 頻譜邊帶能不能提升？
> **(CE3)** 只用 1 工況——納入全 4 工況、做跨工況分類泛化會怎樣？
> **(CE4)** 怎麼讓它能接收一筆量測、即時吐出故障類別（產品化）？

---

## 1. 現況基線（延伸從這裡接上，數字來自既有產物）

| 指標 | 值 | 來源 |
| --- | --- | --- |
| baseline 分類（健康+人工 · 5 折 CV） | macro-F1 **1.00**（RandomForest） | `outputs/metrics/paderborn_eval.json` |
| 人工→真實泛化 | macro-F1 **0.20**、acc 0.24 | 同上 |
| 落差（domain shift） | **0.80** | 同上 |
| 既有特徵 | 30 維時域（`vib_*` + `cur1_*` + `cur2_*`，各 10） | `src/data/build_paderborn_dataset.py` |
| 工況 / 量測 | 1 工況（N15_M07_F10）、22 碼、440 量測 | `config.yaml` `paderborn:` |

> **CE1 的攻擊目標就是把人工→真實的 0.20 往上抬**；CE2/CE3 拓展特徵與工況；CE4 純產品化。

---

## 2. 四條延伸軌（控制發散）

| 軌 | 一句話 | 賣點 | 改動面 | 風險 |
| --- | --- | --- | --- | --- |
| **CE1 領域自適應救「人工→真實」**（頭條） | 用 CORAL 特徵對齊 / 少量真實樣本 few-shot 微調縮小 0.80 落差 | 把被動的「泛化失敗」變成「診斷 domain shift 並嘗試修復」的研究貢獻（與 B+ E1 同構） | 在 `train_paderborn` 外掛自適應層，重用 E1 的 CORAL | 可能修不好——但「試了仍不足」也是更強結論 |
| **CE2 MCSA 頻譜邊帶特徵** | 加入定子電流故障特徵頻率邊帶能量（真正的 MCSA） | 名實相符的「電流診斷」，補上頻域深度 | 特徵層新增頻域抽取（重用 FFT 模式） | 中；需正確的故障頻率與供電頻率邊帶計算 |
| **CE3 全 4 工況 + 跨工況泛化** | 納入 4 工況、做留一工況（LOCO 類比）分類泛化 | 樣本翻倍、呼應 B+ 跨工況主題 | `config.conditions` 擴充 + LOCO 評估腳本 | 低；解析較重、跑時較長 |
| **CE4 即時預測 FastAPI 端點** | 上傳一筆量測 → 回傳故障類別與機率 | 端到端產品化（與模組 A 對齊） | 新增 `predict_paderborn` + FastAPI 端點 | 低 |

> **MVP 邊界**：時間只夠一條 → 做 **CE1**（直接攻最弱、最有研究價值的 0.80 落差）。
> 要診斷深度 → 加 **CE2**；要更強泛化故事 → 加 **CE3**；要 demo 完整流程 → 加 **CE4**。四條無相依、可任意組合。

---

## 3. 可重用 vs 需新增

**可直接重用（不改）：**
- `src/models/eval_xjtu_domain_adapt.py`：CORAL / transductive z-score 實作（CE1 直接套用到電流+振動特徵）。
- `src/features/vibration_features.py`：時域抽取；其 FFT 模式可延伸出 CE2 的頻譜邊帶。
- `src/models/train_paderborn.py`：`split_artificial_real` / 分類管線（CE1/CE3 在其上加層）。
- `src/models/predict.py` 與 `app/backend/`：模組 A 的 `ModelBundle` / FastAPI 端點樣式（CE4 仿作）。
- `src/ui/charts.py::class_confusion_heatmap`（各軌結果共用）。

**需新增（薄層）：**
- CE1：`train_paderborn` 增「adapt」分支（CORAL 對齊源/目標特徵後再分類；或 few-shot 納入 k 筆真實樣本），輸出 baseline vs adapt 對照。
- CE2：`build_paderborn_dataset` 的特徵抽取增 MCSA 頻帶能量（供電頻率 ± 故障特徵頻率邊帶），新增 `cur*_band_*` 欄。
- CE3：`config.paderborn.conditions` 擴為 4 工況；新增留一工況評估（仿 `train_rul_loco` 思路，但為分類）。
- CE4：`src/models/predict_paderborn.py`（載 `paderborn_clf.joblib` + 特徵抽取）＋ FastAPI `/predict_paderborn` 端點。

---

## 4. 各軌詳細

### CE1（頭條）：領域自適應救「人工→真實」—— 攻擊 0.80 落差

**問題**：人工(EDM/雕刻)故障的訊號分布與真實疲勞損傷不同（covariate shift），故 baseline 完美但真實近乎瞎猜。

**候選手段（由簡到繁，建議依序試、做成消融表）：**
1. **CORAL 特徵對齊**：把人工(源)特徵協方差對齊到真實(目標)未標註特徵（重用 `eval_xjtu_domain_adapt.coral_align`），再用同一分類器預測。**僅用目標未標註特徵**，屬合法無監督 DA。
2. **工況/來源感知標準化**：源、目標各以自身統計標準化（transductive），對齊一階分布。
3. **Few-shot 微調**：納入**少量**真實樣本（如每類 k=3~5 筆）進訓練，量化「需多少真實標籤才夠」。**須誠實揭露用了真實標籤**（非純無監督）。

**敘事升級**：「人工→真實落差 0.80；我們以 CORAL/few-shot 把真實 macro-F1 由 0.20 提升到 ___（如實填）。」**就算只部分改善或無效**，也比 MVP 的被動結論強。

**誠實**：手段 1/2 僅用目標未標註特徵（無監督 DA）；手段 3 用了少量真實標籤須寫明屬 few-shot、非零樣本；改善幅度如實呈現。

**實測結果（2026-06-28，固定分類器 RandomForest、`adapt_paderborn.py`）**：

| 手段 | macro-F1（3 類） | outer/inner F1 | 監督性 |
| --- | --- | --- | --- |
| baseline（無自適應） | **0.200** | 0.300 | — |
| CORAL 協方差對齊 | **0.038** | 0.057 | 無監督（僅 target 特徵） |
| 工況感知標準化（transductive z-score） | **0.189** | 0.283 | 無監督 |
| few-shot k=1/類 | 0.288 ± 0.036 | — | 用真實標籤 |
| few-shot k=3/類 | 0.380 ± 0.053 | — | 用真實標籤 |
| few-shot k=5/類 | 0.416 ± 0.050 | — | 用真實標籤 |
| few-shot k=10/類 | **0.541 ± 0.042** | — | 用真實標籤 |

- **無監督仿射對齊無效**：CORAL 反而更糟（0.038）、z-score 幾無變化（0.189），皆 ≤ baseline 0.200。
  另跨 `coral_reg` 0.01→100 掃描全落在 0.04–0.17（reg→大趨近 identity 才回到 baseline）——確認非調參問題，
  而是 artificial→real shift **不是單純協方差/平移位移**，線性對齊修不動。
- **few-shot 有效且可量化**：每類僅 10 筆真實標籤即把 macro-F1 自 0.20 抬到 **0.54**，量化「要多少真實標籤才夠」。
- baseline 0.200 與 MVP `paderborn_eval.json` 泛化 0.20 一致（自洽檢查通過）。

### CE2：MCSA 頻譜邊帶特徵 —— 名實相符的電流診斷

**做法**：對 `phase_current_1/2` 做 FFT，量取**供電頻率 f_s 附近的故障特徵邊帶**（外/內環故障在 f_s ± k·f_defect 產生邊帶）；新增 `cur1_band_*` / `cur2_band_*` 能量特徵。
**重用**：`vibration_features` 的 FFT / 頻帶能量模式（IMS 已有 `freq_domain_features` 的 band-energy 寫法可參考）。
**誠實**：故障特徵頻率須由軸承幾何與轉速正確推導；標明是否假設固定轉速。

### CE3：全 4 工況 + 跨工況泛化

**做法**：`config.conditions` 由 1 擴為 4（N15_M07_F10 / N09_M07_F10 / N15_M01_F10 / N15_M07_F04）；除既有人工→真實外，新增**留一工況**分類泛化（訓練 3 工況、測第 4 工況），量化跨工況 domain shift。
**重用**：`train_rul_loco` 的留一工況切分思路（改為分類指標）。
**誠實**：解析量增 4 倍；結果限於所選碼。

### CE4：即時預測 FastAPI 端點 —— 產品化

**做法**：`predict_paderborn.py` 載 `paderborn_clf.joblib` + 對上傳量測抽特徵 → 回傳 `{fault_class, class_probabilities}`；FastAPI 加 `/predict_paderborn`（仿模組 A `/predict`）。
**重用**：模組 A `predict.py` 的 `ModelBundle` 載入 + `app/backend` 端點/schema 樣式。
**誠實**：定位為**故障類別**判讀，非 RUL；輸入須為同格式量測。

---

## 5. config / 測試 / 邊界

- CE1：`config.paderborn` 新增 `domain_adapt:` 子段（手段開關、few-shot k）。測試：合成兩分布序列驗 CORAL 對齊、few-shot 不洩漏全部真實標籤。
- CE2：`paderborn` 新增供電頻率 / 故障頻率參數；測試：合成含已知邊帶的訊號驗頻帶能量抽取。
- CE3：`config.conditions` 擴充；測試：留一工況切分正確、無工況洩漏。
- CE4：仿模組 A 端點測試（合成量測 → 回傳合法 schema）。
- 四軌**完全不改 A/B/B+ 主線與模組 C MVP 既有產物**，皆為疊加。

---

## 6. 分階段交付（每步附驗收）

### CE1 — 領域自適應救人工→真實 ✅（2026-06-28 完成）
| 步 | 工作 | 驗收 | 狀態 |
| --- | --- | --- | --- |
| 1 | CORAL 對齊版人工→真實 | 對照表 baseline vs CORAL（macro-F1 / 混淆矩陣），無目標標籤洩漏 | ✅ CORAL 0.038 < baseline 0.200 |
| 2 | few-shot（每類 k 筆真實）微調 | 隨 k 的學習曲線；標明用了真實標籤 | ✅ k=1→10：0.288→0.541 |
| 3 | 消融對照表 + 回寫 | baseline / CORAL / few-shot 並列；誠實標註改善幅度 | ✅ 見 §4 表 + 前端消融卡 |

### CE2 — MCSA 頻譜邊帶
| 步 | 工作 | 驗收 |
| --- | --- | --- |
| 1 | 電流頻帶能量特徵 | 新增 `cur*_band_*`；合成訊號驗證 |
| 2 | 重跑分類 + 對照時域版 | macro-F1 是否提升（baseline 與真實兩者） |

### CE3 — 全 4 工況 + 跨工況泛化
| 步 | 工作 | 驗收 |
| --- | --- | --- |
| 1 | 擴 config + 重建特徵表（4 工況） | parquet 含 4 工況 |
| 2 | 留一工況分類泛化 | 跨工況 macro-F1 對照、混淆矩陣 |

### CE4 — 即時預測端點
| 步 | 工作 | 驗收 |
| --- | --- | --- |
| 1 | `predict_paderborn` + 端點 | 上傳量測 → 合法 JSON（類別 + 機率） |

> **最小交付**：CE1 步 1–3 + 回寫即可成為「診斷並嘗試修復 domain shift」報告頭條；CE2/CE3/CE4 視時間加值。

---

## 7. 報告故事線與誠實聲明

> 模組 C（MVP）：人工→真實泛化崩潰（落差 0.80）→
> **CE1**：以 CORAL / few-shot 嘗試縮小落差，真實 macro-F1 由 0.20 提升至 ___（如實）→
> **CE2**：加入 MCSA 頻譜邊帶，把「電流診斷」做到名實相符 →
> **CE3**：納入全 4 工況、做跨工況分類泛化 →
> **CE4**：以 FastAPI 端點把模組 C 接成可即時預測的服務。

**誠實聲明（務必寫進報告）**：
- CE1 無監督手段僅用 target 未標註特徵；few-shot 須揭露用了少量真實標籤；改善幅度如實呈現，修不好就說「需更多真實樣本 / 更強 DA」。
- CE2 的故障特徵頻率須正確推導；標明轉速假設。
- 全程屬**故障分類**非 RUL；電流為**真實試驗台**訊號但非產線伺服馬達；結果限於所選碼/工況。

---

> **交叉連結**：[`MODULE_C_PADERBORN_PLAN.md`](MODULE_C_PADERBORN_PLAN.md)（模組 C 主線與 MVP 結果）、
> [`MODULE_B_PLUS_EXTENSIONS_PLAN.md`](MODULE_B_PLUS_EXTENSIONS_PLAN.md)（同構的延伸軌寫法）、
> [`DATASET_EVALUATION.md`](DATASET_EVALUATION.md)（資料取捨）。
