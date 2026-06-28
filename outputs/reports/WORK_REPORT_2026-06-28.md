# 工作報告 — 2026-06-28

> **狀態（2026-06-28）**：完成**模組 C 延伸軌 CE1（領域自適應）**——在既有「人工→真實」切分上
> 試三種補救，得到一個誠實且有研究價值的結論：**無監督仿射對齊修不動此 domain shift，但少量真實
> 標籤（few-shot）有效且可量化**。承接 [`WORK_REPORT_2026-06-27.md`](WORK_REPORT_2026-06-27.md)。
> 相關文件：[`../../docs/MODULE_C_PADERBORN_EXTENSIONS_PLAN.md`](../../docs/MODULE_C_PADERBORN_EXTENSIONS_PLAN.md)、
> [`../../docs/MODULE_C_PADERBORN_PLAN.md`](../../docs/MODULE_C_PADERBORN_PLAN.md)、
> [`../../docs/FINAL_REPORT.md`](../../docs/FINAL_REPORT.md)。

## 1. CE1 領域自適應——攻擊人工→真實 0.80 落差

模組 C MVP 的頭條是被動結論：人工(EDM/雕刻)故障訓練在真實加速壽命損傷上幾乎泛化失敗
（baseline macro-F1 1.00 → 真實 0.20、落差 0.80）。CE1 把它升級為「**診斷 domain shift 並嘗試修復**」。

- **新增 `src/models/adapt_paderborn.py`**：在**同一**「人工→真實」切分上跑四評估，固定分類器
  （baseline CV 選最佳 = RandomForest，與 MVP 一致），唯一變動的是特徵轉換 / few-shot 標籤：
  - `baseline`：無自適應（自洽重現 MVP 的 0.20）。
  - `coral`：把源（健康+人工）特徵協方差對齊到真實目標協方差（重用 B+ 的 `coral_align`）。**僅用目標未標註特徵**。
  - `transductive_zscore`：源/目標各以自身統計標準化（重用 `per_condition_zscore`）。**僅用目標未標註特徵**。
  - `few_shot`：每類納入 k 筆真實樣本進訓練、測其餘真實；每個 k 對 5 個隨機抽樣取平均 ± std（取走的 k 筆從 test 移除，無洩漏）。**用了真實標籤，如實揭露**。
- 輸出 `outputs/metrics/paderborn_domain_adapt.json`（`results.{baseline,coral,transductive_zscore,few_shot}` + `summary`）。

## 2. 結果（誠實）

| 手段 | macro-F1（3 類） | outer/inner F1 | 監督性 |
| --- | --- | --- | --- |
| baseline（無自適應） | **0.200** | 0.300 | — |
| CORAL 協方差對齊 | **0.038** | 0.057 | 無監督 |
| 工況感知標準化 | **0.189** | 0.283 | 無監督 |
| few-shot k=1/類 | 0.288 ± 0.036 | — | 用真實標籤 |
| few-shot k=3/類 | 0.380 ± 0.053 | — | 用真實標籤 |
| few-shot k=5/類 | 0.416 ± 0.050 | — | 用真實標籤 |
| few-shot k=10/類 | **0.541 ± 0.042** | — | 用真實標籤 |

- **無監督仿射對齊無效**：CORAL 反而更糟、z-score 幾無變化，皆 ≤ baseline 0.200。另跨 `coral_reg`
  0.01→100 掃描全落在 0.04–0.17（reg 大趨近 identity 才回到 baseline）——**確認非調參問題**，而是
  artificial→real shift **不是單純協方差/平移位移**，線性對齊修不動。
- **few-shot 有效且可量化**：每類僅 10 筆真實標籤即把 macro-F1 自 0.20 抬到 **0.54**，回答「要多少真實標籤才夠」。
- **自洽檢查**：baseline 0.200 與 MVP `paderborn_eval.json` 泛化 0.20 一致。
- **誠實雙指標**：真實測試集無 healthy 類 → 3 類 macro-F1 含一個 0 分項，故另報 outer/inner 二類 `binary_f1`。

## 3. 整合與測試

- **後端**：`services.paderborn_domain_adapt` + `GET /paderborn/domain_adapt`；JSON 納入 git/.dockerignore
  白名單（雲端只讀 JSON、不需重算）。
- **前端**（Next.js 模組 C 頁）：新增「CE1 領域自適應」消融卡——無監督對齊 vs baseline 長條（含
  二類 F1 + 監督性標註）+ few-shot 學習曲線（recharts LineChart，3 類 + outer/inner 兩線）+ 誠實聲明。
- **測試**：`tests/test_adapt_paderborn.py`（5 項：CORAL 縮協方差距離、轉換不觸碰 target 標籤、few-shot
  無洩漏且 test 數正確、k 過大被捨、指標助手 label 處理）+ 後端 `/paderborn/domain_adapt` 端點測試。
- **驗證**：`pytest` **129 passed / 1 skipped**；前端 `tsc` + `eslint` + `next build`（22 路由）全過；
  決定性重跑數字一致。

## 4. docs 同步

- `MODULE_C_PADERBORN_EXTENSIONS_PLAN.md`：CE1 狀態戳 + §4 結果表 + §6 交付表打勾。
- `MODULE_C_PADERBORN_PLAN.md` §4：加 CE1 結果引言。
- `FINAL_REPORT.md`：頂部狀態戳、§8 CE1 表 + 誠實結論、§12 結果彙整列、模組 C 誠實聲明。
- `MODULE_B_RESULTS.md`：模組 C 段補 CE1 一行。

## 待辦 / 後續

- ~~CE1 領域自適應（救人工→真實 0.80 落差）。~~ **✅ 完成（本日）**。
- CE2（MCSA 頻譜邊帶）/ CE3（全 4 工況跨工況泛化）：**需重抓 Paderborn 原始 `.mat`**（`data/raw/paderborn` 目前不存在）。
- few-shot 可延伸：CORAL + few-shot 組合、學習曲線外推到「達 baseline 1.00 需多少真實標籤」。
- HF 後端重新部署（新端點 `/paderborn/domain_adapt` + 白名單 JSON 就緒，設定不需改）。
