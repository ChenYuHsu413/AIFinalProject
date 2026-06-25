# Prompt Log — 2026-06-25

> **狀態（2026-06-25）**：主線重構（模組 Servo）prompt 紀錄。

## 使用者需求（摘要）

把專案主線從「多資料集模型分析平台」改成「以 PHM 伺服馬達滾珠螺桿退化資料為主線、
結合 ML / DL / 訓練模擬器 / LLM 維護助理 / 爬蟲知識庫的應用型系統」。原則：不刪 Model
A/B/C（降為補充）、先分析再改、盡量不破壞既有可運行架構。明確要求：健康狀態分類
（ylabel LN/LO/MED/HI）+ DV 回歸；不把 run_index 當 RUL；正式模型 vs 教學訓練模擬器；
特徵組設計；馬達欄位解釋；LLM 維護助理（含無 API Key fallback）；Python 爬蟲 + 維修
知識庫（白名單、尊重 robots、可離線）；深度學習為第二部分（離線、Dashboard 唯讀）；
首頁與 Demo 以 Servo 為核心；部署只放模型 / demo / metrics / 小知識庫。

後續指示：「可以幫我多建立一個分支，然後用 placeholder 全部先做起來。」

## 執行（本輪）

1. 分析既有架構（單檔 Streamlit + sidebar 導覽、predict/train/model_registry、四軌資料）。
2. 輸出開發計畫（理解 / 目標 / 新模組 / 改頁面 / 檔案清單 / MVP 階段 / mock 範圍）。
3. 建分支 `feat/servo-mainline`，以 placeholder 把 Phase 1–7 全做起來（見
   [`../outputs/reports/WORK_REPORT_2026-06-25.md`](../outputs/reports/WORK_REPORT_2026-06-25.md)）。
4. 全程 `streamlit AppTest` 逐頁驗證、`pytest` 51 passed、FastAPI TestClient 驗證 Servo 端點。

## 注意事項紀錄

- LLM 以 Anthropic SDK，模型 `claude-opus-4-8`；此模型層級**不可送 temperature/top_p**
  （否則 400），故 config 移除 temperature。無 `ANTHROPIC_API_KEY` 走離線 fallback 範本。
- RAG 用 sklearn TfidfVectorizer（字元 n-gram，適合中文），**不新增 runtime 依賴**；爬蟲 /
  LLM / DL 依賴只進 `requirements-dev.txt`，雲端 runtime 不受影響。
- 誠實性：Servo 為模擬資料、placeholder 訓練、不宣稱 RUL；如實標示於 UI / 文件。
