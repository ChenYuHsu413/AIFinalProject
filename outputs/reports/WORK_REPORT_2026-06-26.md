# 工作報告 — 2026-06-26

> **狀態（2026-06-26）**：模組 Servo 主線（分支 `feat/servo-mainline`）延伸調整完成並驗證。
> 側邊欄補充模組收合、LLM 維護助理改多供應商（免費優先）、新增 `.env` 金鑰載入、修正 Groq
> 連線。承接 [`WORK_REPORT_2026-06-25.md`](WORK_REPORT_2026-06-25.md)（主線首輪重構）。

## 本次完成

1. **側邊欄收合**：模組 A / B / B+ / C 收進可收合的「📦 補充模組（對照 / 歷史）」expander，
   **預設收合**；當前頁屬補充模組時自動展開。主線「模組 Servo」維持置頂常駐。
2. **LLM 維護助理改多供應商**（`src/llm/maintenance_assistant.py`）：
   - 依 `config.yaml::llm.providers` 順序嘗試 **Groq → OpenRouter → Gemini**（三家皆
     OpenAI 相容端點、有免費額度，以 Python 標準庫呼叫，**不新增 runtime 依賴**）→
     **Anthropic**（SDK），全部不可用才退回**離線範本**。
   - UI 顯示偵測到的供應商，回答下方標示實際來源徽章（如 `🟢 Groq` / `⚪ 離線範本`）。
3. **.env 金鑰載入**：新增零依賴載入器 `src/utils/env.py`（不覆蓋既有環境變數、略過空值），
   於 app 啟動與 assistant 匯入時自動載入；`.env` 加入 `.gitignore`，提交 `.env.example` 範本。
4. **Groq 連線修正**：OpenAI 相容請求加自訂 `User-Agent`，解決 Groq 邊緣 Cloudflare error
   1010（HTTP 403）對 urllib 預設 UA 的封鎖。
5. **測試調整**：`test_llm_assistant_fallback` 改以 `available_providers()` 判斷跳過
   （本機有 `.env` 時跳過、CI 無 `.env` 時仍測離線路徑）。

## 連線實測（2026-06-26）

| 供應商 | 結果 | 說明 |
| --- | --- | --- |
| Groq | ✅ 正常 | 加 User-Agent 後可用；目前由它回答 |
| OpenRouter | ⚠ 429 | key 有效，`:free` 模型上游暫時限流 |
| Gemini | ⚠ 429 | key 有效，免費額度用盡（quota exceeded） |

依序嘗試機制下，目前由 Groq 回答；其暫時不可用時自動往下試，全失敗才用離線範本。

## 驗證

- `pytest`：50 passed、1 skipped（fallback 測試本機因有設金鑰而跳過）。
- Streamlit 各頁 AppTest 無例外；模擬器訓練、儀表板預測、LLM 生成（Groq）皆實測通過。
- 服務啟動 / 健康檢查通過；本次結束已關閉伺服器（port 8501 釋放）。

## 後續

- 真實 PHM 資料下載後重跑 build/train/dl 並設 `placeholder: false`（見
  [`../../docs/MODULE_SERVO_PLAN.md`](../../docs/MODULE_SERVO_PLAN.md) 第 10 節）。
- 可考慮把 OpenRouter 預設改為較不易限流的 `:free` 模型，或加重試/退避。
