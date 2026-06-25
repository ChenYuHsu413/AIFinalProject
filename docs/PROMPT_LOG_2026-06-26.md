# Prompt Log — 2026-06-26

> **狀態（2026-06-26）**：模組 Servo 主線重構的延伸調整（側邊欄收合、LLM 多供應商、
> .env 金鑰、Groq 連線修正）。承接 [`PROMPT_LOG_2026-06-25.md`](PROMPT_LOG_2026-06-25.md)。

## 使用者需求（依序）

1. 「可以幫我跑起來嗎」 — 啟動 Streamlit 主程式並確認服務正常。
2. 「可以幫我把補充模組都先收合起來嗎? 然後 AI 診斷的部分 請幫我改成可以接 Gemini,
   openrouter 還有 groq 等免費模型，最後再用 template 回答」
   — 側邊欄補充模組收合；LLM 維護助理改多供應商（免費優先）+ 離線範本墊底。
3. 「可以幫我建立一個 .env 嗎」 — 建立 `.env` 並讓 app 自動載入金鑰。
4. 「我填好了!」 — 測試哪家供應商接通。
5. 「先幫我關閉 server 然後幫我把 prompt 存成 md 檔 生成今天對應的工作檔案 還有更改 readme」
   — 關閉服務、補今日 prompt log 與工作報告、更新 README。

## 執行重點

- **側邊欄**：A/B/B+/C 收進可收合的「📦 補充模組」expander（預設收合，active 在其中時自動展開）；
  主線 Servo 維持置頂常駐。
- **LLM 多供應商**：依 `config.yaml::llm.providers` 順序嘗試 Groq → OpenRouter → Gemini
  （OpenAI 相容，標準庫呼叫，**不增加 runtime 依賴**）→ Anthropic（SDK），全失敗才用離線範本。
- **.env**：新增零依賴 `src/utils/env.py` 載入器（不覆蓋既有環境變數、略過空值）；`.env` 進
  `.gitignore`，提交 `.env.example`。
- **連線實測（2026-06-26）**：Groq 原被 Cloudflare error 1010（403）擋——urllib 預設 UA 之故，
  加自訂 `User-Agent` 後正常；OpenRouter / Gemini 回 429（免費額度 / 上游限流，屬暫時或配額）。
  目前由 Groq 回答。

## 注意事項

- 因 `.env` 自動載入，本機 `test_llm_assistant_fallback` 會偵測到供應商而跳過；已改以
  `available_providers()` 判斷（CI 無 `.env`，仍會測離線範本路徑）。
- 誠實性不變：Servo 為模擬資料、placeholder 訓練、不宣稱 RUL。
