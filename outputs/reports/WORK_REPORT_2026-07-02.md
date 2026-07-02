# 工作報告 — 2026-07-02

> **狀態（2026-07-02）**：完成**全專案 code review + UI/UX 檢視**，並依嚴重度修復全部主要發現——
> (1) 前端 5 頁 API 呼叫補 catch + 錯誤提示（後端離線不再永久轉圈）；(2) 後端 assistant endpoint
> 輸入驗證（惡意/畸形 payload 由 500 → 200 fallback 或 400）+ 7 個 endpoint 攔 `TypeError`；
> (3) 儀表板 trueLabel 錯配與趨勢圖 24H X 軸重複標籤；(4) 主題統一（亮色主色紫→cyan、暗色純黑→深 slate）；
> (5) 競態條件一族 + 低嚴重度收尾。共 17 個程式檔 + 2 份文件，pytest 129 passed、tsc/eslint 乾淨。
> 相關文件：[`../../docs/WEB_REVAMP_PLAN.md`](../../docs/WEB_REVAMP_PLAN.md)、
> [`../../docs/MODULE_B_PLUS_EXTENSIONS_PLAN.md`](../../docs/MODULE_B_PLUS_EXTENSIONS_PLAN.md)。

## 1. Code review（雙 agent 平行審查 + 實機 UI 檢視）

- **Python 後端**（`app/backend/` + `src/` 重點模組）與 **Next.js 前端**（`web/src` 全部 61 檔）
  各派一個 review agent 平行審查，逐項重讀驗證後回報；UI/UX 由本 session 以 dev server 實測
  （DOM 快照 + CSS 變數檢查 + 亮/暗/手機三情境；preview 截圖工具在本機一直逾時，改用文字工具驗證）。
- **確認乾淨**：train/test 無資料洩漏（洩漏欄位有 drop、特徵選擇只跑訓練摺）、無硬編碼 secret、
  `/figures` 無 path traversal、前後端 API schema 逐欄一致、主題切換無 hydration 問題、
  圖表除零/NaN 皆有防護。
- 發現依嚴重度分級後按「先修會壞給使用者看的」順序處理（§2）。

## 2. 修復內容（依修復順序）

### 2.1 前端 API 呼叫補 catch + 錯誤提示（高）

`servo/assistant`、`servo/dashboard`、`servo/simulator`、`servo/knowledge` 四頁的使用者觸發
API 呼叫原本只有 `try/finally` 沒有 `catch`——後端一掛，assistant 頁**永久顯示「估測中…」**、
其餘頁按鈕按下去毫無反應（unhandled rejection）。比照 `module-a/predict` 的既有 pattern
補 `catch` + 紅色 `Note` 錯誤提示；assistant 頁樣本下拉在估測中 disable，杜絕重複請求競態。

### 2.2 後端 assistant 輸入驗證 + TypeError 攔截（高）

- `src/llm/maintenance_assistant.py` 新增 **`_sanitize_prediction()`** 於 `generate_report` /
  `answer_question` 入口做防禦性清洗（數值欄 coerce float、`top_features` 逐項補預設鍵）。
  root cause 是 `_format_structured` 與**離線 fallback 範本**都硬索引 `t['feature']` / 套 `:.3f`，
  schema 又只宣告 `prediction: dict`——一處清洗同時修好 LLM prompt 與 fallback 兩條路徑。
- `app/backend/main.py`：7 個接受特徵資料的 endpoint（`/predict`、`/batch_predict`、
  `/predict/explain`、`/predict/batch`、`/predict_full`、`/paderborn/predict`、`/servo/predict`）
  的 `except ValueError` 擴為 `except (TypeError, ValueError)`——數值欄夾雜字串/list 時回 400
  而非 500；`/ims/snapshot` 補攔 `FileNotFoundError` → 404。

### 2.3 trueLabel 錯配 + 趨勢圖 X 軸（高/中）

- `servo/dashboard`：`trueLabel` 原本跟著下拉選單即時變，但 `pred` 是上次按估測時的樣本——
  對 #3 估測後切到 #5，✓/⚠ 會拿 #3 的預測比 #5 的標籤。新增 `predIdx` 記住估測當下的樣本，
  比對與「LLM 維護助理」連結都改用它。
- `HealthTrendPanel`：X 軸 `(i*step) % 60` 在 24H 檔位產生 12 個重複的 `00'`（順帶造成重複
  React key、異常紅點畫錯位置）。改為經過時間 `H:MM`（0:00–22:00），每檔位標籤唯一。

### 2.4 主題統一（UI/UX）

- **亮色主色**：靛紫 `oklch(0.511 0.262 277)` → 與暗色主題、側欄 logo 一致的 **cyan 系**
  `oklch(0.52 0.12 205)`（accent / ring 同步）——亮暗色現在是同一個品牌的日夜版本。
- **暗色背景**：純黑 `oklch(0 0 0)` → 帶藍調深 slate `oklch(0.145 0.012 248)`（符合 CSS 註解
  原本宣稱的 "deep slate field"）；側欄 `oklch(0.17)` 比背景淺半階，與內容區有層次。

### 2.5 競態條件 + 低嚴重度收尾

- **競態一族**：`module-a/what-if`（cancelled 旗標 + 切變數時 spinner 重新顯示）、
  `module-b/explore`（請求序號）、`equipment/[id]`（序號 + effect cancelled 旗標）——
  快速切換時舊回應不再蓋掉新結果。
- `lib/fleet.ts`：localStorage 快取加形狀驗證（`topFeature.feature` 必須存在）——
  舊 schema 快取不再讓首頁 render 直接 throw。
- `AlertWorkOrderQueue`：狀態欄只讓**未完成**工單覆蓋告警狀態——已完成舊工單不再把新告警
  顯示成 Resolved。
- `CountUp`：分頁隱藏時 rAF 不觸發、數字凍在 0——隱藏時直接跳到終值。
- 無障礙：`module-c` 與 `module-b-plus/applications` 的兩個 `<select>` 補 `aria-label`。

### 2.6 誠實性措辭修正（依專案紅線）

E3 串流回放原宣稱「與逐前綴重算等價」，但**失效門檻（`fail_percentile`）是整段軌跡離線校準後
固定**，非嚴格「只看前 k 筆」。已同步修正三處：`app/backend/services.py` docstring、
`module-b-plus/applications` 前端說明文字、`docs/MODULE_B_PLUS_EXTENSIONS_PLAN.md`（附日期戳）。

## 3. 驗證

| 項目 | 結果 |
| --- | --- |
| pytest（全套） | **129 passed / 3 skipped** |
| `tsc --noEmit` / `eslint src` | 0 錯誤 |
| 畸形 payload：`{"top_features":[{}]}`、字串 DV | `/servo/assistant/*` **200**（fallback 報告），原本 500 |
| list 塞進 feature 值 | `/servo/predict` **400**，原本 500 |
| 實測離線情境 | 停後端 + 切樣本 → 顯示「估測失敗」提示，不再永久轉圈 |
| 主題 | 瀏覽器實測亮/暗 CSS 變數：primary 皆 cyan 系、暗色背景非純黑、側欄淺半階 |
| 頁面煙霧測試 | 首頁 / assistant / what-if / explore 載入正常、console 無錯誤 |

## 4. 已知未修（留待決定）

- **設計取捨三項**：設備健康卡資訊密度（建議只留分數+狀態+top-1 特徵）、11px 小字（建議底線 12px）、
  雙語標題規則不一——屬美感方向選擇，待拍板再改。
- **`src/models/predict.py:216` 潛伏地雷**：無 `predict_proba` 的模型走 `decision_function`
  批次內 min-max 當機率，單筆預測必然輸出「健康 100 分」。目前 SVC 有 `probability=True` 不受影響；
  修法涉及行為選擇（如何把 margin 轉機率），先記錄不動。
- **公開部署前**：CORS `allow_origins=["*"]` + 伺服器代打 LLM 金鑰需收緊（rate limit / origin 白名單）；
  LLM 提示注入（使用者問題直接拼 prompt）建議在報告限制章節承認。

## 工具備忘

- Claude Preview 的 `preview_screenshot` 在本機持續逾時（eval/snapshot 正常）——UI 驗證可改用
  a11y snapshot + `getComputedStyle` 檢查 CSS 變數，不依賴截圖。
- 背景分頁 rAF 不觸發：headless 檢視時 `CountUp` 類動畫元件顯示起始值而非終值，易誤判為 bug
  （本次順手加了 hidden fallback 後此現象消失）。
