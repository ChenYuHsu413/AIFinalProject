# 網頁改版規劃 — Streamlit 單體 → FastAPI + Next.js 前後端分離

> **狀態（2026-06-26）**：Phase 0（API 契約盤點）完成。已掃描 `app/streamlit_app.py`
> （2076 行）與既有 9 個 FastAPI endpoint，產出缺口表與四階段計畫。目標定位為**作品集 /
> 履歷對外展示**，時程寬裕（一個月以上），採**漸進遷移**：先補完後端 API、保留 Streamlit
> 當 fallback，再逐頁搬到 Next.js。
>
> **進度（2026-06-26）**：**Phase 1（補完 FastAPI）已完成**——T1–T15 全數補齊，後端新增約
> 25 個 endpoint 覆蓋模組 A/B/B+/C、Servo 主線、訓練模擬器、LLM 助理與知識庫；新增 43 條
> API 測試，全套 93 passed / 1 skipped（其中 API 測試 43 條）。下一步為 Phase 1.5 驗證閘（T16）與 Phase 2 Next.js。
>
> **技術選型已定案（2026-06-26）**：後端 **FastAPI**（維持現有 `app/backend/`）；前端
> **Next.js（App Router）+ TypeScript + Tailwind + shadcn/ui**，以「盡可能好看 / 高度客製」為
> 目標、保留 SSR；圖表用 Recharts / Plotly.js。部署目標 **GCP Compute Engine VM + nginx
> 反向代理**（nginx 前置，後接 uvicorn/gunicorn 與 Next.js node 程序）。本機在 **Python venv**
> 下開發（venv 建在 D:\，避免還原卡清空 C:\）。課程雖教 Django，但本專案無 DB/admin 需求，
> FastAPI 更合身，故不改用 Django。

> **視覺改版（2026-06-26）**：Phase 2 前端進入**工業監控視覺改版**，定位升級為
> **AI Servo Motor Health Command Center（伺服馬達健康監控與智慧維護指揮中心）**。已完成
> 骨架（深色工業主題 token、token 化 `ui-kit`/`sidebar`/`status-bar`、新 IA 加入「運維中心 →
> 告警/工單、報表中心」）、**Overview 監控首頁**（KPI 列／設備健康卡／系統狀態島／告警預覽／
> Legacy 入口）、**Servo 健康儀表板重設計**（HealthScoreGauge＋風險標籤＋感測器趨勢圖＋特徵面板），
> 以及 **/alerts、/reports** 新頁。新增相依 **recharts**。新增 `lib/mock.ts` 集中 mock（機群／告警／
> 工單／遙測），介面對齊未來真 API 形狀，待 Servo Dataset 模組接真實資料即可抽換。Servo
> 既有五頁的 API 串接邏輯不動；模組 A/B/B+/C 保留為 Legacy 對照、移出首頁主視覺。
> Phase 5（Simulator／Glossary／Knowledge／Assistant 暗色化）亦已完成：移除四頁殘留的
> `bg-white`／`from-*-50`／白底 `<select>`／淺色 chip，混淆矩陣改 cyan 色階、LLM 報告 Markdown
> 加 `prose-invert`，並修 stub 頁白底。
>
> **視覺語言對齊 Kiranism（2026-06-26）**：以 `Kiranism/next-shadcn-dashboard-starter` 為**風格
> 參考（非整包導入）**，抓取其 `app-sidebar`／overview `layout`／`header` 原始碼後萃取 dashboard
> 語彙，重建為本專案版：新增真 shadcn `ui/card`（data-slot）與 `ui/badge` 原語、`MetricCard` 改為
> dashboard-01 KPI 樣式（描述→大號 `tabular-nums` 數字→角落 trend Badge→footer 趨勢兩行 + 卡片頂部
> `from-primary/5` 漸層）、新增麵包屑 `Header`（sticky/translucent，取代舊 StatusBar）、Sidebar 改 shadcn
> 中性 `bg-sidebar-accent` active 樣式、Overview 重排為 heading→KPI→Hero 面積圖（`FleetHealthChart`）+
> 設備排行→機群卡→系統狀態→告警→Legacy。保留專案暗色工業色盤（cyan/slate）。驗證：`tsc --noEmit`
> 0 錯、新檔 ESLint 乾淨、全 13 路由皆 200。
>
> **響應式側欄（2026-06-26）**：新增 `sidebar-context`（`SidebarProvider`）；桌機可收合為 `w-16`
> icon rail（tooltip + localStorage 記憶），手機由 `Header` 漢堡鈕開抽屜選單（遮罩 + 點連結自動關）。
> 原本 sidebar 在 `md` 以下完全隱藏、無行動版導覽的缺口已補上。
>
> **設備詳情頁 + 真預測（2026-06-26）**：新增 `/equipment/[id]`，機群卡／排行點擊可 drill-down。
> 頁面上半為該設備 mock 健康快照與遙測趨勢；下半把 mock 機群**橋接到真實參考模型**——挑一筆與
> 該設備狀態相符（ylabel）的 demo 運轉段送入 `POST /servo/predict`，呈現實際模型輸出（機率／異常
> 特徵／建議處置）。Header 麵包屑支援動態路由；mock 與真 API 區塊在 UI 上明確標示。真預測區可
> 「換一筆代表段重估」並顯示所用 demo #；告警表設備名可點進詳情頁。主題背景改為全黑（OLED）。
>
> **機群改接後端真模型（2026-06-26）**：新增後端 `GET /servo/fleet`（`services.servo_fleet`）——
> 合成設備識別（id/名稱/位置/狀態）+ **真實參考模型**在代表性 demo 運轉段上算出的健康分數／狀態／
> 風險／退化／信心／主要異常特徵（非真實 PHM 遙測）。前端新增 `useFleet()`（`lib/fleet.ts`）改打此
> API、mock 當 fallback，Overview 與設備詳情消費之，並標示資料來源（參考模型 / mock）。新增
> `test_servo_fleet`（API 測試）。
>
> **告警／工單也改接真模型（2026-06-26）**：新增後端 `GET /servo/alerts`、`GET /servo/work_orders`
> （`services.servo_alerts/servo_work_orders`）——由**真機群衍生**：風險/狀態/異常特徵來自模型，
> 告警類型與建議處置依 top feature 對應，工單由告警排程（IDs/排程屬示意性運維包裝）。前端新增
> `useFleetOps()`（`lib/ops.ts`），Overview 與 `/alerts` 頁改打 API、mock 當 fallback 並標示來源；
> 新增 `test_servo_alerts`、`test_servo_work_orders`。**仍為標示清楚的 mock**：設備識別、遙測趨勢、
> 14 班次趨勢圖、KPI 趨勢 Badge（皆非真實 PHM 遙測）。
>
本文件相對連結：[`README.md`](../README.md)、[`MODULE_SERVO_PLAN.md`](MODULE_SERVO_PLAN.md)、
[`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)、[`MODULE_B_PLUS_XJTU_PLAN.md`](MODULE_B_PLUS_XJTU_PLAN.md)、
[`MODULE_C_PADERBORN_PLAN.md`](MODULE_C_PADERBORN_PLAN.md)。

---

## 1. 背景與決策

- 現況：`app/streamlit_app.py` 為 2076 行單體 demo UI；`app/backend/`（FastAPI）已存在，
  但只覆蓋 9 個 endpoint，集中在模組 A 與 Servo 預測。
- 問題：Streamlit 大量功能是**直接讀 `outputs/`、直接呼叫 `src/`**，沒走 API。Next.js
  不能像 Streamlit 直接讀檔，所以**改版瓶頸不是前端美術，是「API 還沒補完」**。
- 決策：採前後端分離。後端 FastAPI 固化為穩定 API，前端改用 Next.js（App Router + TS +
  Tailwind + shadcn/ui）。**漸進遷移，Streamlit 全程保留當 fallback，不一次砍掉。**

### 誠實性紅線（搬到新 UI 必須原封帶過去）

依專案 `CLAUDE.md` 第 3 條，新前端的文案與行為**不得**弱化以下揭露：

- AI4I 2020 為**合成資料**，非真實伺服馬達。
- IMS Set 2 為**單軌跡**，結果不得宣稱可泛化；不在單軌跡做深度 RUL 回歸。
- Paderborn 為**真實 PMSM 試驗台**訊號（非產線伺服馬達）、含人工/真實兩種損傷、屬**故障分類非 RUL**、為子集 MVP。
- Servo 主線目前以 **placeholder 合成資料**訓練。
- Servo 結構化輸出的 `consistency_warning`（分類器狀態與 DV 風險矛盾時提醒）必須保留。
- LLM 助理定位為輔助、不直接控制馬達；金鑰遷移到 server 端。

---

## 2. 既有 9 個 endpoint（已覆蓋）

`GET /health`、`GET /model_info`、`POST /predict`、`POST /batch_predict`（CSV multipart）、
`GET /metrics`、`POST /predict_full`、`GET /failure_type_metrics`、`GET /servo/model_info`、
`POST /servo/predict`。

對應已可直接搬的能力：Servo 預測、模組 A 單筆預測、批次 CSV、模型比較表、故障類型指標、側欄模型狀態。

---

## 3. API 契約缺口表（Phase 0 盤點結果）

來源：`app/streamlit_app.py`、`src/ui/servo_views.py`、`app/backend/*`，及被呼叫的
`src/`（`servo_predict`、`servo_simulator`、`maintenance_assistant`、`maintenance_rag`、
`rul_extrapolation`、`maintenance_advice`）。

| 能力 / 畫面 | 目前取資料方式 | 已有 endpoint? | 需新增 endpoint |
| --- | --- | --- | --- |
| 側欄模型卡 + 狀態列 | direct-src | ✅ `/health`、`/model_info` | — |
| 首頁 KPI / 各模組 KPI 條 | direct-file（`_metric_json`） | 部分 | `GET /metrics/summary?module=` |
| Servo 健康儀表板 | direct-file（樣本列）+ direct-src `predict_servo()` | ✅ `/servo/predict`、`/servo/model_info` | `GET /servo/samples`（demo 樣本列） |
| AI 訓練模擬器（請求內即時訓練） | direct-file + direct-src `servo_simulator.*` | ❌ | `POST /servo/simulate`、`GET /servo/reference_metrics` |
| 馬達欄位辭典 / 特徵集 | direct-src 常數 | ❌ | `GET /servo/glossary`、`GET /servo/feature_sets` |
| LLM 維護助理（報告 + 問答，有狀態 + 外呼） | direct-src + 外部 HTTP/SDK | ❌ | `POST /servo/assistant/report`、`/qa`、`GET /servo/assistant/providers`（建議 SSE + server 金鑰） |
| 知識庫 RAG（列檔 + TF-IDF 搜尋） | direct-src in-process index | ❌ | `GET /knowledge/documents`、`GET /knowledge/search?q=` |
| 模組 A 單筆預測 | direct-src | ✅ `/predict_full` | — |
| 模組 A SHAP 解釋 | direct-src `explain_record()` | ❌ | `POST /predict/explain` |
| 模組 A What-if 1D/2D（625 格網） | direct-src 迴圈 `predict_records` | 部分（單點 `/predict`） | `POST /predict/batch`（JSON 批次） |
| 模組 A 批次 CSV 上傳 | direct-file + direct-src | ✅ `/batch_predict` | — |
| 模組 A 模型排行榜（50 組） | direct-file `model_comparison.csv` | ✅ `/metrics` | — |
| 模組 A 門檻調整器（混淆矩陣） | direct-file `test_predictions.csv` | ❌ | `GET /metrics/test_predictions` |
| 模組 A 故障類型第二階段 | direct-src/file | ✅ `/failure_type_metrics` + `/predict_full` | — |
| 模組 A 訓練/評估圖（PNG 圖庫） | direct-file `outputs/figures/*.png` | ❌ | `GET /figures/{name}`（StaticFiles） |
| 模組 B（IMS）健康總覽 + 警報滑桿 + 時間回放 | direct-file | ❌ | `GET /ims/health_curve`、`GET /ims/metrics` |
| 模組 B RUL 預測 vs 實際 | direct-file | ❌ | （併入 `/ims/health_curve`） |
| 模組 B 互動探索（換指標、重算 FPT、原始波形/FFT） | direct-file + direct-src | ❌ | `POST /ims/health_indicator`、`GET /ims/snapshot/{i}`（大 payload，需降採樣） |
| 模組 B+（XJTU）多軌跡泛化（per-condition、15 軸承、HI overlay、LOBO/LOCO） | direct-file | ❌ | `GET /xjtu/generalization`、`/xjtu/health_overlay`、`/xjtu/lobo_loco` |
| 模組 B+ E1 域適應消融 | direct-file | ❌ | `GET /xjtu/domain_adapt` |
| 模組 B+ E2 維護建議卡 | direct-file + direct-src `maintenance_advice()` | ❌ | `GET /xjtu/rul_predictions` + `POST /maintenance/advice` |
| 模組 B+ E3 串流回放動畫 | direct-file + direct-src | ❌ | `GET /xjtu/replay/{condition}/{bearing}`（預算 frames JSON） |
| 模組 C（Paderborn）電流故障診斷 | direct-file `paderborn_eval.json` | ❌ | `GET /paderborn/eval` |
| 關於頁 | 靜態 markdown | n/a | —（前端靜態內容） |

**統計**：約 25 項能力中，6 項已（大致）有 API；**約 19 項需新 endpoint**。模組 B / B+ / C、
訓練模擬器、LLM 助理、知識庫**目前零 API 覆蓋**。

### 三個硬點

1. **LLM 助理有狀態**：依賴 dashboard 頁帶過來的 `servo_pred`，不是無狀態 POST；金鑰目前
   in-process 讀 `.env`。搬後端要設計 session/context 契約、金鑰移 server、建議 SSE 串流。
2. **訓練模擬器是請求內即時訓練 sklearn**：compute-bound，要處理逾時（同步加時限 vs 背景任務）。
3. **大資料**：IMS/XJTU parquet、波形/FFT、replay frames 都不小，需 server 端降採樣/分頁
   （E3 現已降到 ≤100 frames，API 要照做）。

---

## 4. 四階段計畫

- **Phase 0 — API 契約盤點**：✅ 完成（本文件第 3 節）。
- **Phase 1 — 補完 FastAPI**：✅ 完成（2026-06-26）。T1–T15 全數補齊，新增約 25 個 endpoint、
  43 條 API 測試，全套 93 passed / 1 skipped（其中 API 測試 43 條）。由易到難、demo 全程不斷線。
- **Phase 1.5 — 驗證閘**：✅ 完成（2026-06-26）。改採 API 整合測試（`tests/test_backend_integration.py`）
  驗證跨端點頁面流程與一致性，確認契約完整；未重構 Streamlit thin client（風險較低）。
- **Phase 2 — Next.js 骨架 + 漸進搬頁**：🚧 進行中（2026-06-26 起）。T17 scaffold（`web/`，Next 16.2.9
  + React 19 + Tailwind v4 + shadcn/ui，需 Node 24）、T18 版面/導覽/狀態列（亮色漸層品牌 +
  可收合補充模組）皆完成；**T19 全部頁面完成**（Servo 五頁 + 補充模組 A/B/B+/C 各頁 + 關於頁，2026-06-26）。
  Streamlit 全程留 fallback。本機開發：後端 `uvicorn ...:app`、前端 `cd web && npm run dev`，
  `web/.env.local` 的 `NEXT_PUBLIC_API_BASE_URL` 指向後端（本機 8000 被占用時用 8010 等）。
- **Phase 3 — 部署 + 收尾**：GCP Compute Engine VM；nginx 反向代理前置（`/api` → uvicorn/
  gunicorn 跑 FastAPI、`/` → Next.js node 程序），以 systemd/pm2 常駐；CI 更新；誠實性紅線
  文案驗收；docs 同步補日期戳。

---

## 5. 完整執行順序（backlog）

> 原則：**由易到難、靜態先於重算、讀取先於外呼、demo 全程不斷線**。每個 T 完成後可獨立提交。

### Phase 1 — 補完 FastAPI

**Stage 1.1 — 靜態讀檔類 GET（最快，清掉一大半缺口）** ✅ 全數完成（2026-06-26）
- [x] T1 `GET /figures/{name}` — `StaticFiles` 掛載 `outputs/figures/`（2026-06-26 完成，含 API 測試）
- [x] T2 `GET /metrics/summary?module=` — 統一讀 `servo_*`、`ims_rul`、`xjtu_*`、`paderborn_eval` JSON（2026-06-26 完成，含測試；module=servo/B/Bplus/C，A 用 /metrics+/model_info）
- [x] T3 `GET /paderborn/eval` — 讀 `paderborn_eval.json`（2026-06-26 完成，含測試；模組 C 整頁正規端點，回完整 method/features/results/summary）
- [x] T4 `GET /xjtu/generalization`、`GET /xjtu/lobo_loco`、`GET /xjtu/domain_adapt`（2026-06-26 完成，含測試；抽 `_read_json_or_empty` helper DRY 掉讀檔模式）
- [x] T5 `GET /ims/metrics`、`GET /ims/health_curve`（讀 `ims_rul_predictions.csv` + `ims_rul.json`）（2026-06-26 完成，含測試；兩支各司其職不重複 meta，CSV 的 NaN 以 `to_json` 轉成合法 null）
- [x] T6 `GET /knowledge/documents`、`GET /knowledge/search?q=`（包 `maintenance_rag`）（2026-06-26 完成，含測試；RAG 用函式內延遲 import 避免拖慢啟動）
- [x] T7 `GET /servo/glossary`、`GET /servo/feature_sets`、`GET /servo/samples`、`GET /servo/reference_metrics`（2026-06-26 完成，含測試；靜態常數延遲 import、reference 含 dl baseline）

**Stage 1.2 — 模組 A 補洞** ✅ 全數完成（2026-06-26）
- [x] T8 `POST /predict/batch`（JSON list；給 What-if 1D/2D 用，取代 625 次單點呼叫）（2026-06-26 完成，含測試；空陣列回空結果、重用 BatchPredictResponse）
- [x] T9 `POST /predict/explain`（SHAP；回 shap_values / feature_values / base_value）（2026-06-26 完成，含測試；非樹模型回 supported:false，shap 延遲 import）
- [x] T10 `GET /metrics/test_predictions`（回 y_true / y_proba 陣列，門檻運算放前端）（2026-06-26 完成，含測試；2000 列兩陣列）

**Stage 1.3 — 大資料 / 重算類** ✅ 全數完成（2026-06-26）
- [x] T11 `GET /ims/health_indicator?indicator=`（換指標重算 HI/FPT；改用 GET，讀取 idempotent）、`GET /ims/snapshot/{index}`（波形降採樣到 2048 點、頻譜封頂 2 kHz）（2026-06-26 完成，含測試；snapshot 需 1.5GB 原始資料，缺檔回 available:false，雲端 demo 不提供、CI 無資料時測試自適應）
- [x] T12 `GET /xjtu/health_overlay`、`GET /xjtu/rul_predictions`、`GET /xjtu/replay/{condition}/{bearing}`（預算 frames）（2026-06-26 完成，含測試；overlay 每曲線降採樣 ≤200 點、replay ≤100 frames 回結構化欄位不送 HTML、找不到軌跡回 404）
- [x] T13 `POST /maintenance/advice`（包 `maintenance_advice()`）（2026-06-26 完成，含測試；回 asdict(Advice)，預設 alarm 30 / margin 0.3）

**Stage 1.4 — 硬點（最後啃）** ✅ 全數完成（2026-06-26）
- [x] T14 `POST /servo/simulate`（+ `GET /servo/simulate/options`；`/servo/reference_metrics` 已於 T7 完成）（2026-06-26 完成，含測試；實測訓練 <0.4s → 採同步，無需背景任務）
- [x] T15 LLM 助理：`GET /servo/assistant/providers`、`POST /servo/assistant/report`、`POST /servo/assistant/qa`
      （2026-06-26 完成，含測試）— **無狀態**（prediction 放 request body）、server 端 RAG 自動檢索、
      金鑰讀 server `.env`、回 `{text, source}`、無金鑰自動回退離線範本。**不串流**（SSE 留待 Phase 2）。
      純包裝既有 `maintenance_assistant`，未動供應商/模型邏輯，redline 的報告/問答分離 prompt 原樣保留。

### Phase 1.5 — 驗證閘 ✅ 完成（2026-06-26）
- [x] T16 補一組 API 整合測試確認契約完整（採此法，而非重構 2075 行 Streamlit thin client，風險較低）。
      `tests/test_backend_integration.py` 涵蓋真實**跨端點頁面流程與一致性**：
      - Servo 儀表板鏈：`/servo/model_info` → `/servo/samples` → `/servo/predict` → `/servo/assistant/{report,qa}`
        （證明真實 prediction 直接可餵 LLM 助理，正是先前 `top_features.z` 接縫）。
      - 模組 A 鏈：`/predict_full` → `/predict/explain` + `/metrics/test_predictions`（門檻調整器資料）。
      - 模組 B 一致性：`/ims/metrics` 的 `fpt_index` == `/ims/health_indicator`（同 b1_rms）重算結果。
      - 模組 B+ 鏈：`/xjtu/rul_predictions` 某列 → `/maintenance/advice`；`generalization` 軸承數 == overlay 曲線數。
      - 模組 C 一致性：`/paderborn/eval` 的 summary == `/metrics/summary?module=C`。
      全套 99 passed / 1 skipped。**契約完整性確認：Phase 0 盤點的 19 個缺口全數補齊並驗證可組成各頁所需。**

### Phase 2 — Next.js 前端
- [x] T17 scaffold Next.js（App Router / TS / Tailwind / shadcn/ui）、API client、env 設定、CORS 對接
      （2026-06-26 完成）：前端在 `web/`（D:\，避還原卡）。實裝 **Next.js 16.2.9 + React 19 + Tailwind v4**，
      需 **Node 24**（`nvm use 24.14.0`；Node 18 太舊跑不動，見 [[node-toolchain-on-c-drive]]）。shadcn/ui 已 init；
      `src/lib/api.ts` 型別化 client（base URL 走 `NEXT_PUBLIC_API_BASE_URL`，prod 用 `/api` 給 nginx 代理）；
      最小首頁打 `/health` 驗證前後端串通；`web/build` 通過。後端已設 `allow_origins=["*"]`，CORS 對接 OK。
- [x] T18 共用版面 + 導覽 + 模型狀態列（接 `/health`、`/model_info`、`/servo/model_info`）
      （2026-06-26 完成）：左側 grouped 側邊欄（`nav.ts` 鏡像 Streamlit NAV_GROUPS、lucide 圖示、active 高亮、
      誠實性 pill「決策輔助/不控制馬達」）+ 頂部狀態列（後端連線點 + 主線模型/macro-F1 + placeholder 合成資料警示）。
      建立全部 18 條路由（首頁總覽 + 16 個 stub + about），DRY `StubPage` 依 `usePathname` 顯示標籤；build 18 routes 通過。
- [x] T19 逐頁搬（2026-06-26 完成）：**Servo 主線五頁 + 補充模組 A/B/B+/C 各頁 + 關於頁全部完成** — 健康儀表板（`/servo/predict`
      + samples，含一致性警告/真實標籤比對）、AI 訓練模擬器（`/servo/simulate` + options +
      reference_metrics，CSS 混淆矩陣 + DL 唯讀區）、LLM 維護助理（`/servo/assistant/{report,qa,providers}`，
      react-markdown 渲染 + 來源 badge + 離線 fallback）、馬達欄位解釋（`/servo/glossary` + feature_sets）、
      維修知識庫（`/knowledge/{documents,search}`）。**補充模組各頁（2026-06-26 完成）：**
      A — predict / what-if（`/predict/batch` sweep）/ batch（`/batch_predict` CSV）/ evaluation（`/model_info`+
      `/metrics`+`/failure_type_metrics`）；B（IMS）— overview / rul / explore（`/ims/*`，HI 曲線 + 指標切換）；
      B+（XJTU）— generalization（`/xjtu/generalization`+`/xjtu/domain_adapt`）/ applications（`/xjtu/health_overlay`
      多軸承 HI 重疊）；C — Paderborn（`/paderborn/eval`，人工→真實泛化頭條 + baseline/真實雙混淆矩陣）；
      關於頁（三軌定位 + 誠實性聲明）。新增 `ConfusionMatrix` 共用元件；各頁皆附對應誠實性註記。
      共用 `components/ui-kit.tsx`（Card/Stat/Note/Bar/PageTitle）+ `lib/servo.ts` 色彩/標籤 map。
      踩雷修正：`/servo/simulate` 回傳 `task` 為 `classification/regression`（非 `clf/reg`），clf/reg
      判別改用結果欄位（`confusion_matrix` 存在＝分類）。
- [ ] T20 PNG 圖先 `<img>` 顯示；互動圖表逐張換 Recharts / Plotly.js
      （目前機率/特徵/比較/重建誤差皆以 CSS 橫條呈現；IMS/XJTU 曲線等待 T19 補充模組時再評估上圖表庫）

### Phase 3 — 部署 + 收尾
- [~] T21 GCP Compute Engine VM 佈署：nginx 反向代理（`/api`→FastAPI、`/`→Next.js）、
      systemd/pm2 常駐、HTTPS 憑證；更新 `.github/workflows/ci.yml`
      **部署準備完成（2026-06-26）：**新增 `deploy/nginx/servo-command-center.conf`（`/`→:3000、
      `/api/`→:8000 去前綴）、`deploy/systemd/servo-{backend,frontend}.service`、完整 runbook
      [`docs/DEPLOYMENT.md`](DEPLOYMENT.md)（VM/venv/Node24/build/systemd/certbot/redeploy）、CI 新增
      `web` job（lint + tsc + `NEXT_PUBLIC_API_BASE_URL=/api` 的 prod build，已本機驗證 build 通過、22 路由）。
      **實際上線（VM 開立 + DNS + 憑證）待執行。**
- [ ] T22 誠實性紅線文案驗收；同步 README §4/§11/§12 與本文件，補日期戳

---

## 6. 注意事項

- 機器有還原卡（C:\ 重開機清空）：`node_modules` 等重工具鏈確保落在 D:/git 範圍或及時 push。
- 本改版屬大型架構更新，依專案 `CLAUDE.md` 第 2 條，動工前後翻 `docs/` 同步、第 1 條補日期戳。
