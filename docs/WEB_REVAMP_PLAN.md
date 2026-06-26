# 網頁改版規劃 — Streamlit 單體 → FastAPI + Next.js 前後端分離

> **狀態（2026-06-26）**：Phase 0（API 契約盤點）完成。已掃描 `app/streamlit_app.py`
> （2076 行）與既有 9 個 FastAPI endpoint，產出缺口表與四階段計畫。目標定位為**作品集 /
> 履歷對外展示**，時程寬裕（一個月以上），採**漸進遷移**：先補完後端 API、保留 Streamlit
> 當 fallback，再逐頁搬到 Next.js。本文件為規劃，**尚未動程式碼**。
>
> **技術選型已定案（2026-06-26）**：後端 **FastAPI**（維持現有 `app/backend/`）；前端
> **Next.js（App Router）+ TypeScript + Tailwind + shadcn/ui**，以「盡可能好看 / 高度客製」為
> 目標、保留 SSR；圖表用 Recharts / Plotly.js。部署目標 **GCP Compute Engine VM + nginx
> 反向代理**（nginx 前置，後接 uvicorn/gunicorn 與 Next.js node 程序）。本機在 **Python venv**
> 下開發（venv 建在 D:\，避免還原卡清空 C:\）。課程雖教 Django，但本專案無 DB/admin 需求，
> FastAPI 更合身，故不改用 Django。

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
- **Phase 1 — 補完 FastAPI**：補齊 19 個缺口 endpoint。由易到難，demo 不斷線。
- **Phase 1.5 — 驗證閘**：把 Streamlit 改成「只透過 API 取資料」也能跑；跑得起來代表契約完整。
- **Phase 2 — Next.js 骨架 + 漸進搬頁**：App Router + TS + Tailwind + shadcn/ui；PNG 先 `<img>`，
  互動圖表逐張換 Recharts/Plotly.js；Streamlit 留 fallback。
- **Phase 3 — 部署 + 收尾**：GCP Compute Engine VM；nginx 反向代理前置（`/api` → uvicorn/
  gunicorn 跑 FastAPI、`/` → Next.js node 程序），以 systemd/pm2 常駐；CI 更新；誠實性紅線
  文案驗收；docs 同步補日期戳。

---

## 5. 完整執行順序（backlog）

> 原則：**由易到難、靜態先於重算、讀取先於外呼、demo 全程不斷線**。每個 T 完成後可獨立提交。

### Phase 1 — 補完 FastAPI

**Stage 1.1 — 靜態讀檔類 GET（最快，清掉一大半缺口）**
- [x] T1 `GET /figures/{name}` — `StaticFiles` 掛載 `outputs/figures/`（2026-06-26 完成，含 API 測試）
- [x] T2 `GET /metrics/summary?module=` — 統一讀 `servo_*`、`ims_rul`、`xjtu_*`、`paderborn_eval` JSON（2026-06-26 完成，含測試；module=servo/B/Bplus/C，A 用 /metrics+/model_info）
- [x] T3 `GET /paderborn/eval` — 讀 `paderborn_eval.json`（2026-06-26 完成，含測試；模組 C 整頁正規端點，回完整 method/features/results/summary）
- [x] T4 `GET /xjtu/generalization`、`GET /xjtu/lobo_loco`、`GET /xjtu/domain_adapt`（2026-06-26 完成，含測試；抽 `_read_json_or_empty` helper DRY 掉讀檔模式）
- [ ] T5 `GET /ims/metrics`、`GET /ims/health_curve`（讀 `ims_rul_predictions.csv` + `ims_rul.json`）
- [ ] T6 `GET /knowledge/documents`、`GET /knowledge/search?q=`（包 `maintenance_rag`）
- [ ] T7 `GET /servo/glossary`、`GET /servo/feature_sets`、`GET /servo/samples`、`GET /servo/reference_metrics`

**Stage 1.2 — 模組 A 補洞**
- [ ] T8 `POST /predict/batch`（JSON list；給 What-if 1D/2D 用，取代 625 次單點呼叫）
- [ ] T9 `POST /predict/explain`（SHAP；回 shap_values / feature_values / base_value）
- [ ] T10 `GET /metrics/test_predictions`（回 y_true / y_proba 陣列，門檻運算放前端）

**Stage 1.3 — 大資料 / 重算類**
- [ ] T11 `POST /ims/health_indicator`（換指標重算 HI/FPT）、`GET /ims/snapshot/{i}`（波形/FFT，server 降採樣）
- [ ] T12 `GET /xjtu/health_overlay`、`GET /xjtu/rul_predictions`、`GET /xjtu/replay/{condition}/{bearing}`（預算 frames）
- [ ] T13 `POST /maintenance/advice`（包 `maintenance_advice()`）

**Stage 1.4 — 硬點（最後啃）**
- [ ] T14 `POST /servo/simulate` + `GET /servo/reference_metrics` — 決定同步時限 vs 背景任務
- [ ] T15 LLM 助理：`GET /servo/assistant/providers`、`POST /servo/assistant/report`、`POST /servo/assistant/qa`
      — server 端金鑰、context 契約（帶 `servo_pred`）、SSE 串流

### Phase 1.5 — 驗證閘
- [ ] T16 把 Streamlit 改成 thin client（只透過 API 取資料）或補一組 API 整合測試，確認契約完整

### Phase 2 — Next.js 前端
- [ ] T17 scaffold Next.js（App Router / TS / Tailwind / shadcn/ui）、API client、env 設定、CORS 對接
      （node_modules / venv 建在 D:\，避免還原卡清空 C:\）
- [ ] T18 共用版面 + 導覽 + 模型狀態列（接 `/health`、`/model_info`、`/servo/model_info`）
- [ ] T19 逐頁搬：Servo 儀表板 → 模組 A → 模組 B → 模組 B+ → 模組 C → LLM 助理 → 知識庫 → 關於頁
- [ ] T20 PNG 圖先 `<img>` 顯示；互動圖表逐張換 Recharts / Plotly.js

### Phase 3 — 部署 + 收尾
- [ ] T21 GCP Compute Engine VM 佈署：nginx 反向代理（`/api`→FastAPI、`/`→Next.js）、
      systemd/pm2 常駐、HTTPS 憑證；更新 `.github/workflows/ci.yml`
- [ ] T22 誠實性紅線文案驗收；同步 README §4/§11/§12 與本文件，補日期戳

---

## 6. 注意事項

- 機器有還原卡（C:\ 重開機清空）：`node_modules` 等重工具鏈確保落在 D:/git 範圍或及時 push。
- 本改版屬大型架構更新，依專案 `CLAUDE.md` 第 2 條，動工前後翻 `docs/` 同步、第 1 條補日期戳。
