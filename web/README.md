# Web — AI Servo Motor Health Command Center

> **狀態（2026-06-26）**：伺服馬達健康監控與智慧維護指揮中心。暗色工業風 admin dashboard，
> 視覺語言參考 `Kiranism/next-shadcn-dashboard-starter`（風格參考，非整包導入），介面字型採
> 思源黑體（Noto Sans TC）。Servo 主線五頁接真 FastAPI；機群／告警／工單暫為 mock，介面已對齊
> 未來真 API 形狀。完整規劃見 [`../docs/WEB_REVAMP_PLAN.md`](../docs/WEB_REVAMP_PLAN.md)。

本前端是 [Next.js](https://nextjs.org)（App Router）專案，作為後端 FastAPI（`app/backend/`）的展示介面。

## 技術棧

- Next.js 16（App Router, Turbopack）+ React 19 + TypeScript
- Tailwind CSS v4 + shadcn/ui（`@base-ui/react`）+ lucide-react
- recharts（趨勢／面積圖）、react-markdown（LLM 報告）
- 字型：Noto Sans TC（思源黑體，SIL OFL）+ Geist

## 本機開發

需要先啟動後端 FastAPI（見專案根 README §12，預設 `http://localhost:8000`）。

```bash
npm install
# 指向本機 uvicorn；未設定時 fallback 為 "/api"
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

開 [http://localhost:3000](http://localhost:3000)。

| 指令            | 說明                         |
| --------------- | ---------------------------- |
| `npm run dev`   | 開發伺服器（HMR）            |
| `npm run build` | production build             |
| `npm run start` | 啟動 production server       |
| `npm run lint`  | ESLint                       |

## 環境變數

| 變數                        | 用途                                                    |
| --------------------------- | ------------------------------------------------------- |
| `NEXT_PUBLIC_API_BASE_URL`  | FastAPI 後端來源。Dev 指向本機 uvicorn；Prod 可設為 `/api` 由 Nginx 反向代理。 |

## 頁面結構

- **總覽 Overview**（`/`）— KPI、機群健康卡、系統狀態、告警預覽、Legacy 入口
- **Servo 健康儀表板 / 訓練模擬器 / 欄位解釋 / 知識庫 / LLM 助理**（`/servo/*`）— 接真 API
- **告警 / 工單**（`/alerts`）、**報表中心**（`/reports`）
- **Legacy**（`/module-a`、`/module-b`、`/module-b-plus`、`/module-c`）— 對照與歷史模組

## 與後端的資料邊界

- **接真 API**：Servo 預測、訓練模擬、欄位、知識庫、LLM 助理、參考模型指標。
- **暫為 mock**（集中於 `src/lib/mock.ts`）：多設備機群、告警、工單、遙測趨勢；待 Servo Dataset
  模組提供對應 endpoint 後抽換即可，UI 不動。
