---
title: Servo Health API
emoji: ⚙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Servo Command Center — FastAPI backend (Hugging Face Space)

這份 `README.md` 是 **Hugging Face Docker Space 的根 README**：上方 YAML frontmatter
告訴 HF 用 Docker SDK、對外服務 **port 7860**（對應根 `Dockerfile` 的 `EXPOSE 7860`）。

此 Space 只跑 **FastAPI 後端**；前端（Next.js）部署在 Vercel，直接呼叫此 Space 的網址。

## 它提供什麼

- Servo 主線：`/servo/predict`、`/servo/fleet`、`/servo/alerts`、`/servo/work_orders`、
  `/servo/simulate`、`/servo/assistant/*`
- 模組 A / C：`/predict`、`/predict_full`、`/predict/batch`、`/batch_predict`、`/metrics`、
  `/metrics/test_predictions`、`/paderborn/eval`、`/paderborn/samples`、`/paderborn/predict`（即時推論）
- 模組 B+：`/xjtu/generalization`、`/xjtu/lobo_loco`、`/xjtu/rul_predictions`、
  `/xjtu/health_overlay`、`/xjtu/replay/*`、`/maintenance/advice`（皆由已提交產物驅動）
- 知識庫：`/knowledge/*`
- 健康檢查：`/health`、Swagger：`/docs`

> **狀態（2026-06-27）**：`data/processed/{paderborn,xjtu}_features.parquet` 已納入 build context，
> 故**模組 C 即時推論**與 **B+ 全頁（E2/E3/泛化/HI 重疊）雲端可運作**。僅 **IMS**（`/ims/*`，其
> 處理後 parquet 未打包）仍回 `available:false`、前端優雅降級；如需 IMS 雲端互動，把
> `data/processed/ims_set2_features.parquet` 加入 `.dockerignore` 白名單即可。

## 怎麼建立此 Space

見專案 [`docs/DEPLOYMENT.md`](https://github.com/ChenYuHsu413/AIFinalProject/blob/main/docs/DEPLOYMENT.md) 的
**§9 免費部署替代方案**。重點：把 `deploy/huggingface/Dockerfile` 放成 Space 根目錄的
`Dockerfile`、把這份檔放成 Space 根目錄的 `README.md`，並一併附上後端程式碼**與執行期資料**：

- `src/`、`app/`、`config.yaml`、`requirements.txt`、`requirements-dev.txt`
- `outputs/models/`、`outputs/metrics/`（模型 + 評估 JSON）
- **`data/processed/servo_features.parquet`、`servo_feature_demo.csv`、`servo_sample_predictions.csv`**
  （`/servo/simulate`、`/servo/samples`、`/servo/fleet` 必需，缺了會 **503**）
- `data/knowledge/`（`/knowledge/*` 用）
- 連同本 repo 的 `.dockerignore`（已設好讓上述已提交檔留在 build context）

> 最快做法：把整包 repo 推到 Space remote（已提交的 `outputs/`、`data/processed/servo_*`、
> `data/knowledge/` 都會跟著進去），再用 `deploy/huggingface/` 的 `Dockerfile`/`README.md`
> 覆蓋 Space 根目錄那兩個檔即可。
