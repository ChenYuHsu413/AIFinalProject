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
- 模組 A / C：`/predict`、`/predict/batch`、`/batch_predict`、`/metrics`、`/paderborn/eval`
- 知識庫：`/knowledge/*`
- 健康檢查：`/health`、Swagger：`/docs`

> IMS / XJTU 原始大型資料集未隨 Space 打包，`/ims/*`、`/xjtu/*` 會回 `available:false`
> （前端會優雅降級）；其餘端點正常。

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
