# 部署指南 — AI Servo Motor Health Command Center（GCP VM + nginx）

> **狀態（2026-06-26）**：T21 部署準備。前端（Next.js）+ 後端（FastAPI）以 nginx 反向代理整合於
> 單一 GCP Compute Engine VM；systemd 常駐、certbot HTTPS。本文件附 `deploy/` 下的 nginx 與
> systemd 範本。相對連結：[`WEB_REVAMP_PLAN.md`](WEB_REVAMP_PLAN.md)、[`../README.md`](../README.md)。

## 架構

```
                ┌────────────────────────── GCP VM (Ubuntu) ──────────────────────────┐
   Internet ──► nginx :80/:443
                  ├─ location /        ──► Next.js  (next start)  127.0.0.1:3000
                  └─ location /api/    ──► FastAPI   (uvicorn)     127.0.0.1:8000
                └─────────────────────────────────────────────────────────────────────┘
```

前端以 `NEXT_PUBLIC_API_BASE_URL=/api` build，所有 API 走 `/api/*`，由 nginx 去前綴轉到 uvicorn。

## 0. 快速檢查清單

- [ ] VM（Ubuntu 22.04+，≥ e2-medium）開好，防火牆放行 **80 / 443**
- [ ] 安裝 `python3-venv`、`nginx`、`git`、**Node 24**
- [ ] `git clone` repo（模型產物已隨 repo）
- [ ] 後端：建 venv → `pip install -r requirements-dev.txt` → systemd `servo-backend` → `curl /health`
- [ ] 前端：`NEXT_PUBLIC_API_BASE_URL=/api npm ci && npm run build` → systemd `servo-frontend`
- [ ] nginx：套 `deploy/nginx/...conf`（改 `server_name`）→ `nginx -t` → reload
- [ ] HTTPS：`certbot --nginx -d your.domain`
- [ ] 驗收：外網開首頁、`/api/health`、`/api/servo/fleet` 皆正常

### 連接埠與環境變數

| 項目 | 值 | 備註 |
| --- | --- | --- |
| nginx | 80 / 443 | 唯一對外 |
| FastAPI (uvicorn) | 127.0.0.1:**8000** | 僅本機；nginx `/api/` 轉入 |
| Next.js (next start) | 127.0.0.1:**3000** | 僅本機；nginx `/` 轉入 |
| `NEXT_PUBLIC_API_BASE_URL` | `/api` | **build-time**；務必 build 時帶上 |
| `GROQ_API_KEY` 等 | （選用） | 後端 `.env`，LLM 助理；不設則用離線範本 |

## 1. VM 與系統需求

- GCP Compute Engine，Ubuntu 22.04+，建議 **e2-medium（2 vCPU / 4GB）**以上（servo_reg 模型載入吃記憶體）。
- 防火牆放行 **80 / 443**（HTTP/HTTPS）。後端 8000、前端 3000 **只綁 127.0.0.1**，不對外。
- 安裝：

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip nginx git
# Node 24（前端需 Node 24，見 web/）
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt install -y nodejs
```

## 2. 取得程式碼與模型產物

```bash
cd /home/ubuntu
git clone https://github.com/ChenYuHsu413/AIFinalProject.git FinalProject
cd FinalProject
```

模型產物（`outputs/models/*.joblib` 等）已隨 repo 提交（見 `.gitignore` 的 `!` 例外），
clone 後即可用；若要重訓見 README。RAW 大型資料集非必要（資料相依端點會自適應回 `available:false`）。

## 3. 後端（FastAPI）

```bash
cd /home/ubuntu/FinalProject
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt      # 含 FastAPI 後端
# （選用）LLM 供應商金鑰：
#   echo 'GROQ_API_KEY=...' >> .env
deactivate
```

掛上 systemd：

```bash
sudo cp deploy/systemd/servo-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now servo-backend
curl -s http://127.0.0.1:8000/health        # 應回 {"status":"ok",...}
```

## 4. 前端（Next.js）

`NEXT_PUBLIC_API_BASE_URL` 為 **build-time** 變數，務必在 build 時帶上：

```bash
cd /home/ubuntu/FinalProject/web
NEXT_PUBLIC_API_BASE_URL=/api npm ci
NEXT_PUBLIC_API_BASE_URL=/api npm run build
```

掛上 systemd：

```bash
sudo cp deploy/systemd/servo-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now servo-frontend
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3000   # 應 200
```

> 每次更新前端後需 **重 build 再 restart**：
> `cd web && NEXT_PUBLIC_API_BASE_URL=/api npm run build && sudo systemctl restart servo-frontend`

## 5. nginx 反向代理

```bash
sudo cp deploy/nginx/servo-command-center.conf /etc/nginx/sites-available/servo
# 編輯 server_name 為你的網域（或留 _ 用 IP 測試）
sudo ln -s /etc/nginx/sites-available/servo /etc/nginx/sites-enabled/servo
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

此時用 VM 的外部 IP 或網域開啟即可看到前端；API 走 `/api/*`。

## 6. HTTPS（certbot）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your.domain          # 自動改寫 nginx server 區塊加 443 + 憑證
sudo systemctl reload nginx
```

## 7. 更新流程（redeploy）

```bash
cd /home/ubuntu/FinalProject
git pull
# 後端有改：
. .venv/bin/activate && pip install -r requirements-dev.txt && deactivate
sudo systemctl restart servo-backend
# 前端有改：
cd web && NEXT_PUBLIC_API_BASE_URL=/api npm ci && NEXT_PUBLIC_API_BASE_URL=/api npm run build
sudo systemctl restart servo-frontend
```

## 8. 觀測 / 疑難排解

```bash
sudo systemctl status servo-backend servo-frontend nginx
sudo journalctl -u servo-backend -f
sudo journalctl -u servo-frontend -f
```

- 前端 API 全部 404 / CORS：確認 build 時有帶 `NEXT_PUBLIC_API_BASE_URL=/api`，且 nginx `/api/` 的
  `proxy_pass` 結尾有斜線（去前綴）。
- 502：後端或前端 service 未起，檢查上面 `journalctl`。
- 記憶體不足（OOM kill servo-backend）：升級機型或把 uvicorn `--workers` 調為 1。

---

## 9. 免費部署替代方案（Vercel + Hugging Face Spaces）

> **狀態（2026-06-26）**：不想動用 GCP 額度時的免費路線。**前端 → Vercel**、
> **後端（含 ML 模型）→ Hugging Face Spaces（Docker）**。兩者皆免費、免信用卡；HF 免費
> CPU-basic 為 **2 vCPU / 16 GB RAM**，載入 sklearn servo 模型不會 OOM（512MB 級免費後端會撐爆）。
> 取捨：HF Space 閒置約 48 小時會睡，有人訪問才喚醒（約 30 秒），對作品集 demo 無妨。

與 §1–§8 的「單一 VM + nginx 同源」不同，這裡是**前後端不同網域**，接線方式因此不同：

| 項目 | 單一 VM（§1–§8） | 免費拆開（§9） |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE_URL` | `/api`（nginx 同源代理） | **後端 Space 完整網址**，如 `https://<user>-servo-health-api.hf.space` |
| 後端 CORS | 同源不需 | 已是 `allow_origins=["*"]`，**免改** |
| 後端 port | 8000 | **7860**（HF 慣例） |

### 9.1 後端 → Hugging Face Spaces（Docker）

1. 在 <https://huggingface.co> 建立新的 **Space** → SDK 選 **Docker** → Blank。
2. 把以下放進該 Space 的 git repo 根目錄：
   - `deploy/huggingface/Dockerfile`  → 改名為 **`Dockerfile`**
   - `deploy/huggingface/README.md`   → 放成 **`README.md`**（其 YAML frontmatter 已設 `app_port: 7860`）
   - 後端程式碼：`src/`、`app/`、`config.yaml`、`requirements.txt`、`requirements-dev.txt`
   - 模型與指標：`outputs/models/`、`outputs/metrics/`（已隨本 repo 提交）
   - **執行期資料（缺了相關端點會 503 / available:false）**：`data/processed/servo_features.parquet`、
     `servo_feature_demo.csv`、`servo_sample_predictions.csv`（`/servo/simulate`、`/servo/samples`、
     `/servo/fleet` 必需）；**（2026-06-27 新增）** `paderborn_features.parquet`（`/paderborn/samples`
     → 模組 C 即時推論）、`xjtu_features.parquet`（`/xjtu/replay`、`/xjtu/health_overlay` → 模組 B+
     E3／HI 重疊）；以及 `data/knowledge/`（`/knowledge/*`）。以上皆已在 `.dockerignore` 白名單。
   - 連同本 repo 的 `.dockerignore`（已調好讓上述已提交檔留在 build context；無此檔則 COPY 會抓不到）
   - 最快做法：把本專案 repo 整包推到 Space remote（已提交的 `outputs/`、`data/processed/servo_*`、
     `data/knowledge/` 會一起進去），再用上面兩個檔覆蓋根目錄的 `Dockerfile`/`README.md`。
3. push 後 HF 會自動 build（約數分鐘）。完成後測：
   `curl https://<user>-<space>.hf.space/health` → `{"status":"ok",...}`。

### 9.2 前端 → Vercel

1. 在 <https://vercel.com> **Import** 你的 GitHub repo。
2. **Root Directory** 設為 `web`（重要，否則 Vercel 會在 repo 根找不到 Next.js）。
3. Framework 自動偵測 Next.js；Build/Output 用預設。
4. **Environment Variables** 加一條（Production）：
   `NEXT_PUBLIC_API_BASE_URL = https://<user>-<space>.hf.space`（你的 HF Space 網址，**結尾不加斜線**）。
5. Deploy。完成後開 Vercel 給的網址即是線上版；前端的 API 呼叫會直接打到 HF 後端。

> 改了環境變數要 **Redeploy** 才生效（`NEXT_PUBLIC_*` 是 build-time inline）。

### 9.3 其他免費平台速查（2026）

| 平台 | 免費條件 | 用途 |
| --- | --- | --- |
| **Vercel** | Next.js Hobby 免費 | ⭐ 前端 |
| **Hugging Face Spaces** | 2vCPU/16GB，閒置 48h 睡 | ⭐ ML 後端 |
| Render | 自動偵測 FastAPI，但 512MB + 閒置 15min 睡 | 後端可用但**怕 OOM** |
| Railway | 僅 $1/月額度 | 跑不滿 24/7，要 $5/月 |
| Fly.io | 新用戶**已無免費**、需信用卡 | ❌ |
| GCP always-free e2-micro | 1GB RAM 永久免費（不吃 $300） | 偏緊，可跑後端 only |

> 也可只用 **HF Spaces 跑既有 Streamlit app**（單一平台、免費、16GB），但就看不到 Next.js Command Center。

---

> 備註：repo 內既有的 `Dockerfile` / `docker-compose.yml` 為 **FastAPI + Streamlit** 舊組合的容器化；
> §1–§8 採 **VM + systemd + nginx** 整合 Next.js 前端；§9 為**免費拆開託管**路線。三者擇一即可。
