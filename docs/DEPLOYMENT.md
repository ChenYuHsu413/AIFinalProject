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

> 備註：repo 內既有的 `Dockerfile` / `docker-compose.yml` 為 **FastAPI + Streamlit** 舊組合的容器化；
> 本指南採 **VM + systemd + nginx** 路線整合 Next.js 前端，兩者擇一即可（Docker 化前端為後續選項）。
