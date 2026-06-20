# 工作報告 — 2026-06-20

> 專案：AI 伺服馬達故障風險預測與預測性維護建議系統
> 資料集：UCI AI4I 2020（合成）
> Repo：<https://github.com/ChenYuHsu413/AIFinalProject>

---

## 1. 摘要（TL;DR）

從零建立一個端到端的預測性維護原型，並在同一天內推進到「demo 級成熟原型」：完成 CRISP-DM 全流程的程式碼骨架、10 個模型 × 5 個特徵組合的比較、SHAP 單筆解釋、互動式決策門檻、第二階段故障類型分析、What-if 敏感度分析、Optuna 超參數調整、自動產生的模型卡、雙語 README、Docker、GitHub Actions CI，最後 push 到 GitHub 並加 MIT LICENSE。9 個單元測試全程綠燈。

最佳模型：**Gradient Boosting / D_rfe_top8**（F1 0.887、Recall 0.809、ROC-AUC 0.969、PR-AUC 0.906）。

---

## 2. 工作時間軸（依推進階段）

### Phase 1 — 專案骨架（零到可執行）
- 建立 41 個檔案：`src/`（data/features/models/visualization/utils）、`app/`（streamlit + backend）、`tests/`、`scripts/`、`notebooks/`、`outputs/`、`data/`。
- 訂 `config.yaml` 集中管理路徑、模型、特徵組合、風險門檻、維護建議門檻。
- 撰寫 `requirements.txt`、`.gitignore`、`README.md`（英文版）、`data/README.md`、`REPORT_OUTLINE.md`。
- 預先處理 data leakage：把 `TWF/HDF/PWF/OSF/RNF` 從 X 排除；ColumnTransformer 只在訓練折 fit。
- 9 個單元測試（leakage 防護、衍生特徵、scaler、advice 規則、輸出 schema）。

### Phase 2 — 端到端執行（驗證流程跑得通）
- Python 3.14.5 環境安裝完整 ML stack：scikit-learn 1.9、pandas 3.0、XGBoost 3.3、LightGBM 4.6 等。
- 從 UCI 下載 `ai4i2020.csv`（10000 筆，故障比例 3.39%，無缺失）。
- 處理 UCI CSV 的 UTF-8 BOM 問題（`utf-8-sig`）。
- EDA：產出 5 張圖表（target、Type、failure types、numeric 分布、相關熱圖）。
- 訓練：10 模型 × 5 特徵組合 = **50 次訓練**。
- 評估：混淆矩陣 `[[1931, 1], [13, 55]]`、ROC / PR 曲線、原生 + permutation 特徵重要性。
- 啟動 FastAPI（8000）與 Streamlit（8501）並用 curl 驗證 5 個端點。

### Phase 3 — Bug fix + 中文化
- 修 Streamlit 因 sys.path 設定造成的 `ModuleNotFoundError`（在 `streamlit_app.py` 與 `app/backend/main.py` 開頭手動加入專案根目錄）。
- 把所有使用者面向的字串改成繁體中文：Streamlit UI、維護建議文字、FastAPI title/description、CLI 訊息、README、data/README、REPORT_OUTLINE、notebook markdown。
- 同步更新 `test_predict.py` 改為比對中文關鍵字（扭矩 / 刀具磨耗 / 故障機率偏高 / 健康）。

### Phase 4 — 第一輪深度功能（SHAP + 互動門檻）
- 新增 `src/models/explain.py`：`TreeExplainer` 處理 SHAP 跨版本回傳格式差異；只支援樹模型，非樹模型 fallback。
- `evaluate.py` 加入儲存 `outputs/metrics/test_predictions.csv`（2000 筆 `y_true`/`y_proba`）。
- Streamlit 手動預測頁加入「Top 10 SHAP bar chart」（紅推故障、綠拉健康）。
- Streamlit 模型評估頁加入「決策門檻 slider」+ 即時重算混淆矩陣與 P/R/F1。
- 教學亮點：thr=0.30 時 F1=0.892 > 預設 thr=0.50 的 0.887。

### Phase 5 — 第二輪深度功能（二階段 + What-if）
- 新增 `src/models/train_failure_types.py`：對 TWF/HDF/PWF/OSF/RNF 各訓練一個 RF 模型。
- 結果：PWF F1=1.0（確定性條件）、HDF/OSF ~0.89、TWF F1=0 但 ROC 0.96（門檻問題）、RNF F1=0（隨機事件）。
- `predict.py` 新增 `predict_failure_types()`、`predict_full()` 與五個故障類型的中文 notes。
- FastAPI 新增 `/predict_full`、`/failure_type_metrics`；Pydantic schema 對應更新。
- Streamlit 手動頁加入第二階段故障類型 bar chart。
- 新增「What-if 敏感度分析」頁：6 個 slider、1D 掃描曲線、2D 風險地景熱圖（25×25 格）。

### Phase 6 — Optuna + 模型卡
- 新增 `src/models/tune.py`：對 model_comparison 前 3 名跑 Optuna（15 trials × 3-fold CV）。
- 7 個搜尋空間：LR / RF / GBM / SVM / MLP / XGBoost / LightGBM。
- 誠實結論：調參結果（test F1 最高 0.870）**未勝過**原始 GBM 0.887，系統自動保留原始模型。
- 新增 `src/models/model_card.py`：從 `best_model.joblib` + `best_model_meta.json` + `best_model_eval.json` + `tuned_params.json` 動態組合 9 區塊 markdown，自動接入 `evaluate.py`。
- 模型卡誠實呈現「調參未勝過原始」、RNF 限制、PR-AUC 比 Accuracy 重要等說明。

### Phase 7 — DevOps（Docker + CI）
- `Dockerfile`：`python:3.11-slim` + `libgomp1`，layer cache 友善（requirements.txt 先複製）。
- `docker-compose.yml`：YAML anchor 共用 image，`api` + `ui` 兩服務，bind mount `data/` 與 `outputs/`，api 內附 healthcheck。
- `.dockerignore`：排除 .venv、outputs、data/raw、cache、tests、notebooks 等。
- `.github/workflows/ci.yml`：matrix py3.11/3.12，py_compile + pytest + Docker buildx smoke test（GHA cache 加速）。

### Phase 8 — 推上 GitHub
- 初始化 git，commit 52 個檔案，push 到 `ChenYuHsu413/AIFinalProject` main 分支（commit `5600efd`）。
- 加 MIT LICENSE（commit `9ea171c`），對齊 README badge。
- 提供 repo description 與 topics 的 3 種設定方案（網頁、curl + PAT、gh CLI）。

---

## 3. 主要成果清單

### 程式檔案（最終 53 個）
- 11 個資料 / 特徵 / 模型 / 視覺化模組（`src/`）
- 4 個應用層檔案（`app/streamlit_app.py` + FastAPI 三件組）
- 3 個測試檔案（`tests/`）
- 1 個 notebook（`notebooks/01_eda.ipynb`）
- 4 個 DevOps 檔案（`Dockerfile`、`docker-compose.yml`、`.dockerignore`、`.github/workflows/ci.yml`）
- 6 個設定 / 文件（`config.yaml`、`requirements.txt`、`.gitignore`、`README.md`、`LICENSE`、`data/README.md`）

### 訓練 / 評估產物（已生成）
- 50 行的 `model_comparison.csv`（10 模型 × 5 特徵組合）
- 45 個 Optuna trials（`tuning_history.csv`）
- 12 張比較圖、5 張 EDA 圖、5 張評估圖（共 22 張 PNG）
- `MODEL_CARD.md`、`best_model_meta.json`、`tuned_params.json`、`failure_type_comparison.csv`、`feature_importance_*.csv`

### 跑得起來的服務（同時可在背景執行）
- FastAPI（8 個端點）：`/health`、`/model_info`、`/predict`、`/predict_full`、`/batch_predict`、`/metrics`、`/failure_type_metrics` + Swagger UI
- Streamlit（5 個頁面）：手動單筆預測、What-if 敏感度分析、批次 CSV 上傳、模型評估結果、關於本專案

---

## 4. 重要技術決策（與背後理由）

| 決策 | 理由 |
|---|---|
| 用 Pipeline + ColumnTransformer，scaler 只在訓練折 fit | 避免 data leakage，是 sklearn 標準做法 |
| 把 `TWF/HDF/PWF/OSF/RNF` 從 X 移除 | 它們是 `Machine failure` 的確定性成因，否則就是洩漏 |
| 以 F1 而非 Accuracy 選最佳模型 | 故障比例 3.39%，Accuracy 會誤導 |
| 採 `class_weight="balanced"` | 不平衡資料的標準處理 |
| 中文化僅做使用者面向字串、程式碼註解保留英文 | 保留開發者可讀性，同時提供 demo 友善的介面 |
| Optuna 調參失敗時保留原始模型 | 誠實 > 強行勝過；同時把結果寫進模型卡 |
| 自動產生模型卡 | 避免「文件 / 部署模型脫節」這個常見問題 |
| Streamlit + FastAPI 兩種 UI 都做 | 教學 demo（Streamlit）+ 部署故事（FastAPI）兩邊都顧 |
| Docker 共用一個 image、用 compose 改 command | 簡化維運，layer cache 也只算一次 |
| 不把資料 / 模型 binary push 上 GitHub | 資料下載指示放 README；模型可由腳本重生 |

---

## 5. 教學亮點（報告 / 簡報時可強調）

1. **誠實的科學態度** — 45 個 Optuna trials 跑完沒贏，這個結果**直接呈現在 MODEL_CARD.md**。
2. **CV vs Test 差異** — GBM 的 CV F1（0.891）>實際 Test F1（0.869），可作為「為什麼要 held-out test」的例子。
3. **預設閾值不是最優** — 互動門檻頁可現場演示 thr=0.30 比 thr=0.50 同時更好的 F1 與 Recall。
4. **隨機資料無解** — RNF F1=0、ROC=0.62 可作為「沒訊號就沒模型」的例子。
5. **確定性資料完美可分** — PWF F1=1.0（AI4I 中 PWF 由 `torque × rpm` 確定性決定），可作為「資料生成過程決定上限」的反例。
6. **規則式 + 機率式並用** — 維護建議由規則產生（可解釋），故障機率由模型產生（資料驅動），兩者互補。

---

## 6. 已知問題 / 限制

- **AI4I 是合成資料**，所有絕對數值不能直接外推到實際工廠。
- **沒有時間序列**，無法做嚴謹 RUL 預測。
- **決策門檻預設 0.5**，未做成本敏感調整（已在 UI 提供互動探索）。
- **TWF 模型在 thr=0.5 表現差**（F1=0），需動態調門檻才能用。
- **RNF 完全不可預測**（依資料集設計屬隨機）。
- **Docker 沒有本機實際 build**（Docker Desktop 未啟動，YAML 已用 `docker compose config` 驗證）。
- **GitHub Actions CI 尚未跑過第一次**（push 後才會觸發，badge 還是 no status）。
- **GitHub repo description / topics 還沒設**（需用網頁、curl + PAT 或 gh CLI 設定）。

---

## 7. 後續可做（依優先序）

1. **設 repo description / topics**（5 秒，看 README 旁邊或方案 A）。
2. **打開 Docker Desktop 跑一次 `docker compose up`** 確認映像真的能起來。
3. **去 GitHub Actions tab 看第一次 CI run 結果**，需要時微調 workflow。
4. **接 Dependabot**（每月自動 PR 升 deps）。
5. **加入 PR / issue templates** 讓 repo 更專業。
6. **接入實際伺服馬達資料**（電流、電壓、震動、溫度、警報碼、維修紀錄）後，重新訓練並改為 survival / RUL 模型。
7. **MLOps**：定期重訓 cron、特徵漂移監控、模型版本管理、預測審計。
8. **更豐富的維護建議**：把規則式建議改為由 LLM 條件化生成的結構化敘述。

---

## 8. 數字與檔案統計

| 指標 | 值 |
|---|---:|
| 推上 GitHub 的檔案數 | 53（包含 LICENSE） |
| 程式碼總行數（src/ + app/ + tests/，估算） | ~3500 行 |
| 訓練 + 比較模型總次數 | 50 + 45 = 95（含 Optuna trials） |
| 圖表產出數量 | 22 張 PNG |
| 單元測試 | 9 / 9 passed |
| FastAPI 端點 | 8 個 |
| Streamlit 頁面 | 5 個 |
| 啟動腳本指令 | `train`、`evaluate`、`train_failure_types`、`tune`、`predict`、`run_eda`、`streamlit run`、`uvicorn`、`docker compose up` |
| README 章節 | 20 章 |
| Git commits | 2（initial + LICENSE） |

---

## 9. 重要連結

- **GitHub repo**：<https://github.com/ChenYuHsu413/AIFinalProject>
- **CI workflow**：<https://github.com/ChenYuHsu413/AIFinalProject/actions>
- **CI badge**：已在 README 開頭
- **資料集來源**：<https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- **本地 FastAPI**：`http://127.0.0.1:8000`（含 `/docs`）
- **本地 Streamlit**：`http://127.0.0.1:8501`
- **模型卡**：`outputs/models/MODEL_CARD.md`
- **報告大綱**：`outputs/reports/REPORT_OUTLINE.md`
