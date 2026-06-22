# 工作報告 — 2026-06-22

> 專案：AI 伺服馬達故障風險預測與預測性維護建議系統
> 主題：**環境重建 + 正式部署上線**（從新機器零環境到公開 Demo URL）
> Repo：<https://github.com/ChenYuHsu413/AIFinalProject>
> 線上 Demo：<https://aifinalproject-test.streamlit.app/>

---

## 1. 摘要（TL;DR）

今天把專案搬到新機器（Windows 11 / `D:\AI Class ChenYu\...`，Python 3.14）後從零跑起來，並一路推進到**正式部署上線**。重點四件事：(1) 在全新環境重建 venv、裝套件、下載資料、訓練兩階段模型；(2) 修掉兩個 Streamlit deprecation 警告；(3) 把模型/資料/評估結果推上 GitHub 並精簡部署用 requirements；(4) 成功部署到 Streamlit Community Cloud，拿到公開網址。

今天 3 個 commit：`dace65d` → `f3a29da` → `5f48a25`，全部已 push 到 `main`。線上 Demo 已可公開存取（HTTP 303，正常）。

---

## 2. 今日工作時間軸

### Round 1 — 新環境從零建立 + 啟動伺服器
- 新機器上專案只有原始碼，**沒有 venv、沒有套件、沒有資料、沒有模型**。
- 建立 `.venv`（Python 3.14.3），裝核心套件（numpy/pandas/scikit-learn/streamlit/plotly 等都有 3.14 wheel，安裝順利）。
- 從 UCI 下載 AI4I 2020 資料集 → `data/raw/ai4i2020.csv`。
- 訓練第一階段模型 `python -m src.models.train`：10 模型 × 5 特徵組合 CV。
  - 最佳模型 = **gradient_boosting / D_rfe_top8**（F1 = 0.887、Recall = 0.809、ROC-AUC = 0.969）。
- 啟動 Streamlit（port 8501），確認可正常服務。

### Round 2 — Bug 修復：`stylable_container` deprecation
- 症狀：首頁「快速入口」四個磚塊下方顯示 `stylable_container is deprecated…` 警告。
- 根因：`src/ui/style.py` 的 `zone()` 與 `dash_button_tile()` 用了 `streamlit-extras` 的 `stylable_container`，新版已淘汰。
- 修法：改用原生 `st.container(key=...)`（Streamlit 會自動在 DOM 掛 `st-key-{key}` class），CSS 改成指向 `.st-key-{key}` selector，新增 `_css_container()` helper。外觀完全不變。
- 驗證：用 `streamlit.testing.v1.AppTest` headless 跑首頁，4 個磚塊正常渲染、0 例外、畫面無 deprecation 文字。

### Round 3 — 訓練第二階段故障類型模型
- `python -m src.models.train_failure_types`：每個故障類型一個 RandomForest（class_weight=balanced）。
- 結果：HDF F1=0.88、PWF F1=1.00、OSF F1=0.90（很好）；TWF / RNF F1=0（正樣本太少，全資料各只有 46 / 19 筆，屬資料集限制非程式錯誤）。
- 模型存到 `outputs/models/failure_type_model.joblib`，故障類型分析頁可用。

### Round 4 — Bug 修復：`use_container_width` deprecation
- 症狀：終端機洗版 `Please replace use_container_width with width`（2025-12-31 後將移除）。
- 先驗證安全：逐一檢查 6 種元件（button / download_button / form_submit_button / dataframe / plotly_chart / image）在 Streamlit 1.58 的簽章，全部支援 `width='stretch'`，且是 `use_container_width=True` 的官方等價替換。
- 替換 `use_container_width=True` → `width='stretch'`：`streamlit_app.py` 21 處、`style.py` 1 處、`charts.py` 1 處（docstring）。
- 驗證：AppTest 把**六個頁面**全部跑過 → 0 例外、0 deprecation。

### Round 5 — 部署準備（step 1：模型 + 資料上 GitHub）
- 發現關鍵問題：`.gitignore` 把 `outputs/models/*.joblib` 和 `data/raw/*.csv` 排除，所以雲端部署抓不到模型/資料。
- 設定 repo 本地 git 身分（ChenYuHsu413 / jolene…），`git add -f` 強制把兩個模型 + 資料集 CSV 加入。
- Commit + push：`dace65d`（含 Round 2/4 的 deprecation 修復）。

### Round 6 — 精簡部署用 requirements
- 目標：縮短 Streamlit Cloud build 時間（原本含 xgboost/lightgbm/optuna/jupyter 等很重）。
- **實測驗證法**：把要排除的套件用 `sys.modules[...]=None` 擋掉，再用 AppTest 跑全部頁面 + 實際觸發一次 SHAP 解釋，看會不會壞。
- 抓到一個隱藏依賴：評估頁的 pandas `Styler.background_gradient` 需要 **matplotlib**（非直接 import，純看程式碼會漏掉）→ 必須保留。
- 結論：`requirements.txt` 改為 slim 執行集（11 套件）；完整訓練/開發環境移到 `requirements-dev.txt`（`-r requirements.txt` + 訓練/調參/後端/測試/notebook）。
- 砍掉：seaborn、xgboost、lightgbm（模型是 sklearn 不需要）、optuna、fastapi/uvicorn/pydantic、pytest、jupyter/ipykernel。
- Commit + push：`f3a29da`。

### Round 7 — 補齊評估頁資料 + README Demo 連結
- 跑 `python -m src.models.evaluate`：產生 12 張圖（混淆矩陣 / ROC / PR / 各模型比較 / 特徵重要性）+ 6 個 metrics 檔。
  - 混淆矩陣（測試集 2000 筆）：`[[1931, 1], [13, 55]]`（漏判 13、誤報 1）。
- `git add -f` 強制把 `outputs/figures/*.png` 與 `outputs/metrics/*.csv|json` 加入 → 評估頁線上也有完整圖表（不再只顯示「請先訓練」提示）。
- README：標題下加 **Open in Streamlit 徽章** + 置中的 🚀 線上 Demo 連結。
- Commit + push：`5f48a25`。

### Round 8 — 正式部署上線 + 收尾
- 使用者在 https://share.streamlit.io 部署成功，app 名稱設為 `aifinalproject-test`，與 README 連結對上。
- 驗證線上網址：HTTP 303（Streamlit Cloud 標準轉址，代表 app 已啟動）。
- 清理：刪除沒用到的下載殘留檔 `data/raw/ai4i.zip`（未被任何程式引用、未進版控）。

---

## 3. 最佳模型（重新訓練後）

| 項目 | 值 |
|---|---|
| 模型 | gradient_boosting |
| 特徵組合 | D_rfe_top8（RFE 選 8 個特徵） |
| Accuracy | 0.993 |
| Precision | 0.982 |
| Recall | 0.809 |
| F1 | 0.887 |
| ROC-AUC | 0.969 |
| PR-AUC | 0.906 |

第二階段故障類型模型：RandomForest per-type，HDF/PWF/OSF 表現優異，TWF/RNF 受限於樣本數。

---

## 4. 今日 commit 鏈

```
5f48a25  Add evaluation figures/metrics for deployment; add live demo link to README
f3a29da  Slim requirements.txt for deployment; full set moved to requirements-dev.txt
dace65d  Add trained models + dataset and fix deprecations for deployment
```

3 個 commit，全部已 push 到 `origin/main`。

---

## 5. 服務狀態（停工前）

| 服務 | 位置 | 狀態 |
|---|---|---|
| 線上 Demo | https://aifinalproject-test.streamlit.app/ | ✅ 已部署上線（公開） |
| 本機 Streamlit | port 8501 | ❌ 已關閉 |

本機重啟（已不需重新訓練）：

```powershell
cd "D:\AI Class ChenYu\AIClass\FinalProject\AIFinalProject"
.venv\Scripts\streamlit run app\streamlit_app.py
```

本機重新訓練（需完整環境）：

```powershell
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m src.models.train
.venv\Scripts\python -m src.models.train_failure_types
.venv\Scripts\python -m src.models.evaluate
```

雲端：之後只要 `git push`，Streamlit Cloud 會自動重新部署。

---

## 6. 重要連結

- **線上 Demo**：<https://aifinalproject-test.streamlit.app/>
- **GitHub repo**：<https://github.com/ChenYuHsu413/AIFinalProject>
- **GitHub Actions**：<https://github.com/ChenYuHsu413/AIFinalProject/actions>
- **昨日工作報告**：`outputs/reports/WORK_REPORT_2026-06-21.md`
- **報告大綱**：`outputs/reports/REPORT_OUTLINE.md`
- **模型卡**：`outputs/models/MODEL_CARD.md`

---

今天先這樣 👋
