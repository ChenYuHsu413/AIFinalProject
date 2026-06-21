# 工作報告 — 2026-06-21

> 專案：AI 伺服馬達故障風險預測與預測性維護建議系統
> 主題：**UI 全面升級**（從「能跑」到「demo-grade」）
> Repo：<https://github.com/ChenYuHsu413/AIFinalProject>

---

## 1. 摘要（TL;DR）

今天整天都在打磨 Streamlit 前端。從昨天的「功能完整但介面像 Jupyter」開始，分**九個小回合**把 UI 推進到「demo 級可視覺展示」：建立主題系統、把 matplotlib 換成 Plotly、加入 shadcn-style 卡片、修兩個棘手的 Streamlit 渲染 bug、重做 sidebar、加入 dashboard 首頁、最後把首頁的卡片做成可點按鈕。

最後一個 commit：`02afcc0`（Dashboard tiles are now clickable）。9 個單元測試全程綠燈，所有改動已 push 到 main。

---

## 2. 今日工作時間軸

### Round 1 — 第一輪 UI 升級（theme + Plotly + shadcn）
- 建立 `.streamlit/config.toml`（teal 工業風 #0d9488 + 柔灰背景）
- 建立 `src/ui/style.py`：全域 CSS、hero banner、note box、metric card、risk pill、section header
- 建立 `src/ui/charts.py`：用 Plotly 取代 matplotlib（SHAP bar、failure_type bar、1D sweep、2D heatmap、CM）
- 重寫 `app/streamlit_app.py` 串接新元件
- 安裝 `plotly`、`streamlit-shadcn-ui`，加入 requirements.txt

### Round 2 — 視覺多樣化 pass（破除「死板」感）
- 每頁加 tabs（手動 4 個 / What-if 3 個 / Eval 3 個）
- 新增 `failure_probability_gauge`（Plotly Indicator 半圓表）
- 新增 `input_radar`（polar 雷達圖：5 個輸入的「運轉指紋」）
- 加 `panel_open/panel_close`、`big_stat`、`kpi_strip`、`hero(chips=...)`
- Hero 漸層動畫 + 結果卡 fade-up CSS
- About 頁直接秀 `docs/0620.png` infographic

### Round 3 — Bug 修復 1：HTML leak 第一次
- 症狀：第一頁出現 `<div class="zone zone-sky">` 純文字
- 根因：`panel_open/panel_close` 用兩個獨立的 `st.markdown` 注入孤立 `<div>` 開頭/結尾，Streamlit 把每個 markdown 當獨立 DOM 元素無法包住中間 form
- 修法：改用 `streamlit-extras.stylable_container`（CSS `:has()` 選擇器）
- 結果是 `style.zone(kind, key)` context manager

### Round 4 — Bug 修復 2：HTML leak 第二次（真正根因）
- 症狀：修完 Round 3 後仍然能看到 HTML 標籤被當文字
- 根因：triple-quoted HTML 在函式內有 8 個空白縮排，Streamlit Markdown parser 把 4+ 空白當 code block escape 標籤
- 修法：新增 `_render(html)` helper 用 `textwrap.dedent(html).strip()` 後再 `st.markdown`
- 套用到 `hero` / `big_stat` / `fallback_metric_card` / `kpi_strip` 四個 helper
- 兩條教訓已存到 memory：[[feedback-streamlit-markdown-indent-gotcha]]、[[feedback-ui-streamlit-polish]]

### Round 5 — Sidebar 重新設計
- 安裝 `streamlit-option-menu`
- gradient brand header（🔧 + 兩行字）
- Bootstrap icon 導覽（bullseye / lightbulb / inbox / bar-chart / info-circle）
- 自訂 `sidebar_model_card`、`sidebar_dataset_card`
- footer 加 pill（`DECISION SUPPORT` / `NOT CONTROL`）+ repo 連結
- 替掉那個醜的綠色 `st.sidebar.success`

### Round 6 — 批次 + 評估視覺升級
- 批次頁：`risk_donut` + `probability_histogram` + Top-5 highest-risk 表
- 評估頁：`leaderboard_bar`（水平 bar chart）+ metric selector + top-N slider
- flat 50 列 dataframe 收進 expander，加 green gradient

### Round 7 — Phase A/B/C/D 大推進
- **Phase A** 新元件：`sparkline`、`metric_with_bar`、`advice_card`、`action_bar`、`dash_tile`
- **Phase B** 手動頁 advice 變 6 類彩色卡（🌡️/⚙️/🔧/💨/🚨/✅）；What-if 加 session_state 歷史 sparkline + ▲▼ delta
- **Phase C** 頂部 KPI 換 `metric_with_bar` + 加 action bar 4 顆連結（FastAPI docs / GitHub / Model Card / UCI）
- **Phase D** 全新「首頁總覽」dashboard 頁（預設 landing page）

### Round 8 — Shortcode → Unicode emoji
- 症狀：HTML 內塞的 `:dart:` `:books:` 等 shortcode 沒被翻譯，以原樣文字顯示
- 修法：全部換成 Unicode emoji 字元（🎯📚📜📁📊💡📥🚀🔍➡️🗺️🎚️🖼️🔧）
- `Grep` 確認 streamlit_app.py 已無任何 `:xxx:` 殘留

### Round 9 — Dashboard 卡片可點擊
- 新 helper `dash_button_tile`：`stylable_container` 包 `st.button`，CSS 改造成卡片外觀
- 點擊 → `session_state.nav_jump = idx` + `st.rerun()`
- `option_menu` 加 `manual_select=_nav_manual`（pop 後傳入）
- sidebar nav 與 dashboard tiles 互通

---

## 3. 新增 / 修改的檔案

| 檔案 | 角色 |
|---|---|
| `.streamlit/config.toml` | 全域 theme |
| `src/ui/__init__.py` | 包初始化 |
| `src/ui/style.py` | 約 600 行：CSS + 15+ HTML widget helper |
| `src/ui/charts.py` | 12 個 Plotly chart builder |
| `app/streamlit_app.py` | 約 900 行：6 頁面、雙語 hero、tabs、sidebar、dashboard |
| `requirements.txt` | +plotly +streamlit-shadcn-ui +streamlit-option-menu |

新元件清單（給之後維護）：
- **Hero / Section**：`hero(chips=...)`、`section`
- **Note / Card**：`note(kind)`、`fallback_metric_card`、`big_stat(tone)`、`kpi_strip`、`metric_with_bar(tone)`、`advice_card(auto-classified)`、`dash_tile`、`dash_button_tile`
- **Zone**：`zone(kind, key)` context manager（mint/sand/sky/blush/stone）
- **Sidebar**：`sidebar_brand`、`sidebar_model_card`、`sidebar_dataset_card`、`sidebar_footer`
- **Action**：`action_bar(links)`
- **Plotly**：`shap_bar`、`failure_type_bar`、`one_d_sweep`、`risk_landscape`、`confusion_heatmap`、`failure_probability_gauge`、`input_radar`、`sparkline`、`risk_donut`、`probability_histogram`、`leaderboard_bar`

---

## 4. 學到 / 踩到的坑

| 坑 | 教訓 |
|---|---|
| `panel_open` / `panel_close` 用兩個 `st.markdown` 注入 `<div>` 開頭/結尾 | Streamlit DOM 元素彼此隔離；要包內容必須用 `stylable_container` |
| Triple-quoted HTML 有 4+ 空白縮排 | Streamlit Markdown parser 當 code block escape；永遠 `textwrap.dedent` |
| Shortcode `:wrench:` 只在 native widget 翻譯 | 注入到原生 HTML 就不會翻譯；統一改用 Unicode emoji 字元 |
| `st.button` 不支援 HTML label / multiline | 想做 fancy tile 必須用 `stylable_container` 包按鈕 + CSS 改外觀 |
| option_menu 的 `manual_select` 是 once-use | 配合 `session_state.pop(key, None)` 才不會卡住 |

---

## 5. 還沒做的事（明天接著做）

### Tier S — 影響範圍小、效果大
1. **修 PDF 報告檔加進 repo**：working tree 有 `docs/AI伺服馬達預測性維護專題報告.pdf`，需要決定要不要 add
2. **GitHub repo description + topics**：之前提供過三方案（網頁 / curl + PAT / gh CLI），目前 repo 還是無描述
3. **確認首頁卡片點擊**：今天最後一個 commit 沒有實機驗證點擊效果，明早第一件事

### Tier A — UI 還能推
4. **Optuna trial 視覺化**：parallel coordinates、parameter importance、trial history scatter
5. **批次上傳結果加 columns chart**：每個故障類型在批次中的分布
6. **2D 風險地景效能優化**：25×25 grid 每次拖滑桿都要 625 次推論，會有延遲；可改為背景訓練好快取
7. **JSON expander 加 curl one-liner 範本**：把目前的 JSON 用程式組成 `curl -X POST ... -d '...'` 整段，按一鍵複製
8. **Lottie 動畫 / Loading skeleton**：等待 SHAP 計算時的 placeholder
9. **3D 風險地景**：Plotly Surface 替代 2D heatmap，可繞著看
10. **Manual page 加「隨機示範」按鈕**：從測試集隨機抓一筆健康 / 風險樣本

### Tier B — 程式碼健康度
11. **streamlit_app.py 拆模組**：900 行已經太胖，可以拆 `app/pages/dashboard.py`、`manual.py`、`whatif.py`、`batch.py`、`eval.py`、`about.py`
12. **加新元件的單元測試**：`charts.py` 與 `style.py` 都還沒覆蓋
13. **`scripts/test_smoke_ui.py`**：以無頭 selenium / playwright 抓首頁截圖，確認沒有 HTML leak 回歸
14. **`.pre-commit-config.yaml`**：ruff / black 自動化

### Tier C — 部署 / 工程
15. **Docker 重新 build 驗證**：Docker Desktop 啟動後跑一次 `docker compose up` 驗證映像可起
16. **GH Actions 第一次跑結果確認**：去 Actions tab 看綠紅
17. **Dependabot 設定**：`.github/dependabot.yml`
18. **PR / Issue templates**：`.github/PULL_REQUEST_TEMPLATE.md`、`.github/ISSUE_TEMPLATE/*.yml`
19. **環境變數設定**：FastAPI port、CORS origin 從 env 讀，不寫死

### Tier D — 模型 / 資料
20. **接入實際感測資料**（如取得時）
21. **改成 RUL 預測**（survival analysis）
22. **MLOps**：定期重訓 cron、特徵漂移監控

---

## 6. 明天建議的開工順序

```
1. 確認首頁卡片點擊有正常切換頁面 ............... 5 分鐘
2. PDF 報告決定加 / 不加 ...................... 1 分鐘
3. GitHub repo description + topics 設定 ..... 5 分鐘
4. Tier A 挑 2-3 項做 ........................ 1-2 小時
5. streamlit_app.py 拆模組（如時間允許）...... 1 小時
```

---

## 7. 今日 commit 鏈

```
02afcc0  Dashboard tiles are now clickable
a97d696  UI: replace all emoji shortcodes with unicode emoji
9a98d8e  UI: dashboard page + advice cards + sparkline + action bar
a930668  UI: visual upgrades for batch upload and model leaderboard
cd6ed2d  Sidebar redesign: option_menu nav + brand header + info cards
85007d7  UI: visual variation pass
7d8ecc0  UI polish: Streamlit theme + Plotly charts + shadcn-ui cards
```

7 個 commit，全部已 push 到 `origin/main`。

---

## 8. 服務狀態（停工前）

| 服務 | URL | 狀態 |
|---|---|---|
| Streamlit | 8501 | ❌ 已關閉 |
| FastAPI | 8000 | ❌ 已關閉 |

明天要重啟：

```bash
cd "C:/Users/alung/Documents/WorkSpace/AIHW/FinalProject"
streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

或開兩個終端各自跑前景。

---

## 9. 重要連結

- **GitHub repo**：<https://github.com/ChenYuHsu413/AIFinalProject>
- **GitHub Actions**：<https://github.com/ChenYuHsu413/AIFinalProject/actions>
- **昨日工作報告**：`outputs/reports/WORK_REPORT_2026-06-20.md`
- **報告大綱**：`outputs/reports/REPORT_OUTLINE.md`
- **模型卡**：`outputs/models/MODEL_CARD.md`

---

明天見 👋
