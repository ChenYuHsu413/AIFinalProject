# 工作報告 — 2026-06-24

> 專案：AI 伺服馬達故障風險預測與預測性維護建議系統
> 主題：**模組 B+ 三軌延伸 E1/E2/E3 → 拆獨立分頁 → 引入模組 C（Paderborn 馬達電流診斷，第四軌）**
> Repo：<https://github.com/ChenYuHsu413/AIFinalProject>
> 線上 Demo：<https://aifinalproject-test.streamlit.app/>

---

## 1. 摘要（TL;DR）

今天把系統從**三軌（A/B/B+）推進成四軌（A/B/B+/C）**，並把 B+ 的三條延伸軌全部落地：

- **模組 B+ 延伸 E1/E2/E3 完成**：E1 跨工況自適應 RUL（救 LOCO）、E2 維護建議決策層、E3 即時串流回放。
- **E3 互動體驗大改**：從「逐幀 Streamlit 重跑（會閃）」改為 **Plotly 瀏覽器端 frames 動畫**（零重跑、不閃、可即時切速、播完停末幀）。
- **E1/E2/E3 拆出獨立「B+ 延伸應用」頁（三 tab）**，多軌跡泛化頁回到核心，解決頁面過長。
- **新增模組 C — Paderborn 馬達電流故障診斷**：補上 A/B/B+ 都缺的**馬達定子電流（MCSA）模態**。頭條實驗「健康+人工故障訓練、真實損傷測試」，**真實資料跑出 baseline macro-F1 1.00 → 人工→真實 0.20、落差 0.80** 的誠實 domain-shift 結論。

今天共 **4 個 commit** 已 push 到 `main`（`318bc79` → `04b4755` → `4ac423f` → `4d2c1c7`）。全套 **41 單元測試**綠、Streamlit **12 頁**、git 追蹤檔 **136**。

---

## 2. 今日工作時間軸

### Round 1 — 遠端狀態檢查
- 確認本地 `main` 與 `origin/main` 同步、工作區乾淨。

### Round 2 — 模組 B+ 延伸 E2（維護建議決策層）
- 純函式 `src/models/maintenance_advice.py`：`(health, rul, past_fpt, …) → {風險(綠/黃/紅), 建議維護時間窗, 理由, 成本對照}`；9 項單元測試覆蓋邊界。
- Dashboard 卡片：**巡檢檢查點滑桿**（佔壽命比例）對 15 顆軸承各畫一張風險卡（解決 run-to-failure 無「真實現在」的問題）。
- config 新增 `maintenance:`（安全裕度、示意成本）。

### Round 3 — 模組 B+ 延伸 E1（跨工況自適應 RUL，頭條）
- `src/models/eval_xjtu_domain_adapt.py`：在同一 LOCO 切分上消融三種**目標未標註**手段——壽命比例、transductive z-score、CORAL。
- 結果：baseline **−1.22** → 最佳 **CORAL −0.92**；壽命比例 **oracle 上界 +0.15**（診斷出退化形狀可泛化、瓶頸在壽命尺度）。6 項單元測試（壽命還原、無標籤洩漏、CORAL 協方差對齊）。
- Dashboard 消融對照表 + oracle 上界診斷 + 誠實聲明。

### Round 4 — 模組 B+ 延伸 E3（即時串流回放）
- 初版：`st.fragment` + `st.rerun` 逐快照重播（重用 `build_health_indicator`/`detect_fpt`/`extrapolate_rul`）。
- 使用者回饋 4x 播放「一直刷新會閃」→ **改寫為 Plotly 原生 frames 動畫**（瀏覽器端播放、零 Streamlit 重跑、不閃爍）；新增 `charts.xjtu_replay_animation`。
- 速度控制反覆打磨：最終為**圖內 0.5x–4x 速度鍵**（`mode=immediate`+`fromcurrent` → 播放中可即時切速、不重置，播完停末幀）；狀態框即時顯示健康度/RUL/風險（重用 E2 邏輯）。

### Round 5 — E1/E2/E3 拆獨立分頁
- 新增「**🚀 B+ 延伸應用**」頁，以 `st.tabs` 分 E1/E2/E3；抽出 `_render_bplus_e1/e2/e3` helper，多軌跡泛化頁回到核心。
- 同步頁數 11→12（後因模組 C 再→12，見下）、README §11、關於頁 KPI。
- commit + push（`04b4755`）；`MODULE_B_PLUS_EXTENSIONS_PLAN.md` 第 6 節逐步補 ✅ 與日期戳（`4ac423f`）。

### Round 6 — 規劃並引入模組 C（Paderborn）
- 確認「下一步」資料集＝**Paderborn**（`DATASET_EVALUATION.md` 路線圖 ★下一步）。
- 進 plan mode，3 個 Explore agent 探索資料/分類/整合管線；AskUserQuestion 定案：**人工→真實泛化實驗、振動+電流 MCSA、放好即可跑+子集 MVP**。
- **步 0–3**：`config.yaml paderborn:`、`.gitignore`、`data/README.md`、`load_paderborn.py`（.mat 解析）、`build_paderborn_dataset.py`（`vib_*`/`cur1_*`/`cur2_*` 時域特徵 + `fault_class`/`damage_origin` 標籤）、`train_paderborn.py`（baseline 分層 CV + 人工→真實）、8 項單元測試、`MODULE_C_PADERBORN_PLAN.md`。
- **步 4–5**：Dashboard 模組 C 頁（`class_confusion_heatmap` + KPI + baseline-vs-真實 + 兩張混淆矩陣 + 誠實聲明）+ 導覽接線；文件四軌化（README §1 四軌 / §11 12 頁 / §19、DATASET_EVALUATION、CLAUDE 紅線、關於頁、MODULE_B_RESULTS 交叉連結）。
- 修正 URL：Paderborn 官方下載頁網址（原誤用德文路徑，已查證更正）。

### Round 7 — 真實資料跑通 + commit
- 使用者放好 Paderborn 原始資料（32 碼 × 80 .mat）；`build_paderborn_dataset` → **440 量測 × 35 欄**；`train_paderborn` → **baseline macro-F1 1.00 → 人工→真實 0.20、落差 0.80**。
- 修到一個真 bug：`logistic_regression`(liblinear) 二元限制 → 多類別跳過、config 改用 RF/SVM/GB。
- 文件回寫真實結果與誠實細節（真實損傷大量被誤判健康；測試集無健康類使 macro-F1 機械性偏低）。
- commit + push 模組 C（`4d2c1c7`，17 files、+1277/−53、真實 artifacts 一併提交）。

### Round 8 — 收尾
- 建立 `docs/PROMPT_LOG_2026-06-24.md` + 本工作報告 + 確認待辦。

---

## 3. 核心結果

### 模組 B+ 延伸 E1：跨工況自適應 RUL（救 LOCO 的 −1.22）

| 手段 | LOCO macro R²(hours) | 性質 |
| --- | --- | --- |
| baseline（無自適應） | −1.22 | 重現既有 |
| 壽命比例（可部署還原） | −0.96 | 零洩漏 |
| 工況感知標準化（transductive z-score） | −1.04 | 零洩漏 |
| **CORAL 協方差對齊**（最佳） | **−0.92** | 零洩漏 |
| 壽命比例 · oracle 上界 | **+0.15** | 含洩漏、僅診斷 |

> 三手段都把 −1.22 往上抬但仍為負；oracle +0.15 證明退化形狀可跨工況泛化，瓶頸在推論期不知壽命尺度。誠實的「部分改善、未解決」。

### 模組 C：Paderborn 人工→真實故障泛化（N15_M07_F10、22 碼、440 量測）

| 評估 | accuracy | macro-F1 |
| --- | --- | --- |
| baseline（健康+人工 · 5 折 CV，RandomForest） | 1.00 | **1.00** |
| 人工→真實泛化（測真實損傷） | 0.24 | **0.20** |
| **落差** | | **0.80** |

> 人工故障上完美、真實損傷上幾乎泛化失敗（約 90/220 真實受損被誤判健康）——人工(EDM/雕刻)訊號≠真實疲勞損傷，與 Paderborn 文獻一致。誠實標註：真實測試集無健康類，三類 macro-F1 含 0 分 healthy 會機械性拉低（僅看 outer/inner 亦約 0.3）。

---

## 4. 今日 commit 鏈

```
4d2c1c7  feat(module-c): 引入 Paderborn 馬達電流故障診斷（第四軌）
4ac423f  docs(b-plus-plan): 第 6 節分階段交付逐步標 ✅ 並對齊最終實作
04b4755  feat(b-plus-ui): E3 改瀏覽器端動畫、E1/E2/E3 拆成獨立分頁
318bc79  feat(module-b-plus): 新增三軌延伸 E1/E2/E3（跨工況自適應 RUL、維護建議、串流回放）
```

4 個 commit，全部已 push 到 `origin/main`（接 yesterday 的 `90e6932`）。

---

## 5. 系統狀態

| 項目 | 狀態 |
|---|---|
| 軌道 | **四軌 A / B / B+ / C** |
| Streamlit 頁面 | **12**（首頁 1 + A 4 + B 3 + B+ 2 + C 1 + 關於 1）|
| 單元測試 | **41 / 41** 綠 |
| git 追蹤檔 | **136** |
| 本機 Streamlit | port 8501 · 背景執行中（模組 C 顯示真實數字）|
| 線上 Demo | push 後自動重新部署 |

模組 C 重跑（需 `data/raw/paderborn/<碼>/`）：
```powershell
.venv\Scripts\python -m src.data.build_paderborn_dataset
.venv\Scripts\python -m src.models.train_paderborn
```

---

## 6. 待辦事項 / 未來工作

### 已完成（本日）
- [x] 模組 B+ 延伸 E1（跨工況自適應 RUL）、E2（維護建議）、E3（串流回放，Plotly 瀏覽器端動畫）
- [x] E1/E2/E3 拆獨立「B+ 延伸應用」頁（tabs）
- [x] 模組 C（Paderborn）MVP：資料/特徵/分類/人工→真實泛化 + Dashboard 頁 + 四軌文件 + 真實資料跑通

### 模組 C 延伸（下一步候選）
- [ ] **MCSA 頻譜邊帶特徵**（真正的電流診斷，目前僅時域特徵）。
- [ ] **納入全 4 工況**（目前只 N15_M07_F10）。
- [ ] **領域自適應救人工→真實**（落差 0.80 太大；可比照 E1 用 CORAL / 或納入少量真實樣本微調）。
- [ ] 模組 C **即時預測 FastAPI 端點**（目前僅 Dashboard 讀 artifacts）。

### 全專案
- [ ] 小清理：`SVC(probability=True)` 在 sklearn 1.9 被 deprecated（共用 registry，影響 A 與 C）；可換 `CalibratedClassifierCV`。
- [ ] （推遲）1D-CNN Autoencoder 深度對照（`MODULE_B_DL_PLAN.md`）。
- [ ] （推遲）其餘公開資料集（FEMTO / Mendeley / PMSM）、ESP32 邊緣 IoT 實場接入、MLOps、規則式建議升級 LLM 敘述。

---

## 7. 重要連結

- **線上 Demo**：<https://aifinalproject-test.streamlit.app/>
- **GitHub repo**：<https://github.com/ChenYuHsu413/AIFinalProject>
- **模組 C 規格與結果**：`docs/MODULE_C_PADERBORN_PLAN.md`
- **模組 B+ 延伸規劃**：`docs/MODULE_B_PLUS_EXTENSIONS_PLAN.md`
- **模組 B / B+ 成果**：`docs/MODULE_B_RESULTS.md`
- **外部資料集評估**：`docs/DATASET_EVALUATION.md`
- **昨日工作報告**：`outputs/reports/WORK_REPORT_2026-06-23.md`
- **今日 prompt log**：`docs/PROMPT_LOG_2026-06-24.md`

---

今天先這樣 👋
