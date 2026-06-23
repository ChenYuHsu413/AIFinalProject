# 工作報告 — 2026-06-23

> 專案：AI 伺服馬達故障風險預測與預測性維護建議系統
> 主題：**模組 B+ 多軌跡 / 多工況泛化（XJTU）＋ 全專案三軌文件整合**
> Repo：<https://github.com/ChenYuHsu413/AIFinalProject>
> 線上 Demo：<https://aifinalproject-test.streamlit.app/>

---

## 1. 摘要（TL;DR）

今天把系統從「靜態(A) + 動態(B)」**雙模組**推進成**三軌 A / B / B+**：新增 **模組 B+ — 以
XJTU-SY 軸承資料（15 顆 × 3 工況）做多軌跡、多工況泛化驗證**，補上 IMS 單軌跡缺乏的跨設備
泛化證據。核心發現很誠實也很有深度：**健康監測（FPT / 退化起點偵測）能跨軸承、跨工況泛化；
但監督式「絕對小時數 RUL」受壽命尺度與工況 domain shift 限制**（各工況壽命差達約 20 倍）。

同時完成大量**文件治理與報告整合**：專案規則檔（CLAUDE.md）、外部資料集評估、報告大綱統一
A/B/B+、README 全文補齊三軌、Dashboard「關於本專案」升級、未來工作排序定為 **Paderborn 下一步、
其餘推遲**。

今天共 **9 個 commit** 已 push 到 `main`（`0457126` → … → `a6c640a`）；另有 README 全文更新、
prompt log 與本工作報告待 commit。

---

## 2. 今日工作時間軸

### Round 1 — 組員 AI 建議的顧問式分析（純討論，不動程式）
- 逐條評估 5 點 AI 建議：指出建議 1「多指標投票可更早抓 FPT」與**團隊自己量到的數據矛盾**
  （本軸承突發失效，四指標同時暴衝）；建議 3（1D-CNN AE）本就在路線圖；建議 5（A/B 風險聯動）
  **牴觸既有設計決策**（兩模組刻意獨立）。
- 進一步做完整資料策略顧問分析（資料缺口、6 個公開資料集評估、架構方案、最小可行實作），
  結論：**最大缺口是 IMS 單軌跡無泛化證據；最推薦補 XJTU-SY**。

### Round 2 — 文件治理：日期戳 + 清理 + 專案規則
- 三份 `MODULE_B_*` 文件補 **2026-06-23 狀態戳**；清掉 `MODULE_B_IMS_PLAN.md` 的 `Data.mat`
  殘留提及（該檔本就不存在）。
- 新增 `docs/DATASET_EVALUATION.md`（六資料集評估）與 `docs/MODULE_B_PLUS_XJTU_PLAN.md`（XJTU 規格）。
- 新增**專案 `CLAUDE.md`**：①文檔改動一律附日期戳 ②大型更新回頭翻 `docs/` 同步 ③誠實性紅線。
- `MODULE_B_DL_PLAN.md` 標記「**已推遲**」（先補資料集，DL 對照暫緩）。

### Round 3 — 模組 B+ 起步：XJTU loader + 下載
- WebSearch / WebFetch 確認 XJTU-SY 下載來源與格式（25.6 kHz、32768×2、3 工況 × 5 顆）。
- 寫 `src/data/load_xjtu.py`（仿 `load_ims.py`，**自動偵測並略過 CSV 表頭**）；`.gitignore` 排除
  `data/raw/xjtu/`。
- 釐清資料是 **6-part WinRAR 分卷**需全下載；使用者下載後，loader 驗證讀檔得 `[32768, 2]`
  （Bearing1_1 = 123 快照）✅。

### Round 4 — 模組 B+ 步驟 1–3（先單工況 C1）
- `build_xjtu_dataset.py`（**重用** `vibration_features`，水平通道整套時域特徵 + 垂直 RMS）。
- `eval_xjtu_generalization.py`（**重用** `rul_extrapolation` 的 HI/FPT/外推）：C1 五顆以同一組固定
  參數**全數偵測退化**，平均提前 **1.45 h**、退化區 MAE **0.57 h**。

### Round 5 — 步驟 4 LOBO 監督式 + 步驟 6 報告段落
- `train_rul_lobo.py`：留一軸承監督式 RUL。C1 瞬時特徵 pooled R²≈−0.11，**相似壽命軸承
  （1_1 / 1_3）R² +0.43 / +0.59 → 外推牆消失**（對比 IMS 單軌跡的 −76）。
- 評估「軸承內滾動趨勢特徵」（config 開關 `lobo_use_trend`）：未改善 pooled R²，**預設關閉**。
- 結果寫入 `MODULE_B_RESULTS.md`。

### Round 6 — 步驟 5 Dashboard 分頁 + HEROES KeyError 修復
- 新增「**模組 B+ · 多軌跡泛化**」導覽群組與頁面（KPI、彙總表、健康指標疊圖、LOBO 表），
  純讀已 commit 的 CSV/JSON/parquet → **雲端可顯示、無 torch 依賴**。
- 雲端報 `KeyError: HEROES[page]`（新頁漏加頁首標題）→ 補上修復（`b6aeaef`）；用 AppTest 把
  **全 10 頁**跑過，**0 例外**。

### Round 7 — 跨工況延伸 C2 / C3（15 顆 × 3 工況）
- 複製 C2（37.5Hz11kN）/ C3（40Hz10kN）進專案；`config.yaml` 改 `conditions` 清單；建全 15 顆
  → `xjtu_features.parquet`（9216 × 16）。
- **健康監測 FPT 全 15 顆偵測退化**：C1 1.45 h / C2 2.65 h / C3 17.43 h → 跨軸承＋跨工況泛化成立。
- LOBO 改為**工況內留一**（pooled R²≈−0.62）＋新增 `train_rul_loco.py`**留一工況**（pooled R²≈−1.22）
  → 誠實發現：**絕對 RUL 跨壽命尺度 / 工況失效（domain shift）**，各工況壽命差約 20 倍。
- Dashboard B+ 分頁改跨工況視圖（健康疊圖以工況上色 + LOBO / LOCO 對照）。

### Round 8 — 報告整合 + 全站三軌一致化
- `REPORT_OUTLINE.md` 由 Module A only 擴成統一 **A→B→B+** 大綱；修正與 B/B+ 矛盾的
  「不做 RUL / 無 run-to-failure」舊陳述。
- 未來工作排序：**Paderborn 下一步、其餘推遲**（REPORT_OUTLINE / DATASET_EVALUATION / README 三處一致）。
- Dashboard「關於本專案」頁升級 A/B/B+ 三軌（卡片 + 對照表 + 頁數 9）。
- README 架構區雙模組 → 三軌；並**全文補齊 B/B+**（頂部定位、§3 資料集、§4 檔案樹、§6 下載、
  §11 Streamlit 9 頁分組）。

### Round 9 — 收尾
- 建立 `docs/PROMPT_LOG_2026-06-23.md`（本 session 25 則 prompt）。
- 撰寫本工作報告。

---

## 3. 模組 B+ 核心結果（XJTU，15 顆 × 3 工況）

### 健康監測（FPT）跨工況泛化 ✅

| 工況 | 轉速 / 負載 | 軸承數 | 平均退化提前量 | 平均退化區 MAE |
| --- | --- | --- | --- | --- |
| C1 | 2100 rpm / 12 kN | 5 | 1.45 h | 0.57 h |
| C2 | 2250 rpm / 11 kN | 5 | 2.65 h | 1.04 h |
| C3 | 2400 rpm / 10 kN | 5 | 17.43 h | 6.82 h |
| **全部** | — | **15** | **7.18 h** | **2.81 h** |

> 同一組固定參數（未逐顆、未逐工況調）在 15 條獨立軌跡上全數偵測到退化起點。C3 絕對 MAE 較大
> 是因壽命尺度（最長約 42 h）非方法失效。

### 監督式絕對 RUL：跨壽命尺度的限制 ⚠️

| 驗證設計 | 說明 | 合併 R² | 合併 MAE |
| --- | --- | --- | --- |
| LOBO（工況內留一軸承） | 同工況其他軸承可入訓練 | −0.62 | 11.2 h |
| LOCO（留一整個工況） | 測試工況轉速/負載完全未見 | −1.22 | 14.2 h |

> 僅壽命相近的軸承（C1 的 1_1 / 1_3）達 R² +0.4~0.6；壽命尺度差異拉大（工況內 C2/C3、或跨工況）
> 即崩潰。結論：**能穩健泛化的是健康監測，不是絕對小時數 RUL 回歸**——後者需壽命正規化或領域
> 自適應，列為未來工作。

---

## 4. 今日 commit 鏈

```
a6c640a  docs(readme): 架構區由雙模組升級為三軌 A/B/B+
3aa6bdb  feat(dashboard): 「關於本專案」頁升級為 A/B/B+ 三軌對照
2f6f32f  docs: 報告整合 A/B/B+ 故事線；未來工作改 Paderborn 為下一步、其餘推遲
a9e982e  feat(module-b+): 跨工況泛化 C2/C3（15 顆 × 3 工況）+ LOCO 監督式對照
b6aeaef  fix(dashboard): 補 HEROES 缺少的「多軌跡泛化」頁標題，修正 B+ 分頁 KeyError
f775557  feat(module-b+): Dashboard 多軌跡泛化分頁（步驟 5）
6b1f420  feat(module-b+): LOBO 監督式 RUL 對照 + 報告段落（步驟 4 + 6）
22646db  feat(module-b+): XJTU-SY 多軌跡 RUL 泛化驗證（步驟 1–3）
0457126  docs+feat: 外部資料集評估 + XJTU Module B+ 起步；docs 補狀態戳；新增專案 CLAUDE.md 規則
```

9 個 commit，全部已 push 到 `origin/main`。
**待 commit**：README 全文三軌補齊、`docs/PROMPT_LOG_2026-06-23.md`、本工作報告。

---

## 5. 服務狀態（停工前）

| 服務 | 位置 | 狀態 |
|---|---|---|
| 線上 Demo | <https://aifinalproject-test.streamlit.app/> | ✅ 已部署（push 後自動重新部署）|
| 本機 Streamlit | port 8501 | ❌ 已關閉 |

本機重跑 Module B+ 管線（需 `data/raw/xjtu/` 三工況資料）：

```powershell
.venv\Scripts\python -m src.data.build_xjtu_dataset
.venv\Scripts\python -m src.models.eval_xjtu_generalization
.venv\Scripts\python -m src.models.train_rul_lobo
.venv\Scripts\python -m src.models.train_rul_loco
```

本機啟動 Dashboard：

```powershell
.venv\Scripts\streamlit run app\streamlit_app.py
```

---

## 6. 重要連結

- **線上 Demo**：<https://aifinalproject-test.streamlit.app/>
- **GitHub repo**：<https://github.com/ChenYuHsu413/AIFinalProject>
- **模組 B+ 規格**：`docs/MODULE_B_PLUS_XJTU_PLAN.md`
- **模組 B / B+ 成果**：`docs/MODULE_B_RESULTS.md`
- **外部資料集評估**：`docs/DATASET_EVALUATION.md`
- **報告大綱（A/B/B+）**：`outputs/reports/REPORT_OUTLINE.md`
- **昨日工作報告**：`outputs/reports/WORK_REPORT_2026-06-22.md`
- **今日 prompt log**：`docs/PROMPT_LOG_2026-06-23.md`

---

今天先這樣 👋
