# 工作報告 — 2026-06-27

> **狀態（2026-06-27）**：本日為大改動日，主線完成**真實 PHM 資料導入重訓**、**Next.js 前端
> 互動化**、**資料溯源證明**、**部署修正**與一輪**完整 code review + 清理**；另補**真實資料導入後的
> 過時文字收尾 + 報表頁擴充**（設備別比較 / 告警統計，見 §7）與**深度學習 Phase A：真 PyTorch
> MLP + 神經 autoencoder**（取代 sklearn/PCA 替身，見 §8）、**Phase B：真 1D-CNN on 原始波形**
> （能量包絡，留出 macro-F1 0.729，見 §9）。承接
> [`WORK_REPORT_2026-06-26.md`](WORK_REPORT_2026-06-26.md)。相關文件：
> [`../../docs/MODULE_SERVO_PLAN.md`](../../docs/MODULE_SERVO_PLAN.md)、
> [`../../docs/DATA_PROVENANCE.md`](../../docs/DATA_PROVENANCE.md)、
> [`../../docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md)。

## 1. 真實 PHM FMCRD 資料導入並重訓（主線里程碑）

先前 Servo 以 placeholder 合成資料運作；本日導入**完整真實 PHM FMCRD 資料集**並重訓，
`config.yaml::servo.placeholder` 設為 `false`。

- **資料**：`FMCRD_Data.zip`，8 個 CSV、合計 **106,661,902,496 bytes（106.66 GB）**、~4.4 億列
  時間點；欄位與 `servo_features.RAW_COLUMNS` 完全吻合、`ylabel` 已是 LN/LO/MED/HI。
- **串流聚合 builder**（`src/data/build_servo_from_zip.py`）：從 zip 分塊串流、**不解壓不爆記憶體**、
  線上累計 mean/std/min/max/rms + 三相 current_rms；每檔 checkpoint 可續跑。線上統計與既有
  `aggregate_run`（兩遍法）比對**最大相對誤差 8.8e-13**（等同浮點精度）。
  途中修掉 `train_noisy_LO` 的 `i_3p_c` 少量非數值雜訊（`pd.to_numeric(errors="coerce")`）。
- **DV 正規化**：真實 DV 為物理單位（max≈5012）→ 正規化 0..1；依真實分布**重校
  `servo.dv_risk` = 0.20 / 0.48**（類別均值中點）。
- **split-aware 訓練**（`train_servo.py` / `servo_dl.py`）：特徵表帶 `split` 欄時，train_* 訓練、
  test_* 留出評估（無 split 欄則維持 CV，向後相容 placeholder）。
- **留出測試結果**：分類 logistic_regression macro-F1 **0.757**（n=800）、DV 回歸 RandomForest
  **R²=0.937 / MAE=0.047**；DL 離線 MLP macro-F1 0.711、PCA 重建誤差隨退化單調上升。
- **如實揭露**：`train_noisy_LO` 原始檔僅含 65 段（下載偏少）→ train LO 偏少；test 各類 200 完整。
  FMCRD 為高擬真**模擬**資料集（非真實工廠遙測）。

## 2. 資料溯源證明（可獨立重驗）

- `src/data/servo_data_provenance.py`：讀 zip 中央目錄產生**指紋**（8 檔 CRC32 + 總 bytes，
  任何持同一份檔者可秒驗）+ 聚合統計 + 留出指標 → `outputs/metrics/servo_data_provenance.json`；
  並產 `outputs/figures/servo_provenance.png`（DV 各類分布 + 留出混淆矩陣）。
- 後端 `GET /servo/provenance`；Next.js 儀表板加綠色「訓練於真實 PHM FMCRD」面板（可展開溯源圖）。
- 文件 `docs/DATA_PROVENANCE.md`：指紋表 + 處理管線 + 結果 + 重現步驟 + 誠實性聲明。

## 3. Next.js 前端互動化（Vercel Command Center）

把「會動但前端沒接 / 死頁」補活，皆由已提交產物驅動、雲端可運作：

- **B+ 延伸應用頁**（原幾乎空白）：E2 維護建議（巡檢滑桿 → `/maintenance/advice` +
  `/xjtu/rul_predictions`）、E3 串流回放（`/xjtu/replay`）、LOBO/LOCO 泛化表（`/xjtu/lobo_loco`）。
- **模組 A 評估頁**：互動門檻調節器（`/metrics/test_predictions` 即時重算混淆矩陣 / P / R / F1）。
- **模組 C 診斷頁**：即時推論——新增 `src/models/predict_paderborn.py` +
  `GET /paderborn/samples`、`POST /paderborn/predict`（模型 `paderborn_clf.joblib` 納入 git/HF 白名單）；
  選真實損傷量測可現場看到誤判（泛化落差的單筆呈現）。

## 4. 跨頁帶資料 / UX 修正

- 健康儀表板選樣本後「前往 LLM 助理」**帶入該筆**（`?sample=` URL 參數；Streamlit 端亦修
  session_state 即時重算）。掃描其餘跨頁連動：設備詳情頁同類 bug 一併修，餘者正確。
- 總覽誠實標示：趨勢圖標「示意資料」、KPI 移除捏造的變化量箭頭與文案（數值本身為真）。

## 5. 部署修正（HF build context）

- `.dockerignore` 補白名單：`paderborn_features.parquet`、`xjtu_features.parquet`、
  `ims_set2_features.parquet`、`outputs/figures`——否則雖 git-tracked 仍被排除在 HF Docker
  build 外，導致 C 即時推論 / B+ 全頁 / IMS 互動 / 溯源圖在雲端缺資料。
- 同步 `deploy/huggingface/README.md`、`docs/DEPLOYMENT.md §9.1`。

## 6. Code review + 架構檢查 + 清理

四路並行審查（死檔 / Python / Next.js / 架構）後修正：

- **資料穩健性**：串流聚合除以 0 防護、DV 分母守護、空 zip 報錯；`servo_dl` PCA 改 train-only
  擬合（消除測試洩漏）；`R²`→`R2` print（cp950 console 安全）。
- **Next.js**：equipment 錯誤卡死、module-c 舊判定閃錯 / 載入態、assistant `?sample` 空字串、
  溯源圖 `onError`。
- **清死檔**：`web/src/components/stub-page.tsx`（0 引用）、舊 Streamlit QA 截圖 6 張、
  `scripts/verify_dashboard_clicks.py`（半失效）、`docs/conversation-...txt` 一度刪後**依需求還原**
  （供他人參考所用 prompt）。`nav.ts` 註解更正（Next 為 Streamlit 超集）。
- **補關鍵測試**：`test_build_servo_from_zip`、`test_train_servo_holdout`（留出）、後端
  `/servo/provenance`、`/servo/predict`、`/paderborn/predict` HTTP。**全測試 119 passed / 1 skipped**。
- 審查誤判經查證更正：`paderborn.cv_folds` 實際有讀（未刪）。

## 驗證

- Python：`pytest` **119 passed, 1 skipped**。
- Next.js：`tsc` + `eslint` + `next build`（22 路由）全過。
- 端點契約：FastAPI TestClient 逐一比對回傳形狀與前端 TypeScript interface 吻合。

## 7. 真實資料導入後的過時文字收尾 + 報表頁擴充（同日後補）

切到真實 PHM FMCRD（`placeholder=false`）那次大改，留下一批**寫死的「仍是 placeholder 合成」文字**未同步——
盤點全專案後修正（條件式 `placeholder` 警告會自動失效，故不動；只改寫死處）：

- **前端過時文字**：`about/page.tsx`（三軌定位 Servo 資料列 + 第一條誠實性聲明）、`streamlit_app.py` about 頁
  總覽段——皆由「placeholder 合成 / 待真實 PHM 替換」改為「已導入真實 PHM FMCRD（高擬真**模擬**、非工廠遙測）」。
- **文件同步**：`FINAL_REPORT.md`（§1 前提、§4 資料表、§9 成果表指標 0.748/0.733→**留出 0.757 / R² 0.937 / MAE 0.047**、
  §12 結果彙整、§13 誠實聲明 + 頂部狀態戳）、`MODULE_SERVO_PLAN.md` §1、`DEMO_SCRIPT.md` 開場白 + Q&A、
  `data/README.md` 狀態戳、`WEB_REVAMP_PLAN.md` 誠實性紅線。歷史工作報告（06-25 等）不動。
- **報表頁擴充**（原 placeholder 提示承諾「接真實資料後擴充」，前置條件已達成）：`web/.../reports/page.tsx`
  新增**設備別比較**（`/servo/fleet`，recharts 長條 + DV / 風險）與**告警統計**（`/servo/alerts`，嚴重度
  KPI + 類型分布），皆標示真實模型來源 / mock fallback。**時間區間彙整**如實留待逐時遙測串流（仍為示意 mock）。
- 驗證：前端 `tsc` + `eslint` + `next build`（22 路由）全過；`streamlit_app.py` py_compile 過。

## 8. 深度學習 Phase A — 真 PyTorch（取代 sklearn MLP / PCA 替身）（同日後補）

原 `servo_dl.py` 的「深度學習」實為 sklearn MLP + **PCA 假裝 autoencoder**，無任何 DL 框架。Phase A 改為真 PyTorch：

- **`src/models/servo_dl.py` 改寫**：torch MLP 分類/回歸（隱藏層 64→32）+ **神經 autoencoder**（`7→4→2→4→7`，
  健康 LN-train 擬合、各類重建誤差）取代 PCA。固定種子、CPU、全批 Adam → **可重現**（重跑數字一致）。
  JSON key 不變（向後相容）；新增 `framework` / `architecture`；`method`→`servo_dl_torch`、`note` 改實。
- **留出測試結果**：MLP macro-F1 **0.714**（sklearn 版 0.711，幾乎一致）、DV 回歸 R² **0.959** / MAE 0.039；
  AE 重建誤差 **LN 0.33 < LO 0.37 < MED 0.69 < HI 2.15**（隨退化單調上升）。
- **torch 拆成獨立 `requirements-dl.txt`**（離線訓練專用）：因 HF/root Dockerfile 與 CI 皆裝 `requirements-dev.txt`，
  若把 torch 放 dev 會被拉進雲端映像（~700MB）。故 dev 不含 torch、雲端映像維持精簡；CI 改裝 `requirements-dl.txt` 跑 DL 測試。
- 前端：`simulator` 頁「PCA 重建誤差」標籤→「神經 autoencoder 重建誤差」；`note` 自 JSON 自動更新（其餘零改）。
- 新增 `tests/test_servo_dl.py`（smoke：JSON 形狀 + 重建誤差單調）。**全測試 120 passed / 1 skipped**；前端 tsc/eslint 過。
- docs 同步：`FINAL_REPORT §9`+未來工作、`MODULE_SERVO_PLAN` 狀態戳/§7/下一步。
- **發現**：原始 FMCRD zip 仍在 `Downloads/FMCRD_Data.zip`（22 GB 壓縮 / 106.66 GB 解壓、8 檔、CRC 對齊溯源指紋）
  → **Phase B（原始時序 1D-CNN）解鎖**，待開窗 builder。

## 9. 深度學習 Phase B — 真 1D-CNN on 原始波形（同日後補）

原始 FMCRD 找回後，補上真正的 1D-CNN（不再只是聚合特徵上的 MLP/AE）：

- **開窗 builder**（`src/data/build_servo_windows.py`）：從原始 zip **串流**每段 run，降為原始波形**能量包絡**
  （逐塊 std，8 通道 torque/rotor_speed/三相電流/direct/quadrature/position_error × 256 時間塊）。**設計轉折**：
  先試「每短窗（1024 點）分類」效果差（train 0.37 / test 0.27；連 logistic 吃手工窗統計上限也只 0.36）——
  退化訊號在長時統計、單短窗太雜訊 + 跨檔域偏移。改為**每段 run 一個能量包絡**後大幅改善。
- **1D-CNN**（`src/models/servo_cnn.py`，PyTorch）：Conv1d[8→16→32→64]+BN+GAP+FC 分類 + 1D conv-AE；
  split 依來源檔分離（無洩漏）、固定種子 → 可重現。留出 **Accuracy 0.731 / macro-F1 0.729**
  （與聚合模型 0.757 相當）；conv-AE 重建誤差 **LN 0.40 < LO 0.41 < MED 0.45 < HI 0.51** 單調。
  混淆矩陣顯示誤差集中在 LN↔LO（早期退化難分），MED/HI 幾乎全對——合理可解釋。
- **整合**：後端 `GET /servo/cnn_results`（`services.servo_cnn_results`）；報表頁新增「1D-CNN（原始波形）」卡
  （準確率 / macro-F1 / 輸入規格 + conv-AE 重建誤差條）；原 DL 卡標題正名（「1D-CNN AE / MLP」→「PyTorch MLP + 神經 AE」）。
- **資料/部署**：windowed `.npz` gitignored（暫存、雲端只讀 JSON）；`servo_cnn_results.json` 白名單、隨 `outputs/metrics` 上雲；
  CNN 雲端只讀 JSON、**不需 torch**。config 加 `windows_path` / `cnn_metrics`。
- 新增 `tests/test_servo_cnn.py`（forward 形狀 + run() 在有 npz 時驗證）、後端 `/servo/cnn_results` 測試。
  **全測試 123 passed / 1 skipped**；前端 tsc/eslint/build 全過。
- docs 同步：`FINAL_REPORT §9`（新增 1D-CNN 列）+未來工作、`MODULE_SERVO_PLAN` 狀態戳/§7/下一步。

## 待辦 / 後續

- ~~HF 後端重新部署（推最新 main；新端點 + 三組 parquet + 溯源圖已就緒，設定不需改）。~~
  **✅ 完成（2026-06-27）**：Space `icefeather/aifinalproject` 同步至 main `f083b51`（Space commit `27b07c3`）；
  驗證 `/health` ok、`/servo/provenance` n_files 8 / placeholder=false / clf 0.7566、`/figures/servo_provenance.png` 200、
  `/paderborn/samples` 15、`/xjtu/lobo_loco` 完整。
- ~~Phase B：原始時序 1D-CNN。~~ **✅ 完成（2026-06-27，見 §9）**：留出 macro-F1 0.729、後端 `/servo/cnn_results`、報表頁顯示。
- 後續可做：CNN 深化（更大窗 / 更多 run / 原始逐點 / 頻譜輸入）、Paderborn MCSA 頻譜邊帶特徵。
- 報表頁**時間區間彙整**：待實場 / IoT 逐時遙測串流接入後補上。
