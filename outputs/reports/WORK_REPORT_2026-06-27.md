# 工作報告 — 2026-06-27

> **狀態（2026-06-27）**：本日為大改動日，主線完成**真實 PHM 資料導入重訓**、**Next.js 前端
> 互動化**、**資料溯源證明**、**部署修正**與一輪**完整 code review + 清理**。承接
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

## 待辦 / 後續

- HF 後端重新部署（推最新 main；新端點 + 三組 parquet + 溯源圖已就緒，設定不需改）。
- 後續可做：真正的 1D-CNN / Autoencoder（離線 torch、真實時序）、Paderborn MCSA 頻譜邊帶特徵。
