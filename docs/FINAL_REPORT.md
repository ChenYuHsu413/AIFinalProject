# AI 伺服馬達健康監控與智慧維護指揮中心 — 專題報告

> **狀態（2026-06-28）**：完整書面報告。涵蓋主線 **模組 Servo** 與對照軌 **A / B / B+ / C**、
> 系統實作（FastAPI 後端 + Next.js Command Center 前端 + LLM 維護助理 + 知識庫）與部署
> （Vercel + Hugging Face Spaces）。數字取自 `outputs/metrics/` 已提交之評估檔；主線 Servo 已同步為
> **真實 PHM FMCRD 留出測試**結果（非先前 placeholder 合成）。§9.1 補上**兩階段深度學習完整說明**
> （Phase A PyTorch MLP+AE、Phase B 真 1D-CNN on 原始波形，含深化後的誠實下修）。§8 模組 C 補上
> **CE1 領域自適應**（無監督對齊無效、few-shot 每類 10 筆真實標籤把 macro-F1 0.20→0.54）。
> 線上 Demo：Next.js Command Center（Vercel）+ 後端（HF Spaces）。
> 相關文件：[`REPORT_OUTLINE.md`](REPORT_OUTLINE.md)、[`MODULE_SERVO_PLAN.md`](MODULE_SERVO_PLAN.md)、
> [`MODULE_B_RESULTS.md`](MODULE_B_RESULTS.md)、[`MODULE_C_PADERBORN_PLAN.md`](MODULE_C_PADERBORN_PLAN.md)、
> [`DATASET_EVALUATION.md`](DATASET_EVALUATION.md)、[`DEPLOYMENT.md`](DEPLOYMENT.md)。

---

## 1. 摘要

本專題建立一套**伺服馬達預測性維護原型系統**，定位為「維護決策輔助」——產出故障風險、
健康分數與人類可讀的維護建議，**不對馬達下達任何控制命令**。系統以 **CRISP-DM** 流程開發，
由一條主線與四條對照軌組成：

- **主線 · 模組 Servo**：伺服馬達健康狀態分類（LN/LO/MED/HI）+ 退化值（DV）回歸。
- **模組 A**：UCI AI4I 2020 靜態製程風險分類。
- **模組 B**：NASA/IMS 軸承動態健康度與 RUL（趨勢外推）。
- **模組 B+**：XJTU-SY 多軸承 / 多工況泛化驗證。
- **模組 C**：Paderborn 馬達定子電流（MCSA）故障診斷，驗證「人工故障訓練 → 真實損傷測試」的泛化落差。

主線敘事為：**從靜態風險評估，延伸到動態健康監測，再以多軌跡資料誠實驗證泛化邊界，
並以馬達電流補上電氣模態**。系統另含 **AI 訓練模擬器**、**LLM 多供應商維護助理**、
**TF-IDF 維修知識庫**，並以 **Next.js Command Center** 前端整合為工業監控介面。

> **誠實性前提**：主線 Servo 已以**完整真實 PHM FMCRD 資料集（106.66 GB）** 重訓（`config.yaml::servo.placeholder=false`）；
> FMCRD 為**高擬真模擬**資料集（非真實工廠遙測）。AI4I 為合成資料；IMS 為單軌跡；Paderborn 為真實 PMSM
> 試驗台（非產線伺服馬達）。各結果均如實標示。

---

## 2. 專案背景與動機

工業伺服馬達長期運轉會累積與溫度、扭矩、機械負載、刀具/軸承磨耗相關的劣化。若能在故障前
預估風險，運維單位即可降低非計畫停機、減少緊急維修成本、避免製程報廢。本系統支援的決策為
**派工、巡檢、班次調度**；範圍外為即時控制迴路與安全互鎖。

各模組分工互補：模組 A 聚焦**靜態故障風險分類**（AI4I 無時間維度）；模組 B / B+ 補上
**動態健康度與 RUL**（run-to-failure 時序）；模組 C 補上**電流模態**並檢驗泛化；主線 Servo
則把上述方法落到伺服馬達情境的**健康狀態 + 退化值**雙輸出。

---

## 3. CRISP-DM 流程對應

| 階段 | 對應實作 |
| --- | --- |
| Business Understanding | 本報告 §1–§2、誠實性聲明 §14 |
| Data Understanding | `src/data/`、`scripts/run_eda.py`、`notebooks/01_eda.ipynb` |
| Data Preparation | `src/data/preprocess.py`、`src/features/` |
| Modeling | `src/models/`（A/B/B+/C/Servo 各自的 train/eval） |
| Evaluation | `outputs/metrics/`、`outputs/figures/` |
| Deployment / Application | `app/backend/`（FastAPI）、`web/`（Next.js）、`deploy/`、`docs/DEPLOYMENT.md` |

---

## 4. 資料集說明與限制

| 模組 | 資料集 | 型態 | 真實性 | 目標 |
| --- | --- | --- | --- | --- |
| Servo | PHM **FMCRD** 伺服馬達退化（106.66 GB） | 多通道時域聚合特徵 | 真實 PHM 資料集（高擬真**模擬**，非工廠遙測） | 健康狀態分類 + DV 回歸 |
| A | UCI **AI4I 2020** | 單筆製程點 | 合成 | 故障二元分類 |
| B | NASA/**IMS** Set 2 | 20 kHz 振動 run-to-failure | 實測（單軌跡） | 健康度 / RUL |
| B+ | **XJTU-SY** | 25.6 kHz 振動，15 顆 / 3 工況 | 實測（多軌跡） | 跨軸承 / 工況泛化 |
| C | **Paderborn** | 64 kHz 電流 + 振動 | 實測（試驗台，真實+人工損傷） | 故障分類（健康/外環/內環） |

**關鍵限制（詳見 §14 誠實性聲明）**：AI4I 為合成；Servo FMCRD 為高擬真**模擬**（非工廠遙測）；IMS 為單軌跡不可泛化；
XJTU 為軸承試驗台（非伺服馬達本體，以「旋轉機械 / 電動機」為方法範疇）；Paderborn 為 PMSM
**試驗台**且含人工與真實兩類損傷。原始大型資料集（IMS/XJTU/Paderborn raw，約 21 GB）不進 git。

---

## 5. 模組 A · 靜態風險分類（AI4I 2020）

**資料**：10,000 筆，故障比例約 3.4%（嚴重不平衡）。特徵：`Type`、氣溫、製程溫度、轉速、扭矩、刀具磨耗。

**資料準備**：移除識別碼（`UDI`/`Product ID`）與會洩漏目標的故障類型標籤（`TWF/HDF/PWF/OSF/RNF`）；
`Type` One-Hot、數值 StandardScaler（僅 scale-sensitive 模型，且僅在訓練折 fit）；stratified 80/20 切分。

**特徵工程**（節錄）：`temp_diff`（熱負載）、`power_proxy=扭矩×轉速`、`wear_torque_interaction`、
`temp_wear_interaction` 等交互特徵。

**建模**：10 模型 × 5 特徵組（原始 / +工程 / SelectKBest / RFE / RF-importance）。

**評估**（以 F1 選模；不平衡資料下不可只看 Accuracy）：最佳為 **Gradient Boosting + RFE top-8（`D_rfe_top8`）**。

| 指標 | 值（故障類別 / 正類） |
| --- | --- |
| Accuracy | 0.993 |
| Precision | 0.982 |
| Recall | 0.809 |
| F1 | 0.887 |
| ROC-AUC | 0.969 |
| PR-AUC | 0.906 |

混淆矩陣（測試 2000 筆，68 筆故障）：`[[1931, 1], [13, 55]]` — 漏報 13 / 誤報 1。預測性維護偏重
**Recall**（漏報成本通常大於誤報），此處 Recall 0.81、PR-AUC 0.91 在 3% 正類下屬實用水準。
第二階段故障類型（TWF 等）因罕見類別樣本極少，PR-AUC 偏低為已知限制，如實呈現。

---

## 6. 模組 B · 動態健康度與 RUL（IMS 軸承）

**動機**：補 AI4I 無時間維度的限制，導入 IMS Set 2 run-to-failure 振動資料。

**方法**：20 kHz 振動 → 時/頻域特徵 → 資料驅動健康指標（HI）→ 退化起點 FPT → **指數趨勢外推** RUL。

**成果**（指標 `b1_rms`）：

| 指標 | 值 |
| --- | --- |
| FPT 時間 | 2004-02-16（失效前 **3.12 天**偵測退化） |
| 退化區 RUL MAE | 25.2 h（RMSE 36.7 h） |
| 近失效段 MAE | 30.8 h（外推牆效應） |

**方法學發現**：在**單一退化軌跡**上做監督式絕對 RUL 回歸會失敗（撞外推牆），故改採趨勢外推。
**此結果不可泛化到其他軸承/馬達**（單軌跡）。

---

## 7. 模組 B+ · 多軌跡、多工況泛化（XJTU-SY）

**動機**：IMS 僅單軌跡，無法驗證跨設備泛化 → 導入 XJTU-SY（3 工況 × 5 顆 = 15 條軌跡）。

- **健康監測泛化**：同一組**固定參數**套全 15 顆軸承，**全數成功偵測退化起點（FPT）** → 跨軸承、
  跨工況的「健康監測」可泛化（平均提前預警隨工況而異，輕載工況預警餘裕最大）。
- **監督式絕對 RUL**：
  - **LOBO**（工況內留一）：per-bearing R² 介於約 −0.2 ~ +0.6，pooled ≈ **−0.62**（壽命相近可內插）。
  - **LOCO**（跨工況留一）：R² 約 **−1.7 ~ −191**（嚴重 domain shift），pooled ≈ **−1.22**。
  - **領域自適應（E1）**：以壽命比例 / transductive z-score / CORAL 把 LOCO pooled R² 由
    **−1.22 抬到約 −0.92**（oracle 上界約 +0.15，瓶頸為壽命尺度）。

**結論**：能泛化的是「**健康監測**」；「**絕對小時數 RUL**」受任務框架與壽命尺度限制——
本系統**明確區分**這兩種能力，不誇大絕對 RUL。

---

## 8. 模組 C · 馬達電流診斷（Paderborn，人工→真實泛化）

**動機**：補上**電流模態**（MCSA 非侵入式診斷），並檢驗最關鍵的工業問題——
「用**人工製造**的故障訓練，能否泛化到**真實劣化**？」

**方法**：定子電流 + 振動的 30 維時域特徵 → 故障分類（健康 / 外環 / 內環），Random Forest。

| 實驗 | Accuracy | macro-F1 |
| --- | --- | --- |
| **Baseline**（訓練/測試同分佈：健康+人工故障） | 1.00 | **1.00** |
| **人工 → 真實**（訓練人工故障、測真實加速壽命損傷） | 0.24 | **0.20** |

**泛化落差約 0.80**。Baseline 近乎完美，但換成真實損傷後 macro-F1 崩到 0.20、真實損傷幾乎全被誤判
（混淆矩陣對角線潰散）。這正是「**人工故障訓練無法直接泛化到真實劣化**」的關鍵發現——
本系統**如實呈現此落差、不只報 baseline**。

**CE1 · 領域自適應（嘗試修補落差）**：在同一切分上試三種補救：

| 手段 | macro-F1 | 監督性 |
| --- | --- | --- |
| baseline（無自適應） | 0.200 | — |
| CORAL 協方差對齊 | 0.038 | 無監督（僅 target 特徵） |
| 工況感知標準化 | 0.189 | 無監督 |
| few-shot（每類 10 筆真實標籤） | **0.541 ± 0.042** | 用少量真實標籤 |

**誠實結論**：**無監督仿射對齊修不動**（CORAL/z-score 皆 ≤ baseline；CORAL 跨 reg 0.01–100 全 < 0.20）——
artificial→real shift 不是單純協方差/平移位移；**但每類僅 10 筆真實標籤即把 macro-F1 自 0.20 抬到 0.54**，
量化了「要多少真實標籤才夠」。few-shot 用了真實標籤（非零樣本，如實揭露）；因真實測試集無 healthy 類，
另報 outer/inner 二類 F1。

> 誠實性：Paderborn 為真實 PMSM **試驗台**訊號（MCSA 主張成立），但屬試驗台、**非產線伺服馬達**；
> 屬故障分類**非 RUL**；為子集 MVP。CORAL/z-score 僅用目標未標註特徵（無監督 DA）；few-shot 揭露用了少量真實標籤。

---

## 9. 模組 Servo · 伺服馬達健康估測（主線）

**目標**：把上述方法落到伺服馬達情境，雙輸出——**健康狀態分類**（LN 健康 / LO 輕度 / MED 中度 /
HI 高度退化）+ **退化值 DV 回歸**（0=健康、1=高度退化），並衍生健康分數、風險等級、主要異常特徵、
模型信心與建議處置。

**參考模型成果**（真實 PHM **FMCRD** 資料集，**split-aware 留出測試** n=800）：

| 任務 | 模型 | 指標 |
| --- | --- | --- |
| 健康狀態分類（4 類） | Logistic Regression（engineered） | Accuracy 0.759 · macro-F1 **0.757** |
| 退化值 DV 回歸 | Random Forest（engineered） | MAE 0.047 · RMSE 0.064 · R² **0.937** |
| 深度學習離線對照（PyTorch） | MLP 分類/回歸 + 神經 autoencoder | MLP macro-F1 0.714 · 回歸 R² 0.959；AE 留出重建誤差隨退化單調遞增（LN 0.36 → HI 2.20） |
| **真 1D-CNN（原始波形）** | 1D-CNN 分類 + 1D conv-AE（每段 run 能量包絡 8ch×256，80 runs/檔） | Accuracy 0.709 · macro-F1 **0.692**（依檔分離留出 n=320；小資料 seed 敏感 ±0.03）；conv-AE 留出重建誤差 LN 0.26 → HI 0.36 單調遞增 |

> **誠實性**：指標為真實 FMCRD 留出測試（train_* 訓練、test_* 留出）；FMCRD 為高擬真**模擬**資料集，
> 非真實工廠遙測。`train_noisy_LO` 原始檔僅 65 段（下載偏少）致 train LO 偏少，test 各類 200 完整。
> 一致性檢查：分類器健康狀態與 DV 風險矛盾時輸出 `consistency_warning`。

### 9.1 深度學習（兩階段，離線訓練、雲端唯讀）

深度學習為本專案第二部分。架構原則：**離線訓練、只把結果 JSON 提交**，雲端 / Docker 映像（裝
`requirements-dev.txt`）**不含 torch**、runtime 只讀 JSON；`torch` 獨立放 `requirements-dl.txt`，
雲端映像維持精簡。分兩階段，由淺到深：

**Phase A — PyTorch MLP + 神經 autoencoder（聚合特徵）**（[`src/models/servo_dl.py`](../src/models/servo_dl.py)）
- **MLP 分類 / DV 回歸**（隱藏層 64→32）：在 7 維 engineered 聚合特徵上以 PyTorch 重做，與傳統
  baseline 對照（macro-F1 0.714、R² 0.959）。
- **神經 autoencoder**（`7→4→2→4→7`）：**只用健康（LN）train 樣本**擬合，量各類**留出**重建誤差。
  健康資料學到的「正常樣態」對退化樣本重建變差 → 誤差隨退化單調上升（LN 0.36→HI 2.20），是無監督的
  退化偵測。此處**取代了先前以 PCA 重建誤差充當 autoencoder 的替身**，名實相符為真神經網路。

**Phase B — 真 1D-CNN（原始時序波形）**（[`build_servo_windows.py`](../src/data/build_servo_windows.py) +
[`servo_cnn.py`](../src/models/servo_cnn.py)）
- **資料**：從原始 FMCRD zip（106.66 GB）**串流**每段 run，降為原始波形的**能量包絡**——每段切 256 個
  時間塊、取每塊 8 個物理通道（扭矩 / 轉速 / 三相電流 / direct / quadrature / 位置誤差）的 std，
  得到 (8 通道 × 256 時間塊) 的時序輸入。卷積仍沿時間軸滑動，保留振幅 / 變異的退化訊號又穩定。
- **為何「每段 run 一個樣本」而非「每個短窗」**：先試過每個 1024 點短窗分類，效果接近隨機
  （留出 ≈0.27，連 logistic 吃手工窗統計上限也僅 ≈0.36）——退化訊號在**整段的長時統計**，
  單一短窗的變異估計太雜訊、且 train/test 跨檔有域偏移。改為每段 run 的能量包絡後大幅改善。
- **模型**：1D-CNN 分類（Conv1d 8→16→32→64 + 全域池化 + 線性）+ 1D conv-autoencoder（重建誤差）。
  split **依來源檔分離**（train_* vs test_* 不共享檔，無洩漏）、固定種子可重現。
- **成果與誠實下修**：把資料由 40→**80 runs/檔**（留出 test 320，更具代表性）後做**多 seed 穩健驗證**，
  發現加寬 / dropout / 加入逐塊 mean 等**都未穩健勝出**（各架構平均 0.64–0.70、std 0.03–0.06），
  故保留最簡 narrow conv。最終留出 **Accuracy 0.709 / macro-F1 0.692**（seed 敏感 ±0.03）；
  conv-AE 留出重建誤差 LN 0.26 < LO 0.26 < MED 0.29 < HI 0.36 單調。先前 40-run 單 seed 的 0.729
  屬樂觀抽樣，已如實下修。混淆矩陣顯示誤差集中在 LN↔LO（早期退化最難分），MED/HI 幾近全對。

**結論（誠實對照）**：1D-CNN 確實能**直接從原始波形**學到退化（macro-F1 ~0.69），驗證深度學習在此情境
可行；但在這份**小型 per-run、高擬真模擬**資料上**仍低於聚合特徵主線（0.757）**——退化訊號主要在
長時統計，短資料下深度模型未占優、且分數對 seed 敏感，我們不誇大單次好抽樣。後端 `GET /servo/cnn_results`
與報表頁「1D-CNN（原始波形）」卡將上述結果唯讀呈現。

**AI 訓練模擬器**：使用者可選資料量 / 特徵組 / 演算法，在後端即時訓練小模型（<0.4 s）並與離線
Reference Model 對照，教學「資料量、特徵選擇與演算法如何影響表現」。

---

## 10. 系統實作

**後端（FastAPI，`app/backend/`）**：約 30+ 端點，涵蓋模組 A/B/B+/C 結果、Servo 預測 /
訓練模擬 / 機群 / 告警 / 工單、知識庫與 LLM 助理。
- `GET /servo/fleet`、`/servo/alerts`、`/servo/work_orders`：**機群健康與告警由真實參考模型**
  在代表性 demo 運轉段上即時計算衍生（合成設備識別 + 真模型輸出，非真實遙測）。
- `POST /servo/simulate`：瀏覽器端小模型即時訓練。
- `/servo/assistant/*`：LLM 維護助理（多供應商 Groq / OpenRouter / Gemini 依序嘗試，全失敗回離線範本）。
- `/knowledge/*`：TF-IDF 維修知識庫檢索。

**前端（Next.js App Router + TypeScript + Tailwind v4 + shadcn，`web/`）**：
**AI Servo Motor Health Command Center** — 深色工業風監控介面（視覺語言參考 Kiranism shadcn
dashboard、字型思源黑體、響應式）。頁面：Overview 監控首頁（KPI / 機群健康卡 / 系統狀態 / 告警）、
Servo 健康儀表板、設備詳情頁（`/equipment/[id]`，橋接真模型預測）、訓練模擬器、欄位解釋、
知識庫、LLM 助理、告警/工單、報表中心；模組 A/B/B+/C + 關於頁完整呈現各自結果。前端以
`NEXT_PUBLIC_API_BASE_URL` 接後端契約，無法取得原始資料的互動圖表會優雅降級。

**規則式維護建議**：高 `temp_diff`→檢查散熱、高扭矩→檢查負載、高磨耗→安排保養、高故障機率→立即通報。

---

## 11. 部署

- **線上 Demo（免費）**：前端 **Vercel**（Next.js）+ 後端 **Hugging Face Spaces（Docker，2vCPU/16GB）**。
  跨網域以 `NEXT_PUBLIC_API_BASE_URL` 接，後端 CORS 開放。
- **單機整合（GCP VM）**：nginx 反向代理（`/`→Next.js:3000、`/api/`→uvicorn:8000）+ systemd 常駐 + certbot HTTPS。
- 範本與 runbook 見 [`docs/DEPLOYMENT.md`](DEPLOYMENT.md)（含 `deploy/nginx/`、`deploy/systemd/`、`deploy/huggingface/`）。
- CI（GitHub Actions）：Python 測試 + Docker build smoke + 前端 lint/tsc/build。

---

## 12. 結果彙整

| 模組 | 任務 | 關鍵指標 | 一句話結論 |
| --- | --- | --- | --- |
| A（AI4I） | 故障二元分類 | F1 0.887 · PR-AUC 0.906 · Recall 0.809 | 合成資料上靜態風險分類可用，偏重 Recall |
| B（IMS） | 健康度 / RUL | 失效前 3.12 天偵測 · 退化區 MAE 25 h | 單軌跡可偵測退化，絕對 RUL 受外推牆限制 |
| B+（XJTU） | 多軌跡泛化 | 15/15 偵測 FPT · LOCO R² −1.22→−0.92 | 健康監測可泛化、絕對 RUL 跨工況受限 |
| C（Paderborn） | 電流故障分類 + 領域自適應 | baseline F1 1.0 → 真實 0.20（gap 0.80）；CE1：無監督對齊無效、few-shot 10 筆/類 →0.54 | 人工故障訓練無法直接泛化；線性對齊修不動、需少量真實標籤 |
| Servo（主線） | 健康分類 + DV | macro-F1 0.757 · DV R² 0.937（真實 FMCRD 留出） | 真實 PHM 資料留出測試，流程完整 |

---

## 13. 誠實性聲明（報告防禦紅線）

1. **AI4I 2020** 為合成資料，不得宣稱為真實伺服馬達資料。
2. **Servo 主線** 已以完整真實 PHM **FMCRD** 資料集重訓；FMCRD 為高擬真**模擬**資料集，非真實工廠遙測。
   機群健康為真模型在代表性 demo 運轉段上的輸出、設備識別與遙測趨勢/告警排程為示意包裝。
3. **IMS Set 2** 為單軌跡，結果不可泛化到其他軸承/馬達；不在單軌跡上做深度 RUL 回歸。
4. **XJTU/IMS** 為軸承試驗台資料（非伺服馬達本體）；以「旋轉機械 / 電動機」為方法範疇、伺服馬達為應用情境。
5. **Paderborn** 為真實 PMSM **試驗台**（非產線伺服馬達）；含人工與真實兩類損傷，須如實呈現泛化落差；屬分類非 RUL；為子集 MVP。
6. **ESP32** 定位為未來實場接入 / IoT demo，非現階段訓練資料來源。
7. 本系統提供維護**建議**（決策輔助），**不直接控制馬達**。

---

## 14. 限制與未來工作

**限制**：FMCRD 為高擬真模擬（非工廠遙測）、數值不可外推真實馬達；單軌跡 RUL 偏粗；
監督式絕對 RUL 跨工況/壽命尺度受限；決策門檻採預設、未做成本敏感調整。

**未來工作（已排序）**：
1. ✅ **Servo 深度學習已完成兩階段**：Phase A（PyTorch MLP + 神經 autoencoder，聚合特徵）+
   Phase B（真 1D-CNN 分類 + 1D conv-AE，原始 FMCRD 波形能量包絡，80 runs/檔留出 macro-F1 0.692）。
   深化發現：把資料加倍（更大、更具代表性的測試集）後，多 seed 掃描顯示加寬/dropout 未穩健勝出、
   分數約 0.64–0.70（seed 敏感），把先前 40-run 單 seed 的樂觀 0.729 如實下修。後續可深化：原始逐點
   （非包絡）CNN、頻譜輸入。
2. 把 IMS/XJTU 的互動健康曲線產成可上雲的離線結果，讓雲端 demo 也能完整呈現。
3. 成本敏感門檻、ESP32 邊緣 IoT 實場接入、MLOps（重訓 / 漂移 / 版本 / 審計）。

---

## 15. 結論

本系統從**靜態風險分類（A）**延伸到**動態健康監測 / RUL（B）**，以**多軌跡多工況資料（B+）**
誠實驗證泛化邊界（區分可泛化的健康監測與受限的絕對 RUL），以**馬達電流（C）**補上電氣模態並
揭露「人工→真實」泛化落差，最終收斂到**伺服馬達健康主線（Servo）**並整合為一套可線上展示的
**工業監控指揮中心**。全程以誠實性紅線標示各結果的資料真實性與適用範圍——這正是本專案最核心的
工程素養與報告防禦立場。
