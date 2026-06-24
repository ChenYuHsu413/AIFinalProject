# 模組 B：動態健康度與剩餘壽命預測（IMS 軸承）成果與限制

> **狀態（2026-06-24）**：IMS 趨勢外推 RUL 已於 2026-06-22 完成（產物見 `outputs/metrics/ims_rul.json`、`ims_rul_predictions.csv`）；**模組 B+（XJTU 多軌跡 / 多工況泛化）已完成**（15 顆 × 3 工況；健康監測 FPT 跨工況泛化成立，監督式絕對 RUL 受壽命尺度/domain shift 限制；見下方專節與 `outputs/metrics/xjtu_*`）；**延伸軌 E1（跨工況自適應 RUL）已完成**（baseline + 壽命比例 / transductive z-score / CORAL 消融，最佳 CORAL 把 LOCO R² −1.22 抬到 −0.92、oracle 上界 +0.15；`src/models/eval_xjtu_domain_adapt.py`）；**E2（維護建議決策層）已完成**（純函式 `src/models/maintenance_advice.py` + Dashboard 卡片）；**E3（即時串流回放）已完成**（Plotly 瀏覽器端動畫逐快照重播 HI/FPT/RUL，重用既有函式、無新產物）；E1/E2/E3 已抽出至獨立的「**B+ 延伸應用**」頁、以三個 tab 呈現（多軌跡泛化頁回到核心）；三軌延伸全數完成。1D-CNN AE 雙印證已推遲、尚未實作。

為彌補 AI4I 2020 合成資料「單筆點資料、無時間維度」的限制，本模組導入 NASA / IMS
軸承 run-to-failure 資料集（Set 2，軸承外圈剝落），將系統從**靜態風險分類**延伸為
**動態健康度監測**。流程為：解析 20 kHz 振動快照 → 抽取時域 / 頻域特徵（RMS、峭度、
BPFO 頻帶能量等）→ 由振動特徵建構資料驅動健康指標 → 偵測退化起點（FPT）→ 以指數
趨勢外推估計剩餘壽命（RUL）。

## 成果（健康監測與早期預警）

系統能在軸承徹底失效前 **3.1 天**偵測到退化起點，並在健康分數跌破告警門檻時示警，
具備實務上「提前數天安排維護」的價值。

| 指標 | 數值 |
| --- | --- |
| 退化提前量（FPT → 失效） | 3.1 天 |
| RUL MAE（退化區） | ≈ 25 小時 |
| RUL RMSE（退化區） | ≈ 37 小時 |

## 限制（RUL 為粗估）

本軸承屬**突發型失效**——經比對 RMS、峭度、BPFO、波峰因子四項指標，皆在最後約 2%
壽命才急遽惡化，缺乏早期漸進訊號。因此 RUL 預測在退化早期誤差較大，越接近失效越
收斂。此為該失效模式的固有限制，本研究選擇**如實呈現而非過度配適**。

## 方法學發現

初期曾嘗試以監督式回歸（RandomForest / GradientBoosting + 時間切分）直接預測 RUL，
結果嚴重失敗（MAE ≈ 120 小時、R² ≈ −76）。原因在於單一退化軌跡下，測試段的 RUL
區間（0–48.5 小時）完全落在訓練段（50–164 小時）之外，而樹模型無法外推單調目標。
此一發現印證了：對單一 run-to-failure 軌跡，**趨勢外推法優於監督式回歸**。
保留 `src/models/train_rul.py` 作為「為何不能這樣做」的對照教材。

## 模組 B+：多軌跡泛化驗證（XJTU-SY，跨軸承＋跨工況）

IMS Set 2 只有 B1 一條退化軌跡，無法驗證「健康指標與門檻能否套用到其他軸承」。本模組 B+
導入 **XJTU-SY** 軸承加速壽命資料集，涵蓋 **3 種運轉工況 × 5 顆 = 15 條獨立 run-to-failure
軌跡**，並把**與 IMS 完全相同的趨勢外推管線**（健康指標 → FPT → 指數外推）以**一組固定參數**
套到全部 15 顆，**不逐顆、不逐工況調參**。

### 健康監測（FPT）跨工況泛化 ✅

| 工況 | 轉速 / 負載 | 軸承數 | 平均退化提前量 | 平均退化區 MAE |
| --- | --- | --- | --- | --- |
| C1 | 2100 rpm / 12 kN | 5 | 1.45 h | 0.57 h |
| C2 | 2250 rpm / 11 kN | 5 | 2.65 h | 1.04 h |
| C3 | 2400 rpm / 10 kN | 5 | 17.43 h | 6.82 h |
| **全部** | — | **15** | **7.18 h** | **2.81 h** |

**結論**：同一套不調參的方法在 **15 條獨立軌跡、3 種工況上全數偵測到退化起點**——健康監測
流程具跨軸承、跨工況泛化能力。各工況絕對 MAE 差異主要來自壽命尺度（C3 軸承最長約 42 h、
C1 僅約 2 h），非方法失效。

### 監督式絕對 RUL：跨壽命尺度的固有限制 ⚠️

用 RandomForest（僅瞬時振動特徵、排除時間索引）做兩種交叉驗證：

| 驗證設計 | 說明 | 合併 R² | 合併 MAE |
| --- | --- | --- | --- |
| LOBO（工況內留一軸承） | 同工況其他軸承可入訓練 | −0.62 | 11.2 h |
| LOCO（留一整個工況） | 測試工況的轉速/負載完全未見 | −1.22 | 14.2 h |

- **唯一表現好的是壽命相近的軸承**：C1（5 顆都約 2 h）的 Bearing1_1、1_3 達 R² **+0.43 / +0.59**
  ——相對 IMS 單軌跡的 −76 已是質變（外推牆消失）。
- 但只要**壽命尺度差異拉大，絕對 RUL 即崩潰**：工況內壽命差很大的 C2/C3（如 C3 為 1.9–42 h）
  幾乎全為負；**跨工況（LOCO）更嚴重**（R²≈−1.2，domain shift）。
- 根因是任務框架：各工況/軸承壽命差達約 20 倍，同樣的瞬時振動值對應到完全不同的「剩餘小時數」，
  使「絕對小時數回歸」在跨壽命尺度時 ill-posed。另已評估軸承內滾動趨勢特徵（config 開關
  `xjtu.lobo_use_trend`），未改善合併 R²，預設關閉。

**整體結論**：在多軌跡、多工況下能**穩健泛化的是「健康監測 / 退化起點偵測」**，而非「絕對小時數
RUL 回歸」。後者僅在壽命相近時可內插；跨壽命尺度 / 跨工況則需壽命正規化（如改預測壽命比例）或
領域自適應——此即延伸軌 **E1** 的攻擊目標（見下方專節，已驗證能部分改善但未解決）。這呼應 Module B
的核心論點：RUL 的成敗取決於**任務框架與資料設定**，而非單看模型強弱。

## 模組 B+ 延伸 · E1：跨工況自適應 RUL ✅

> **狀態（2026-06-24）**：已完成（baseline + 三手段消融）。規格見 [`MODULE_B_PLUS_EXTENSIONS_PLAN.md`](MODULE_B_PLUS_EXTENSIONS_PLAN.md) 之 E1；產物 `outputs/metrics/xjtu_domain_adapt.json`。

把上述 LOCO 的崩潰（pooled R² −1.22）當成 **domain shift** 來診斷與修復：在**同一組留一工況切分**上外掛三種**目標未標註**的自適應手段，做成消融對照（`src/models/eval_xjtu_domain_adapt.py`，baseline 重現 −1.223，與既有 `xjtu_loco.json` 一致）。

| 手段 | LOCO 合併 R²（hours，可比） | 合併 MAE | 性質 |
| --- | --- | --- | --- |
| baseline（無自適應） | −1.22 | 14.23 h | LOCO 原始 |
| 壽命比例（可部署還原） | −0.96 | — | 預測剩餘壽命比例 → 乘**來源平均壽命**還原（零洩漏） |
| 工況感知標準化（transductive z-score） | −1.04 | — | 各工況以自身統計標準化，目標用其**未標註特徵**統計 |
| **CORAL 協方差對齊**（最佳） | **−0.92** | — | 把來源特徵協方差對齊到目標（僅用目標未標註特徵） |
| 壽命比例 · **oracle 上界** | **+0.15** | — | 以**目標真實壽命**還原（含洩漏，僅供診斷、不可部署） |

**結論（誠實的「部分改善」）**：

- 三種自適應手段都把 LOCO 合併 R² 從 −1.22 往上抬，**最佳零洩漏手段為 CORAL（−0.92）**；但**全數仍為負**——跨工況絕對小時數 RUL 並未被「解決」。
- **關鍵診斷**：壽命比例在「已知壽命」的 **oracle 上界**達 **+0.15**（正值），代表退化的**形狀（剩餘壽命比例）確實可跨工況泛化**；真正的瓶頸是**推論期不知道該軸承的壽命尺度**（可部署還原須用來源平均壽命，對壽命差異大的工況——如 C1≈2 h vs 來源平均——會嚴重高估）。
- 這把原本被動的「跨工況會崩」升級為「**診斷出 domain shift 主因是壽命尺度、嘗試三種修復、量化其上限**」的研究結論。

**誠實聲明**：z-score / CORAL 僅用 target 的**未標註特徵**（推論期可得、無 RUL 標籤），屬 transductive / 無監督領域自適應，非偷看答案；oracle 還原使用目標真實壽命，僅作**上界診斷**、不可部署；改善幅度如實呈現，未宣稱「解決」跨工況 RUL。

## 模組 B+ 延伸 · E2：維護建議決策層 ✅

> **狀態（2026-06-24）**：已完成（步驟 1–2＋選配成本對照）。規格見 [`MODULE_B_PLUS_EXTENSIONS_PLAN.md`](MODULE_B_PLUS_EXTENSIONS_PLAN.md) 之 E2。

把既有的健康度 / FPT / RUL 估計轉成**可行動輸出**，呼應專案名稱的「預測性維護**建議**」這一段。

- **純函式** `maintenance_advice(health, rul, past_fpt, …) → {風險等級, 建議維護時間窗, 理由}`（`src/models/maintenance_advice.py`，無 config/IO 依賴，9 項單元測試覆蓋綠/黃/紅邊界與時間窗計算）。
- **風險等級**：健康度 ≤ 告警門檻 → 🔴 迫近失效；已過 FPT 但仍高於門檻 → 🟡 退化中；未過 FPT → 🟢 健康。
- **建議時間窗** = 剩餘壽命 ×（1 − 安全裕度，預設 0.3）；RUL 不可估時不給時間窗、改提示提高巡檢頻率。
- **成本對照（選配、示意）**：給定「非預期停機 vs 計畫維護」成本比，輸出一句期望成本說明。
- **Dashboard**：「B+ 延伸應用」頁 E2 tab，含**巡檢檢查點滑桿**（佔壽命比例）；對 15 顆軸承在該時點各畫一張風險卡片。檢查點分布隨時間推進合理（例：40% → 5🟢/10🟡/0🔴；70% → 4🟢/10🟡/1🔴；100% → 15🔴）。

**誠實聲明**：屬**決策支援啟發式**，非控制；成本參數為**示意值**，未對真實維護結果驗證；run-to-failure 資料無「真實的現在」，檢查點滑桿為**模擬巡檢時點**之 what-if，非即時串流（即時回放列 E3）。定位與 sidebar「DECISION SUPPORT · NOT CONTROL」一致。

## 模組 B+ 延伸 · E3：即時串流回放 ✅

> **狀態（2026-06-24）**：已完成（純 UI、無新產物、無新依賴），位於獨立的「**B+ 延伸應用**」頁之 E3 tab。規格見 [`MODULE_B_PLUS_EXTENSIONS_PLAN.md`](MODULE_B_PLUS_EXTENSIONS_PLAN.md) 之 E3。

把整套監測流程演成**會動的監測台**：選一顆軸承，沿時間軸**逐快照**重播。

- **回放核心**：呼叫既有 `build_health_indicator` / `detect_fpt` / `extrapolate_rul`。因滾動 RUL 擬合為**回溯式**，對全序列算一次後 `rul[k]` 即等於「只看過前 *k* 個快照」當下會算出的值；健康基線與失效門檻為**預先校準**的固定參考線（座標軸固定，曲線「長進」穩定畫框）。
- **動畫與控制**：以 **Plotly 原生 frames 在瀏覽器端播放**（▶ 0.5x–4x 速度鍵 / ⏸ 暫停 / 拉桿定位任一快照）。播放中可**即時切速**（`mode=immediate`＋`fromcurrent`，從目前幀續播、不重置），播完**停在最後一幀**。全程**零 Streamlit 重跑 → 不閃爍**（取代早期 `st.fragment`＋`st.rerun` 逐幀重繪的會閃版本）。
- **視覺**：HI 一格格長、跨過 FPT 後目前點轉紅並標示 ★；左上**狀態框**即時顯示健康度 / RUL / 風險（**重用 E2** `maintenance_advice` 邏輯，顏色隨綠/黃/紅變）。驗證（Bearing1_3）：k=13 綠、k=20 過 FPT 轉黃、k=158 健康度跌破告警轉紅，RUL 收斂趨零。

**誠實聲明**：為**離線資料重播**之視覺化，**非真實即時感測串流**；健康基線 / 失效門檻為預先校準參考；ESP32 真場即時接入仍列未來工作。

## 對應實作

| 項目 | 檔案 |
| --- | --- |
| 載入 IMS 快照 | `src/data/load_ims.py` |
| 振動特徵抽取 | `src/features/vibration_features.py` |
| 特徵表 + RUL 標籤 | `src/data/build_ims_dataset.py` |
| 趨勢外推 RUL（採用） | `src/models/rul_extrapolation.py` |
| 監督式回歸（對照，已知失敗） | `src/models/train_rul.py` |
| Dashboard 頁籤 | `app/streamlit_app.py`（「動態健康度 (IMS)」）|
| —— 模組 B+（XJTU 多軌跡 / 多工況）—— | |
| 載入 XJTU 快照 | `src/data/load_xjtu.py` |
| 特徵表（15 顆 / 3 工況） | `src/data/build_xjtu_dataset.py` |
| 跨軸承＋跨工況泛化評估 | `src/models/eval_xjtu_generalization.py` |
| LOBO 監督式 RUL（工況內留一） | `src/models/train_rul_lobo.py` |
| LOCO 監督式 RUL（留一工況） | `src/models/train_rul_loco.py` |
| 跨工況自適應 RUL（E1，消融） | `src/models/eval_xjtu_domain_adapt.py`（測試 `tests/test_domain_adapt.py`）|
| 維護建議決策層（E2，純函式） | `src/models/maintenance_advice.py`（測試 `tests/test_maintenance_advice.py`）|
| 維護建議卡片（E2，Dashboard） | `app/streamlit_app.py`（`_render_bplus_e2`，「B+ 延伸應用」頁 E2 tab）|
| 即時串流回放（E3，純 UI） | `app/streamlit_app.py`（`_render_bplus_e3` / `_xjtu_replay`）＋ `src/ui/charts.py`（`xjtu_replay_animation`）|
| E1/E2/E3 分頁與 tabs | `app/streamlit_app.py`（「B+ 延伸應用」頁；`_render_bplus_e1/e2/e3`）|

完整規劃與分階段交付見 [`MODULE_B_IMS_PLAN.md`](MODULE_B_IMS_PLAN.md)；多軌跡泛化規格見 [`MODULE_B_PLUS_XJTU_PLAN.md`](MODULE_B_PLUS_XJTU_PLAN.md)。

> **模組 C（Paderborn · 馬達電流診斷）**：獨立新軌，以馬達定子電流（MCSA）+ 振動做故障分類、
> 驗證「人工→真實」泛化。規格與真實結果回寫見 [`MODULE_C_PADERBORN_PLAN.md`](MODULE_C_PADERBORN_PLAN.md)。
