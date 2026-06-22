# 模組 B：動態健康度與剩餘壽命預測（IMS 軸承）成果與限制

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

## 對應實作

| 項目 | 檔案 |
| --- | --- |
| 載入 IMS 快照 | `src/data/load_ims.py` |
| 振動特徵抽取 | `src/features/vibration_features.py` |
| 特徵表 + RUL 標籤 | `src/data/build_ims_dataset.py` |
| 趨勢外推 RUL（採用） | `src/models/rul_extrapolation.py` |
| 監督式回歸（對照，已知失敗） | `src/models/train_rul.py` |
| Dashboard 頁籤 | `app/streamlit_app.py`（「動態健康度 (IMS)」）|

完整規劃與分階段交付見 [`MODULE_B_IMS_PLAN.md`](MODULE_B_IMS_PLAN.md)。
