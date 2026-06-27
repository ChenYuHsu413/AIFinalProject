# 專題報告大綱

> 此大綱作為書面或簡報報告的骨架使用。文中提到的數字與圖表都由訓練與評估
> 腳本產生，分別儲存於 `outputs/figures/` 與 `outputs/metrics/`。
>
> **完整書面報告（2026-06-27）**：本大綱已展開為含 Servo 主線、模組 C、前端 Command Center 與
> 部署的完整報告，見 [`docs/FINAL_REPORT.md`](../../docs/FINAL_REPORT.md)（填入真實指標）。
> 下方大綱保留為 A/B/B+ 的原始骨架。

## 1. 摘要
一段文字摘要：問題、資料集（AI4I 2020 合成資料）、方法（CRISP-DM）、最佳模型、
重點指標、系統定位（決策輔助，非馬達控制）。本系統由三軌構成：**模組 A（AI4I 靜態
風險分類）→ 模組 B（IMS 動態健康度 / RUL）→ 模組 B+（XJTU 多軌跡、多工況泛化驗證）**，
主線敘事為「從靜態風險評估，延伸到動態健康監測，再以多軌跡資料誠實驗證其泛化邊界」。

## 2. 專題背景與動機
- 工業伺服馬達運轉的預測性維護需求。
- 計畫性維護 vs. 非計畫性停機的成本對比。
- 模組分工：**模組 A 聚焦靜態故障風險分類**（AI4I 無時間維度）；**模組 B / B+ 補上動態
  健康度與 RUL**（IMS / XJTU run-to-failure 時序資料）。

## 3. 商業理解（Business Understanding）
- 利害關係人：維護工程師、生產規劃。
- 系統支援的決策：派工、巡檢、班次調度。
- 範圍外：即時馬達控制迴路、安全互鎖。

## 4. 資料理解（Data Understanding）
- 10,000 筆，故障比例約 3.4%（嚴重不平衡）。
- 欄位清單與型別。
- 故障類型共生情形（TWF / HDF / PWF / OSF / RNF）。
- 視覺化分析：故障 / 非故障的數值分布、相關矩陣
  （見 `outputs/figures/eda_*.png`）。

## 5. 資料準備（Data Preparation）
- 移除 `UDI`、`Product ID`（純識別碼）。
- 從 X 中移除 `TWF / HDF / PWF / OSF / RNF`，避免標籤洩漏。
- `Type` 採 One-Hot；數值欄位採 StandardScaler（僅對 scale-sensitive 模型）；
  全部包在 `Pipeline + ColumnTransformer` 中。
- 使用 `train_test_split` 做 stratified 80 / 20 切分，`random_state` 由 config
  控管，確保可重現。
- Scaler **僅在訓練折上 fit**。

## 6. 特徵工程
| 特徵                      | 計算方式                              | 工程直覺                                |
| ------------------------- | ------------------------------------- | --------------------------------------- |
| `temp_diff`               | 製程溫度 − 環境溫度                   | 散熱迴路所承受的熱負載                  |
| `power_proxy`             | 扭矩 × 轉速                           | 機械功率代理                            |
| `wear_torque_interaction` | 刀具磨耗 × 扭矩                       | 老舊刀具承受高負載的風險                |
| `wear_speed_interaction`  | 刀具磨耗 × 轉速                       | 老舊刀具高速運轉的風險                  |
| `temp_wear_interaction`   | 製程溫度 × 刀具磨耗                   | 熱負載與刀具年齡的綜合效應              |

## 7. 模型建立（Modeling）
- Baseline：Logistic Regression（`class_weight="balanced"`）。
- 比較對象：Decision Tree、Random Forest、SVM (RBF)、Gradient Boosting、
  KNN、MLP、Naive Bayes，必要時加入 XGBoost / LightGBM。
- 特徵組合比較：
  - A：原始欄位
  - B：原始 + 工程特徵
  - C：SelectKBest（top 8）
  - D：RFE with Logistic Regression（top 8）
  - E：Random Forest importance（top 8）
- 最佳模型再做 permutation importance，存到
  `outputs/metrics/feature_importance_permutation.csv`。

## 8. 模型評估（Evaluation）
- 為何不只看 Accuracy：故障率約 3%，預測「全部沒故障」就有 97% Accuracy
  卻完全沒實用價值。
- 報告指標：Precision、Recall、F1、ROC-AUC、PR-AUC、混淆矩陣。
  在不平衡資料下，**PR-AUC** 是單一最有資訊量的純量指標。
- 取捨討論：
  - Recall 高、Precision 低 → 誤報多、漏報少。
  - Precision 高、Recall 低 → 誤報少、漏報多。
  - 預測性維護一般偏向 Recall（漏掉真正故障的成本通常大於誤報）。

## 9. 最佳模型選擇
- 選擇準則：F1（可於 `config.yaml` 調整）。
- 報告內容：選定的模型、特徵組合、重點指標、推論成本。

## 10. 系統實作
- `src/` 下以 CRISP-DM 階段分模組（data / features / models / visualization）。
- Streamlit Dashboard：手動輸入、批次 CSV、圖表展示。
- FastAPI 服務：`/health`、`/predict`、`/batch_predict`、`/model_info`、`/metrics`。
- 規則式維護建議（`src/models/predict.py::_maintenance_advice`）：
  - 高 `temp_diff` → 檢查散熱。
  - 高扭矩 → 檢查負載。
  - 高刀具磨耗 → 安排保養。
  - 高故障機率 → 立即通報。

## 11. 結果展示
- 引用 `outputs/metrics/model_comparison.csv` 的完整比較表。
- 引用四張比較圖以及混淆矩陣。
- 引用原生與 permutation 特徵重要性圖。

## 12. 模組 B：動態健康度與 RUL（IMS 軸承）
- 動機：補 AI4I 無時間維度的限制，導入 NASA / IMS Set 2 run-to-failure 振動資料。
- 方法：20 kHz 振動 → 時/頻域特徵 → 資料驅動健康指標 → 退化起點 FPT → 指數趨勢外推 RUL。
- 成果：失效前約 3.1 天偵測退化；退化區 RUL MAE ≈ 25 h。
- 方法學發現：監督式回歸在單一退化軌跡上失敗（R²≈−76，無法外推）→ 改採趨勢外推。
- 詳見 `docs/MODULE_B_RESULTS.md`。

## 13. 模組 B+：多軌跡、多工況泛化驗證（XJTU 軸承）
- 動機：IMS 僅單軌跡，無法驗證跨設備泛化 → 導入 XJTU-SY（3 工況 × 5 顆 = 15 條軌跡）。
- 健康監測：同一組固定參數套全 15 顆，**全數偵測退化起點** → 跨軸承、跨工況泛化成立。
- 監督式絕對 RUL：LOBO（工況內）pooled R²≈−0.62、LOCO（跨工況）≈−1.22 → 僅壽命相近可
  內插，跨壽命尺度 / 工況失效（domain shift）。
- 結論：能泛化的是「健康監測」，「絕對小時數 RUL」受任務框架與壽命尺度限制。
- 詳見 `docs/MODULE_B_RESULTS.md`、`docs/MODULE_B_PLUS_XJTU_PLAN.md`。

## 14. 外部真實資料來源評估與擴充策略
- 評估 6 個公開資料集（Paderborn / XJTU-SY / FEMTO / Mendeley / PMSM / CWRU）的真實性、
  契合度與風險，說明為何優先 XJTU-SY（已實作）、次選 Paderborn。
- 完整比較表、限制聲明與遷移路線見 `docs/DATASET_EVALUATION.md`。

## 15. 限制與未來工作
**限制**
- AI4I 2020 為合成資料，數值不可直接外推到真實馬達。
- IMS / XJTU 為軸承試驗台資料（非伺服馬達本體）；本專案以「旋轉機械 / 電動機預測性維護」
  為方法範疇、以伺服馬達為應用情境。
- 監督式絕對 RUL 跨壽命尺度 / 工況泛化受限（見 §13）；決策門檻採預設、未做成本敏感調整。

**未來工作（已排序）**
1. **下一步：Paderborn Bearing Dataset** —— 引入馬達定子電流（MCSA 非侵入式診斷）＋多感測器
   （電流 / 振動 / 溫度）＋真實損傷子集，最貼近「馬達」主題，並補上目前缺的電流模態。
2. （推遲）其餘公開資料集（FEMTO / Mendeley / PMSM）、RUL 壽命正規化 / 領域自適應、
   模組 B（IMS）的 1D-CNN Autoencoder 深度對照（Servo 主線已完成 Phase A/B 深度學習）、
   ESP32 邊緣 IoT 實場接入、成本敏感門檻、MLOps（重訓 / 漂移 / 版本 / 審計）。

## 16. 結論
一段文字總結：本系統從**靜態風險分類（A）**延伸到**動態健康監測 / RUL（B）**，再以**多軌跡、
多工況資料（B+）**誠實驗證泛化邊界——明確區分可泛化能力（健康監測）與受限能力（絕對 RUL），
並以 Paderborn 電流診斷為下一步擴充。
