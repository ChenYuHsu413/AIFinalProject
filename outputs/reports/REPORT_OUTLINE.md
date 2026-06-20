# 專題報告大綱

> 此大綱作為書面或簡報報告的骨架使用。文中提到的數字與圖表都由訓練與評估
> 腳本產生，分別儲存於 `outputs/figures/` 與 `outputs/metrics/`。

## 1. 摘要
一段文字摘要：問題、資料集（AI4I 2020 合成資料）、方法（CRISP-DM）、最佳模型、
重點指標、系統定位（決策輔助，非馬達控制）。

## 2. 專題背景與動機
- 工業伺服馬達運轉的預測性維護需求。
- 計畫性維護 vs. 非計畫性停機的成本對比。
- 為何本原型聚焦在**故障風險**與**健康分數**，而**不**做 RUL。

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

## 12. 限制與未來工作
- AI4I 2020 是合成資料，數值不可直接外推到真實馬達。
- 無時間序列／run-to-failure 資料 → 無法訓練嚴謹的 RUL 模型。
- 決策門檻採預設 0.5，未做成本敏感調整。
- 未來方向：
  - 接入實際伺服馬達遙測（電流、電壓、震動、溫度、警報碼、維修紀錄）。
  - 改以 survival analysis / RUL 模型處理 run-to-failure 資料。
  - 成本敏感門檻調整；模型校準分析。
  - MLOps 整合：定期重訓、漂移監控、模型版本、預測審計。

## 13. 結論
一段文字總結：建立了什麼、適用於哪些情境、不適用於哪些情境。
