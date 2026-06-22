# Model Card — AI 伺服馬達預測性維護原型

> 本文件由 `src/models/model_card.py` 在執行 `python -m src.models.evaluate` 時
> 自動產生，避免手寫文件與實際部署模型脫節。

## 1. 模型基本資訊

| 項目 | 值 |
| --- | --- |
| 模型名稱 | **gradient_boosting** |
| 模型類別 | `GradientBoostingClassifier` |
| 特徵組合 | **D_rfe_top8** |
| 訓練資料 | UCI AI4I 2020（合成資料集） |
| 主要目標 | `Machine failure`（二元分類） |
| 是否經過 Optuna 調參 | 否（採用原始最佳） |
| 模型卡產生時間 | 2026-06-22 09:53 |

## 2. 預期用途與範圍

### ✅ 預期用途
- 在「**目前運轉條件**」下估計伺服馬達故障風險，輸出機率、健康分數、
  風險等級與規則式維護建議。
- 配合第二階段故障類型分類器，提示**可能的故障模式**
  （TWF / HDF / PWF / OSF / RNF）。
- 作為**維護工程師的決策輔助**，協助排程巡檢與保養。

### ❌ 不適用情境
- 直接控制馬達運轉或安全互鎖（本系統**不**發送控制命令）。
- 精準預測剩餘壽命（RUL）：AI4I 2020 不含時間序列／run-to-failure 資料。
- 未經實際工廠資料驗證的高風險場域部署。

## 3. 訓練細節

### 特徵欄位（8 個）
`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `temp_diff`, `power_proxy`, `wear_torque_interaction`, `temp_wear_interaction`, `Type`

### 模型超參數
| 參數 | 值 |
| --- | --- |
| `ccp_alpha` | `0.0` |
| `criterion` | `deprecated` |
| `init` | `None` |
| `learning_rate` | `0.1` |
| `loss` | `log_loss` |
| `max_depth` | `3` |
| `max_features` | `None` |
| `max_leaf_nodes` | `None` |
| `min_impurity_decrease` | `0.0` |
| `min_samples_leaf` | `1` |
| `min_samples_split` | `2` |
| `min_weight_fraction_leaf` | `0.0` |
| `n_estimators` | `100` |
| `n_iter_no_change` | `None` |
| `random_state` | `42` |
| `subsample` | `1.0` |
| `tol` | `0.0001` |
| `validation_fraction` | `0.1` |
| `verbose` | `0` |
| `warm_start` | `False` |

## 4. 測試集評估結果

### 主要指標
| 指標 | 值 |
| --- | ---: |
| Accuracy | 0.9930 |
| Precision | 0.9821 |
| Recall | 0.8088 |
| F1 | 0.8871 |
| ROC-AUC | 0.9685 |
| PR-AUC | 0.9059 |

> **為什麼不只看 Accuracy**：AI4I 故障比例 ~3.4%，「全部猜健康」就有 ~97%
> Accuracy 但完全沒有實用價值。預測性維護情境通常以 Recall + PR-AUC
> 為主軸。

### 混淆矩陣
| | pred 0 (健康) | pred 1 (故障) |
| --- | ---: | ---: |
| **true 0 (健康)** | 1931 | 1 |
| **true 1 (故障)** | 13 | 55 |

### Permutation Importance（前 8）
| 特徵 | importance_mean (F1) |
| --- | ---: |
| num__Rotational speed [rpm] | +0.3410 |
| num__temp_diff | +0.3378 |
| num__wear_torque_interaction | +0.2464 |
| num__power_proxy | +0.2212 |
| num__Torque [Nm] | +0.0660 |
| num__temp_wear_interaction | +0.0613 |
| cat__Type_L | +0.0493 |
| num__Tool wear [min] | +0.0196 |

## 5. 超參數調整摘要

已執行調參但**未勝過**原始最佳模型，仍採用原始版本。

| 模型 | 特徵組合 | best_cv_f1 | test_f1 | best_params |
| --- | --- | ---: | ---: | --- |
| gradient_boosting | D_rfe_top8 | 0.8906 | 0.8689 | `{"n_estimators": 350, "max_depth": 5, "learning_rate": 0.010725209743171996, "subsample": 0.9879639408647978}` |
| lightgbm | B_engineered | 0.8614 | 0.8702 | `{"n_estimators": 200, "num_leaves": 122, "learning_rate": 0.26690431824362526, "min_child_samples": 42}` |
| random_forest | D_rfe_top8 | 0.8362 | 0.8571 | `{"n_estimators": 400, "max_depth": 18, "min_samples_leaf": 1, "max_features": "log2"}` |

## 6. 已知限制

- **合成資料**。AI4I 2020 來自參數化模型，絕對的指標數值不能直接外推到
  真實工廠的伺服馬達。
- **無時間序列**。缺乏 run-to-failure 軌跡，無法做嚴謹 RUL 預測。
- **決策門檻**預設為 0.5。在實際成本模型出現前，可在 Streamlit 的
  「互動式決策門檻調整」頁面探索較低門檻（提高 Recall）。
- **訓練樣本不平衡**。正樣本約 3.4%，所有指標需與基準率一同解讀。
- **RNF 類別**：依資料集設計屬隨機事件，目前模型在此類別上接近隨機猜測，
  屬於合理結果而非模型瑕疵。

## 7. 倫理與安全考量

- 本系統提供**維護決策輔助**；最終派工、停機、換刀的決策由維護工程師
  做出。建議在公司流程中明確標示「AI 建議」與「人類決定」的界線。
- 模型偏差（如對某些 Type 類別表現差異）可在 `outputs/metrics/`
  進一步分析；正式上線前建議在實際機台分群檢驗。

## 8. 維護與重訓建議

- 若取得實際感測資料（電流、電壓、震動、溫度、警報碼、維修紀錄），
  建議**重新訓練**並改以 survival analysis 或 RUL 模型替代本原型。
- 建議每月或每次製程變更後重跑：
  - `python -m src.models.train`
  - `python -m src.models.evaluate`
  - `python -m src.models.train_failure_types`
  - `python -m src.models.tune`（如需要重新調參）

## 9. 相關檔案

- 模型 binary：`outputs/models/best_model.joblib`
- 模型 metadata：`outputs/models/best_model_meta.json`
- 評估報告：`outputs/metrics/best_model_eval.json`
- 跨模型比較：`outputs/metrics/model_comparison.csv`
- 調參 trial 記錄：`outputs/metrics/tuning_history.csv`
- 二階段故障類型指標：`outputs/metrics/failure_type_comparison.csv`
- 圖表：`outputs/figures/`
