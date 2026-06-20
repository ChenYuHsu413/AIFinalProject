# 資料說明 — AI4I 2020 Predictive Maintenance Dataset

為避免授權與檔案大小問題，本資料夾**預設不包含資料集**。

## 如何取得資料集

1. 至 UCI Machine Learning Repository：
   <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
2. 下載 `ai4i2020.csv`。
3. 放置於：`data/raw/ai4i2020.csv`。

放置後目錄樹應為：

```
data/
├── raw/
│   └── ai4i2020.csv        <-- 放在這裡
└── processed/
```

## 欄位 schema

| 欄位                       | 型別      | 用途                                  |
| -------------------------- | --------- | ------------------------------------- |
| UDI                        | int       | 列 ID（建模前移除）                   |
| Product ID                 | string    | 列 ID（建模前移除）                   |
| Type                       | string    | 品質類別 L / M / H                    |
| Air temperature [K]        | float     | 特徵                                  |
| Process temperature [K]    | float     | 特徵                                  |
| Rotational speed [rpm]     | int       | 特徵                                  |
| Torque [Nm]                | float     | 特徵                                  |
| Tool wear [min]            | int       | 特徵                                  |
| Machine failure            | int 0/1   | **主要目標**                          |
| TWF, HDF, PWF, OSF, RNF    | int 0/1   | 故障類型標籤（**不可**作為 X 使用）   |

## 重要注意事項

* **合成資料集。** AI4I 2020 由參數化模型產生，**不是**任何實際伺服馬達的長期紀錄。
* **無 RUL 標籤。** 資料中沒有 time-to-failure，不適合用來訓練剩餘壽命回歸器。
* **故障類型欄位會洩漏目標。** `TWF / HDF / PWF / OSF / RNF` 是 `Machine failure`
  的確定性成因，本專案在前處理時即從 X 移除，以避免 data leakage。
  這五個欄位可保留給「第二階段故障類型分析」使用（請見專案報告）。
