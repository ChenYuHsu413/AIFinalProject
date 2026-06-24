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

---

# 模組 B — IMS 軸承資料集（動態健康度 / RUL）

模組 B 使用 **IMS / NASA Bearing Dataset 的 Set 2（Test 2）**，提供 run-to-failure
全壽命振動數據，用來建立健康分數退化曲線與 RUL 預測。詳見
`docs/MODULE_B_IMS_PLAN.md`。

## 如何取得資料集

1. 從以下任一來源下載 IMS Bearing Dataset：
   - NASA Prognostics Data Repository（Bearing Data Set，由 IMS / 辛辛那提大學貢獻）
   - 或 Kaggle 鏡像：<https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset>
2. 解壓後取出 **第二組實驗**資料夾（常見命名為 `2nd_test`，內含 984 個檔）。
3. 放置於：`data/raw/ims/2nd_test/`。

放置後目錄樹應為：

```
data/
└── raw/
    └── ims/
        └── 2nd_test/
            ├── 2004.02.12.10.32.39   <-- 984 個無副檔名的時間戳檔
            ├── 2004.02.12.10.42.39
            └── ...
```

## 資料 schema

* 每個檔 = 1 秒振動快照 @ 20 kHz，形狀 `20480 × 4`，ASCII、tab 分隔、**無表頭**。
* 4 欄 = 4 個軸承各一個加速度通道（B1~B4）。
* 檔名即時間戳（`YYYY.MM.DD.HH.MM.SS`），用來還原時間順序。
* **B1（第 1 欄）在實驗末期發生外圈剝落**，是本模組的建模對象。

## 重要注意事項

* **不進 git。** Set 2 解壓後約 1.5 GB，`.gitignore` 已排除 `data/`。
* **RUL 為衍生標籤。** 資料本身無 RUL 欄位；由「最後一檔 = 故障」反推
  （`src/data/build_ims_dataset.py`，線性 100→0 健康分數）。
* **建立特徵表：** 放好資料後執行 `python -m src.data.build_ims_dataset`，
  會輸出 `data/processed/ims_set2_features.parquet`。

---

# 模組 C — Paderborn 軸承資料集（馬達電流 MCSA + 振動，故障分類）

模組 C 使用 **Paderborn University Bearing Dataset（KAt-DataCenter，Lessmeier et al. 2016）**，
提供**馬達定子電流 + 振動 + 溫度/轉速/扭矩**多感測器量測，用於**軸承故障分類**（非 RUL），
補上專案目前缺的「電流模態」。頭條實驗：訓練「健康 + 人工故障」、測「真實加速壽命損傷」，
量化人工→真實的 domain shift。詳見 `docs/MODULE_C_PADERBORN_PLAN.md`。

## 如何取得資料集

1. 從以下任一來源下載 Paderborn Bearing Dataset：
   - 官方下載頁（KAt Bearing DataCenter）：
     <https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download>
     資料檔下載入口：<https://groups.uni-paderborn.de/kat/BearingDataCenter/>
   - 或 Kaggle 鏡像（搜尋 "Paderborn bearing"）。
2. 每個**軸承碼**為一個 ZIP（如 `KA01.rar`），解壓得到以該碼命名的資料夾。
3. 把所需軸承碼資料夾放到：`data/raw/paderborn/<軸承碼>/`。
   （MVP 預設用到的碼見 `config.yaml` 的 `paderborn.bearings`；可自行增減。）

放置後目錄樹應為：

```
data/
└── raw/
    └── paderborn/
        ├── K001/                       <-- 健康軸承
        │   ├── N15_M07_F10_K001_1.mat  <-- <工況>_<碼>_<序號>.mat（每工況 20 筆）
        │   └── ...
        ├── KA01/                       <-- 人工外環故障
        ├── KI01/                       <-- 人工內環故障
        ├── KA04/                       <-- 真實外環損傷（加速壽命）
        └── ...
```

## 資料 schema

* 每個 `.mat` = 一筆量測的 MATLAB struct，訊號在 `Y` 欄位的具名陣列。
* 本專案取用的通道（見 `config.yaml` `paderborn.channels`）：
  `vibration_1`（振動）、`phase_current_1` / `phase_current_2`（兩相馬達電流），皆 64 kHz。
* 標籤由軸承碼分組推得：`fault_class ∈ {healthy, outer, inner}`、
  `damage_origin ∈ {healthy, artificial, real}`。

## 重要注意事項

* **不進 git。** 完整資料集約 20 GB，`.gitignore` 已排除 `data/raw/paderborn/`；
  只提交產出的 `data/processed/paderborn_features.parquet` 與 `outputs/metrics/paderborn_*`。
* **真實 + 人工混合損傷。** 含 EDM/雕刻等**人工**故障與加速壽命**真實**損傷；報告須誠實揭露
  「訓練人工、測真實」的設計與泛化落差。
* **屬故障分類，非 RUL。** 不宣稱剩餘壽命；電流為真實 PMSM 試驗台訊號，但屬**試驗台非產線伺服馬達**。
* **解析較重（.mat）。** 透過 `scipy.io.loadmat` 解析巢狀 struct，無需新增依賴。
* **建立特徵表：** 放好資料後執行 `python -m src.data.build_paderborn_dataset`
  （輸出 `data/processed/paderborn_features.parquet`），再 `python -m src.models.train_paderborn`。
