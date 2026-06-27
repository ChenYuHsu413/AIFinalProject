# 資料溯源 — 模組 Servo 真實 PHM FMCRD 資料集

> **狀態（2026-06-27）**：本檔證明模組 Servo 的參考模型訓練於**完整真實 PHM FMCRD
> 資料集（106.66 GB）**，而非先前的 placeholder 合成資料。指紋、聚合統計、留出測試
> 結果皆可獨立重驗（產生器：[`src/data/servo_data_provenance.py`](../src/data/servo_data_provenance.py)，
> 機器可讀輸出：[`outputs/metrics/servo_data_provenance.json`](../outputs/metrics/servo_data_provenance.json)）。

## 0. 誠實性聲明（務必照此措辭）

FMCRD（Fine Motion Control Rod Drive）是**高擬真物理模擬資料集**，**不是**真實工廠／產線
伺服馬達遙測。本專案所稱「真實資料」＝**那份完整的大型公開 PHM 資料集本身**（相對於我們
先前自行合成的 placeholder）。

- ✅ 可說：「以**完整真實 PHM FMCRD 資料集（106 GB）**端到端串流處理並重訓，留出測試評估。」
- ❌ 不可說：「使用真實工廠／產線伺服馬達資料。」（踩誠實紅線）

## 1. 來源指紋（可獨立重驗）

原始壓縮檔 `FMCRD_Data.zip`，**8 個 CSV、共 106,661,902,496 bytes（106.66 GB）未壓縮**。
以下 CRC32／大小**直接讀自 zip 中央目錄**——任何持有同一份檔案者可於數秒內比對：

| 檔名 | 大小 (GB) | CRC32 | split | 類別 |
| --- | ---: | --- | --- | --- |
| `test_load0_1e_m15_200x5.csv`  | 13.28 | `471906ac` | test  | LN |
| `test_noisy_1e_m15_200x5HI.csv`  | 13.33 | `b13257c8` | test  | HI |
| `test_noisy_1e_m15_200x5LO.csv`  | 13.33 | `38253cc3` | test  | LO |
| `test_noisy_1e_m15_200x5MED.csv` | 13.39 | `c9db6874` | test  | MED |
| `train_load0_1e_m15_200x5.csv`  | 13.27 | `cc50f5e5` | train | LN |
| `train_noisy_1e_m15_200x5HI.csv`  | 13.33 | `cb9b40b4` | train | HI |
| `train_noisy_1e_m15_200x5LO.csv`  | 13.33 | `e199567a` | train | LO |
| `train_noisy_1e_m15_200x5MED.csv` | 13.39 | `bf6b6807` | train | MED |

每檔 15 欄，與 [`servo_features.RAW_COLUMNS`](../src/features/servo_features.py) 完全吻合；
`ylabel` 已是 LN/LO/MED/HI；`DV` 為物理單位（max≈5012），訓練前正規化至 0..1。

## 2. 處理管線（不解壓、不爆記憶體）

[`src/data/build_servo_from_zip.py`](../src/data/build_servo_from_zip.py)：

- **串流**逐檔從 zip 分塊讀（~4.4 億個時間點），**線上累計** mean/std/min/max/rms 與
  三相 current_rms，**從不解壓**整包（106 GB 不落地）；每檔 checkpoint 可續跑。
- **正確性**：線上統計與既有 `aggregate_run`（兩遍法）在小樣本上比對，**最大相對誤差 8.8e-13**
  （等同浮點精度）。
- 修掉 `train_noisy_LO` 的 `i_3p_c` 少量非數值雜訊（`pd.to_numeric(errors="coerce")`）。

## 3. 聚合產物（可由 parquet 重驗）

[`data/processed/servo_features.parquet`](../data/processed/servo_features.parquet)：
**1,465 段運轉 × 56 維特徵**，含 `split` 欄。

| split | LN | LO | MED | HI |
| --- | ---: | ---: | ---: | ---: |
| train | 200 | **65** | 200 | 200 |
| test  | 200 | 200 | 200 | 200 |

> **如實揭露**：`train_noisy_LO` 原始檔僅含 65 段（其原始列數約為其他檔的 1/3，下載偏少），
> 故 train LO 偏少；test 各類 200 段完整。

正規化 DV 各類**單調分離**（真資料的物理特徵，合成 placeholder 給不出）：

| 類別 | DV mean | DV range |
| --- | ---: | --- |
| LN  | 0.0003 | 0.000–0.001 |
| LO  | 0.076  | 0.001–0.193 |
| MED | 0.319  | 0.156–0.489 |
| HI  | 0.639  | 0.275–1.000 |

## 4. 留出測試結果（train_* 訓練、test_* 評估）

`eval=holdout_test`、`placeholder=false`、n=800：

- 分類 **logistic_regression macro-F1 = 0.757**
- DV 回歸 **RandomForest R² = 0.937、MAE = 0.047**

混淆矩陣與 DV 分布圖：[`outputs/figures/servo_provenance.png`](../outputs/figures/servo_provenance.png)。
LN/HI 區分清楚、相鄰的 LO↔MED 有重疊——noisy 資料下的誠實表現。

## 5. 如何重新驗證

```bash
# (1) 指紋：列出 8 檔 CRC32 + 總 bytes（讀 zip 中央目錄，數秒）
python -m src.data.servo_data_provenance --zip <FMCRD_Data.zip 路徑>
#     -> outputs/metrics/servo_data_provenance.json + outputs/figures/servo_provenance.png

# (2) 完整重建（串流 106 GB，約 16 分；每檔 checkpoint 可續跑）
python -m src.data.build_servo_from_zip --zip <FMCRD_Data.zip 路徑>
python -m src.models.train_servo        # 留出測試指標
python -m src.models.servo_dl           # DL 離線 baseline
```

線上 demo 亦可即時佐證：後端 `GET /servo/provenance` 回傳本溯源摘要，前端 Servo 儀表板
顯示「訓練於真實 PHM FMCRD」面板（含 placeholder=false 與留出指標）。

相關文件：[`MODULE_SERVO_PLAN.md`](MODULE_SERVO_PLAN.md)（§1 誠實性、§9 重建、§10 導入紀錄）、
[`DEPLOYMENT.md`](DEPLOYMENT.md)。
