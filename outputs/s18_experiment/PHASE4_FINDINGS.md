# 情境 18 Phase 4 — diagnosis JSON 與收官摘要

> **狀態（2026-07-19）**：情境 18 實驗線（Phase 0–4）**收官**。
> 產出 diagnosis JSON 兩筆（critical / warning）、DATA_CONTRACT v1.1 修訂建議書七條。
> 未竟事項見 §5。

產出：`diagnosis_sample_18.json`、[`docs/contracts/DATA_CONTRACT_v1.0.md`](../../docs/contracts/DATA_CONTRACT_v1.0.md)（verbatim）、
[`DATA_CONTRACT_v1.1_PROPOSAL.md`](../../docs/contracts/DATA_CONTRACT_v1.1_PROPOSAL.md)
程式：`src/s18/build_diagnosis_json.py`

---

## 1. severity 門檻與 test 落點

門檻取 train LN 複合分數的經驗分位數（與既有 P95 觸發門檻同源）：

| 級別 | 分位數 | 門檻值 |
|---|---|---|
| watch | P90 | 1.7701 |
| warning | P95 | 1.9437 |
| critical | P99 | 2.5048 |

test 四類落點（各 n=200）：

| 等級 | normal | watch | warning | critical | is_anomaly(>P95) | 中位數 |
|---|---|---|---|---|---|---|
| LN | 79.0% | 8.5% | 10.0% | 2.5% | 12.5% | 1.39 |
| LO | 81.0% | 7.0% | 9.5% | 2.5% | 12.0% | 1.40 |
| MED | 55.0% | 12.0% | 23.0% | 10.0% | 33.0% | 1.71 |
| HI | 5.5% | 4.5% | 28.5% | **61.5%** | **90.0%** | 2.68 |

**接線檢查**：`is_anomaly` 四類數字與 Phase 2 §4.4 觸發率完全一致（同一條線的細分）。

**構造性說明（預先寫明）**：watch 依 P90 構造在健康件上約有 10% 觸發（實測 8.5%）；
critical 依 P99 構造期望 1%，test LN 實測 **2.5%**（門檻取自 train LN，套用至 test LN
時分位數不精確重現 + 抽樣變異）。此為定義使然，非誤報率異常，但「critical 級在健康件上
仍有 2.5%」須跟著指標一併呈現。

**severity 細分對早期邊界一樣無效**：LN 與 LO 在四級的分布幾乎逐格相同
（79.0/8.5/10.0/2.5 vs 81.0/7.0/9.5/2.5）。這是 LN/LO 邊界未突破的**第四次獨立確認**——
提高分級解析度救不了，問題在特徵可分性而非門檻粒度。

## 2. diagnosis JSON 兩筆

| # | run | 真實等級 | anomaly_score | severity | top root cause |
|---|---|---|---|---|---|
| 1 | test run **708** | HI | 4.7224 | **critical** / `reduce_speed_and_notify` / red | `STIFF_TorqueSlope` z=**−11.6** |
| 2 | test run **429** | MED | 2.1416 | **warning** / `adjust_parameters` / orange | `STIFF_TorqueSlope` z=−3.485 |

run 708 的 z=−11.6 與 Phase 2 圖 4 完全一致。第二筆展示分級不是只有極端值才動作。

**紅線遵守（機器檢查）**：
- `BL_DeadZone` **未出現在任何 `root_cause_ranking`**；僅出現於 `not_estimable` 的說明中
- `drive_adjustments` 中**無 P2-60、無 `set_from_feature_value`**；背隙補償改列
  `not_estimable`，理由為「本工況無反轉激勵，背隙量不可估計」
- `drive_adjustments` 一律 `value: null` + `advisory_review`，不輸出可直接寫入驅動器的量值
- `plc_adjustments` 為空陣列（無可信量值）
- `fallback.chain = ["model_output"]`、`trigger_alert = false`，並註明離線批次分析無 fallback 鏈語意
- `meaning` 依措辭紀律撰寫，torque 基特徵一律附「與負載條件存在混合效應」註記

## 3. DATA_CONTRACT 處置

v1.0 **verbatim 收錄**進 `docs/contracts/DATA_CONTRACT_v1.0.md`
（sha256 `cf830718a7dbb990…`，10,108 bytes，以 `cp` 複製後逐位元比對確認）。
**原檔不修改**；v1.1 為獨立提案檔，含七條建議：

① Diagnosis 缺正式 schema（附錄引用之 `api/schemas.py` 未隨合約交付）
② `severity` 型態 §1 字串 / §2 巢狀不一致
③ 情境 18 根因特徵表依實證更新 + 新增「激勵條件前提」欄
④ `severity.action` 未列舉合法值
⑤ `data_provenance_warning` 收編進 schema
⑥ **hash chain 驗證的 `prev_hash` 未比對 —— 鏈驗證實際失效**（附修正碼）
⑦ 門檻語意文件化（P90/95/99 取自零負載 LN 基線）

其中 ⑥ 屬功能性缺陷：`prev_hash` 被賦值但從未參與比較，
故該函式只驗單筆完整性、完全未驗鏈結——刪除中間任意筆或重排順序皆可通過。

## 4. 情境 18 實驗線總結

**標題級措辭**：
> 情境 18：機械傳動層——真實資料**部分驗證**（摩擦/剛性 ✅；背隙：激勵條件不支持 ⚠️）

**成立**：`FR_Coulomb`（within-noisy ρ=0.794）、`STIFF_TorqueSlope`（ρ=−0.496），
均 p<0.001；HI 偵測率 90.0%；對照組 `FE_RMS`/`FE_Max` 全程無訊號，
證明判別力非追隨誤差之換皮。

**不成立**：背隙族四項全部無訊號，歸因為**激勵條件不支持**而非訊號微弱。

**未突破**：LN/LO 早期邊界（AUC ≤ 0.556），與 repo E1–E5 之類別不平衡結論互相印證。

**消融**：兩變體皆無提升（B −0.006、A −0.061）→ s18 特徵的資訊已被既有 21 維涵蓋；
其價值在**物理可解釋性與根因排名**，不在新增判別力。

**方法論貢獻**：本實驗新增第六類落差實例——
**溯源紀律的落差**（資料集層級：FMCRD 出處與授權未記載；合約層級：合約活在交付附件而非版控）。
連同 V6 的 FE 恆等式、GitHub 版 §16 的三次洩漏、以及本次自製自捕的
`_isna` 欄洩漏 train/test 身分，方法論章節共六個實例。

## 5. 未竟事項

1. **FMCRD 資料集出處與授權待確認**（最高優先）。
   `docs/DATA_PROVENANCE.md` 有 CRC32 指紋與檔名，但**無來源 URL、無授權條款、無引用格式**。
   repo 為 PUBLIC + MIT，且已公開 commit 四份衍生特徵表
   （`servo_features.parquet` 等，早於本實驗）。本實驗新增的
   `s18_features_{train,test}.parquet` 屬同類（per-run 聚合，重建不了原始波形），
   風險等級相同。**建議**：於 `DATA_PROVENANCE.md` 新增「來源與授權」節，
   四個資料集各記出處 URL、授權條款、引用格式、衍生資料再散布允許狀態；
   Kaggle 鏡像條款與原始發布方（NASA PCoE / Paderborn KAt / XJTU-SY）條款分開記。
   若 FMCRD 追不到出處，誠實路徑為如實記「來源待確認」+ 考慮停止散布衍生 parquet。

2. **DATA_CONTRACT v1.1 待 zip 專案維護方採納**。七條建議中 ⑥（hash chain 驗證失效）
   屬功能性缺陷，影響 IEC 61508 合規主張，建議優先處理。

3. **`FR_Viscous` 的 regime 切分**。within-noisy 出現 ρ=−0.407 但已依預註冊判準標 N/A
   （同號條件未過，且係數為負而黏滯係數物理上須非負）。
   記為「值得未來以正確的 regime 切分重新定義後檢驗」，本次不追溯翻案。

4. **背隙路徑需另尋工況驗證**。若要驗證情境 18 的背隙支柱，需具**連續軌跡激勵**
   （斜坡/正弦）或**足量方向反轉**的資料集；本資料的階梯定位循環結構上不支持。

5. **contract-matched 驗證閘門未執行**（Phase 3）。變體以自身 config 過閘門需重建
   `servo.feature_demo` 的 smoke 固件；因兩變體均無提升而未執行，
   若未來要推進特徵契約變更則需補做。
