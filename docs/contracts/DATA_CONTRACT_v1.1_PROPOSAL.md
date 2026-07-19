# DATA_CONTRACT v1.1 — 修訂建議書

> **狀態（2026-07-19）**：**提案,非生效版本。**
> v1.0（[`DATA_CONTRACT_v1.0.md`](DATA_CONTRACT_v1.0.md)，sha256 `cf830718a7dbb990…`，
> 10,108 bytes）為外部交付文件，**verbatim 收錄、不作任何修改**。
> 本檔為情境 18 實驗（Phase 0–4）實作合約時發現的問題與建議，
> 待 zip 專案（AI SERVO Analysis Engine）維護方採納後才成為 v1.1。
>
> **依據**：[Phase 0–3 findings](../../outputs/s18_experiment/)、
> [`config/s18_params.yaml`](../../config/s18_params.yaml)、
> 實作產物 [`diagnosis_sample_18.json`](../../outputs/s18_experiment/diagnosis_sample_18.json)。

---

## ① Diagnosis 缺正式 JSON Schema，附錄引用之檔案未隨合約交付

**現況**：v1.0 §6 附錄對 L1 Inference 給出完整 draft-07 schema，但 Diagnosis 只寫
「完整 schema 見 `api/schemas.py` 中的 `DiagnosisResponse` class」（L312）。
該檔案不在交付範圍內。

**影響**：§2 的範例 JSON 成為**事實上的規範**——實作方只能照範例逆推結構，
無法機器驗證，欄位必填性、型態、enum 全部靠人眼比對。本次實作即以範例為準。

**建議**：補上 Diagnosis 的 draft-07 schema 於附錄，與 L1 同格式；
或將 `api/schemas.py` 納入交付。文件不應引用交付範圍外的檔案。

## ② `severity` 型態在 §1 與 §2 不一致

**現況**：§1 L1 Inference 的 `severity` 是**字串**
（`"watch"`，enum `normal/watch/warning/critical`，L57、L293）；
§2 Diagnosis 的 `severity` 是**巢狀物件**
（`{label, action, hmi_color}`，L86–90）。

**影響**：訂閱兩種訊息的前端必須對同名欄位做型態分支，容易誤用。

**建議**：統一為巢狀物件（資訊量較大、可直接驅動 HMI），
L1 保留 `severity.label` 供輕量訂閱者使用；或明確更名以區分兩者。
**本次實作依裁決採 §2 巢狀格式。**

## ③ 情境 18 的 root cause 特徵表依實證更新，並新增「激勵條件前提」欄

**現況**（v1.0 §2 表格 L130）：

| Scenario | 可能根因 | 對應驅動器參數 |
|---|---|---|
| **18 Ball Screw** | `BL_DeadZone`, `BL_ReversalErr`, `STIFF_Slope`, `FR_Coulomb` | P2-60, P2-62, P1-40, P1-50, P2-80 |

**實證結果**（test 800 段，只評估一次）：

| 特徵 | 實證狀態 | 依據 |
|---|---|---|
| `BL_DeadZone` | **結構性不可估計** | 階梯定位工況下指令為分段常數，死區期間指令位移恆為 0；10/20/40 ms 窗長敏感度給出同一結果，證明非調參問題（Phase 1） |
| `BL_ReversalErr` | **無訊號** | within-noisy ρ=0.075（p=0.067，`_zc`）／ρ=−0.035（`_cmd`）（Phase 2） |
| ~~`STIFF_Slope`~~ | **名實不符，應改名** | `dsp_analytics.force_displacement_slope` 實作為 `polyfit(fe, pos)`，**完全未用到 torque**，非任何意義下的剛性。原輸出應改名 `PosFE_Slope` 並降為探索性 |
| **`STIFF_TorqueSlope`**（新增） | **✅ 成立** | `polyfit(FE, torque)[0]`，依規格本意實作；within-noisy ρ=−0.496（p<0.001），LO vs HI AUC 0.161（等效 0.839） |
| **`FR_Coulomb`** | **✅ 成立** | within-noisy ρ=0.794（p<0.001），LO vs HI AUC 0.983 |

**建議新增欄位**：每個特徵標註**激勵條件前提**，例如——

| 特徵 | 激勵條件前提 |
|---|---|
| `BL_DeadZone` / `BL_HystArea` | **需連續軌跡激勵**（斜坡/正弦）；階梯定位工況不可估計 |
| `BL_ReversalErr` | **需方向反轉事件**；本資料每 run 僅 0–3 次，12% 的 run 為 0 |
| `FR_Coulomb` | 需低速帶樣本（`\|v\| < 100`）|
| `STIFF_TorqueSlope` | 需 FE 有足夠變異 |

**理由**：特徵可用性不只取決於感測器與訊號，還取決於**工況是否提供該特徵所需的激勵**。
合約若不記載此前提，實作方會在不適用的工況下取得看似有效的預設值
（`dead_zone_width` 無事件時回傳常數 `0.05`——正是這種靜默失效）。

## ④ `severity.action` 未列舉合法值

**現況**：§2 範例只出現 `"adjust_parameters"`；§1 的前端使用方式（L64–68）
暗示 critical 應為「降速運行 + 通知工程師」，但未給對應的 action 字串。

**影響**：實作方各自命名，PLC/HMI 端無法以 enum 分派。

**建議**：列舉 enum，例如
`none` / `record_trend` / `adjust_parameters` / `reduce_speed_and_notify`，
與 §1 的四級 severity 對應。**本次實作暫用 `reduce_speed_and_notify`**（依 §1 語意命名）。

## ⑤ `data_provenance_warning` 收編進 schema

**現況**：schema 無此欄位。

**建議**：新增選填陣列欄位 `data_provenance_warning: string[]`，
承載模型/基線的已知限制，隨診斷結果一併傳遞至 HMI 與稽核日誌。

**理由**：本次實驗的健康基線取自**零負載**檔案，而異常類取自加載檔案，
`anomaly_score` 的偵測力為**退化與負載條件的混合效應**。這類限制若不隨資料流動，
下游（HMI 顯示、稽核回溯）無從得知指標的適用邊界。
本次實作已產出此欄位作為 schema 外擴充。

## ⑥ hash chain 驗證程式的 `prev_hash` 未比對（功能失效）

**現況**（§4 L191–203）：

```python
def verify_chain(log_path):
    prev_hash = None
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            stored_hash = entry.pop("sha256_hash", "")
            computed = hashlib.sha256(
                json.dumps(entry, sort_keys=True).encode()
            ).hexdigest()
            if stored_hash and computed != stored_hash:
                return False
            prev_hash = stored_hash      # <- 只賦值，從未被讀取比對
    return True
```

`prev_hash` 被賦值但**從未參與任何比較**，且 `entry` 中的 `prev_hash` 欄位
也未與前一筆的 `sha256_hash` 核對。因此此函式只驗證**單筆完整性**，
**完全沒有驗證鏈結**——刪除中間任意筆、或重排順序，皆可通過驗證。

**影響**：IEC 61508 合規稽核所依賴的防篡改保證不成立。

**建議修正**：

```python
def verify_chain(log_path):
    prev_hash = None
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            stored_hash = entry.pop("sha256_hash", "")
            if entry.get("prev_hash") != prev_hash:     # <- 補上鏈結比對
                return False                             # 斷鏈/重排/刪除
            computed = hashlib.sha256(
                json.dumps(entry, sort_keys=True).encode()
            ).hexdigest()
            if stored_hash and computed != stored_hash:
                return False
            prev_hash = stored_hash
    return True
```

（首筆的 `prev_hash` 應約定為 `null` 或固定 genesis 值，一併於文件明定。）

## ⑦ 門檻語意文件化

**現況**：§1 只寫 `is_anomaly = anomaly_score > threshold`，未定義 threshold 的來源與語意。

**建議**：合約明定門檻的產生方式與其語意限制。本次實作的作法可作為範例——

| 級別 | 定義 | 實測值 |
|---|---|---|
| watch | train LN 複合分數 **P90** | 1.7701 |
| warning | train LN 複合分數 **P95** | 1.9437 |
| critical | train LN 複合分數 **P99** | 2.5048 |

- `is_anomaly` 綁定 **P95**，與 severity 分級同源，兩者不可各自漂移。
- **provenance 註記（須跟著門檻走）**：基線分位數取自**零負載 LN**，
  門檻語意含負載條件。
- **構造性說明**：watch 依 P90 構造在健康件上約有 10% 觸發（test LN 實測 8.5%）；
  critical 依 P99 構造期望 1%（test LN 實測 2.5%）。此為定義使然，非誤報率異常。

---

## 附：本次實作與 v1.0 的差異一覽

| 合約 §2 欄位 | 本次實作 | 差異原因 |
|---|---|---|
| `root_cause_ranking` | `STIFF_TorqueSlope`、`FR_Coulomb` 等實際陣容與真實 z-score | 反映實證，不反填範例；`BL_DeadZone` 結構性不可估計，不產出 |
| `drive_adjustments` | 通用參數群 + `advisory_review`，`value: null` | 只出摩擦/剛性路徑；不輸出可直接寫入驅動器的量值 |
| `drive_adjustments` 之 P2-60 | **不產出**，改列於 `not_estimable` | 背隙補償前提為量得到背隙，本工況無反轉激勵 |
| `plc_adjustments` | 空陣列 | 同上，無可信量值可寫入 PLC |
| `fallback` | `chain: ["model_output"]`、`trigger_alert: false` | 離線批次分析，無即時 fallback 鏈語意 |
| `not_estimable` | **新增** | 明示「量不到」與「量到但正常」之區別 |
| `data_provenance_warning` | **新增** | 見 ⑤ |
| `source_run` / `provenance` | **新增** | 可回溯至具體 run 與基線定義 |
