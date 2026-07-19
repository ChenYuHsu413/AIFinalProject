# 情境 18（滾珠絲槓背隙/磨損）× FMCRD 真實資料 — 實驗設計書〔劃線修訂版〕

> **狀態（2026-07-19）**：本檔為原始設計書 v1.0（2026-07-19）的**劃線修訂版**。
> 原檔位於 repo 之外，未更動；此處保存完整決策軌跡供答辯溯源。
>
> **修訂依據**：Phase 0–3 findings（[Phase 0](../outputs/s18_experiment/PHASE0_FINDINGS.md)、
> [Phase 1](../outputs/s18_experiment/PHASE1_FINDINGS.md)、
> [Phase 2](../outputs/s18_experiment/PHASE2_FINDINGS.md)、
> [Phase 3](../outputs/s18_experiment/PHASE3_FINDINGS.md)）、
> 參數與預註冊條文見 [`config/s18_params.yaml`](../config/s18_params.yaml)。
>
> **標記法**：~~刪除線~~ = 經實證證偽的原始承諾（保留不刪）；
> **▶ 修訂** = 取而代之的實證結論，並標註對應 Phase。

**目的：** 在真實 PHM FMCRD 資料上驗證背隙/剛性/摩擦特徵家族（BL/STIFF/FR）對退化等級的敏感度。
**定位：** 偵測性驗證（特徵是否隨退化單調惡化），不宣稱故障分類。
**版本：** v1.0（原始，2026-07-19）→ 劃線修訂版（2026-07-19，Phase 0–3 完成後）

---

## 0. 一句話描述實驗

> 對 FMCRD 每個 run 計算 8 個背隙/剛性/摩擦物理特徵，只用 train 的 LN（健康）段建基線，
> 然後檢驗：test 上這些特徵是否隨 LN→LO→MED→HI 單調惡化、且與 DV 正相關。

**▶ 修訂（Phase 2）**：實證結果為**部分成立**——摩擦（`FR_Coulomb`）與剛性
（`STIFF_TorqueSlope`）成立；背隙（BL）族**全部不成立**，且歸因為**激勵條件不支持**
而非訊號微弱。另因 LN 僅存在於零負載檔案，主要推論改以 **within-noisy**（LO/MED/HI）為準。

---

## 1. 資料

| 項目 | 內容 |
|---|---|
| 來源 | FMCRD_Data.zip（串流聚合，勿解壓） |
| 切分 | train 665 段建基線、test 800 段只評估一次 |
| 聚合單位 | 一個 run = 6 秒 / 5 階梯定位循環 |

**▶ 修訂（Phase 1）**：train 665 段**非四類均衡**——LN 200 / **LO 65** / MED 200 / HI 200。
LO 恰為 §4.3 指定的關鍵邊界，統計檢定力偏低，所有表格強制標 n、AUC 附 bootstrap CI。

**防洩漏鐵律：**
1. 特徵計算只准用 demand/actual/del_pos/torque/rotor_speed/time —— `DV` 與 `ylabel`
   不得出現在特徵管線的任何位置。**▶ 落實（Phase 1）**：以**兩遍掃描**架構結構性保證——
   `build_s18_features.py` 只讀訊號欄，`build_s18_labels.py` 只讀 DV/ylabel，分析階段才 join。
2. 基線統計只從 train 的 LN 段計算。
3. test 800 段只碰一次。**▶ 落實**：primary variant 於 train 選定並鎖進 config 後才動 test。
4. 速度優先用 `rotor_speed`；~~若需位置微分，統一用 `del_pos/dt`~~。
   **▶ 修訂（Phase 0）**：`del_pos` 實為 `rod_demand_pos − rod_actual_pos(t=0)`，
   即**相對 run 起點的指令位移**，非逐點差分；照字面微分會得到指令速度而非實際速度。
   **後半句作廢，一律用 `rotor_speed`。**

---

## 2. 特徵定義

原案 8 特徵。**▶ 修訂後的最終陣容（Phase 1–2）**：

| # | 原案特徵 | 修訂後狀態 |
|---|---|---|
| 1 | `BL_DeadZone` | ~~主力~~ → **結構性不適用**（Phase 1）。階梯指令下死區期間指令位移恆為 0，10/20/40 ms 窗長敏感度給出同一結果，證明非調參問題 |
| 2 | `BL_ReversalErr` | 雙軌 `_cmd` / `_zc`（Phase 1）；Phase 2 within-noisy 無訊號 |
| 3 | `BL_HystArea` | 保留；Phase 2 無訊號 |
| 4 | `BL_DirFE_Asym` | 保留；Phase 2 無訊號 |
| 5 | ~~`STIFF_Slope`（torque 對位移誤差的斜率）~~ | **▶ 名實不符（Phase 0）**：`dsp_analytics.force_displacement_slope` 實作為 `polyfit(fe, pos)`，**完全未用到 torque**。原輸出改名 `PosFE_Slope` 降為探索性；另新增 `STIFF_TorqueSlope = polyfit(FE, torque)[0]` 依規格本意實作 |
| 6 | `STIFF_ComplStd` | 保留；Phase 2 無訊號 |
| 7 | `FR_Coulomb` | **✅ 通過**（Phase 2） |
| 8 | `FR_Viscous` | **N/A**（Phase 1 穩定性判準未過，且係數為負而黏滯係數物理上須非負）→ 僅探索性描述統計 |

對照組 `FE_RMS` / `FE_Max` 依原案保留，Phase 2 確認全程無訊號（ρ ≤ 0.02），
**證明獲勝特徵非追隨誤差之換皮**。

---

## 3. 前置檢核

原案要求檢核方向反轉存在性、取樣率、單位。

**▶ 結論（Phase 0）**：取樣率 50 kHz（均勻，299,951 samples / 5.999 s per run）；
方向反轉**存在**，BL 族不需啟動降級方案。

**▶ 補充修訂（Phase 1）**：前置檢核的 3 個 LN run **未覆蓋指令輪廓的異質性**——
Phase 0 觀察到「每 run 4 次階梯有升有降」，但全量建表後發現**指令換向僅 0–3 次、
且 12% 的 run 為完全單向指令**。此落差不影響任何裁決（NaN policy 已正確處理單向 run），
但如實記錄為抽樣代表性的限制。

---

## 4. 分析流程

### 4.1 單調性
通過標準：8 特徵中 ≥4 個 p<0.01 且方向符合。
**▶ 結果（Phase 2）**：名目 3/7，其中 `BL_ReversalErr_zc` 經 within-noisy 軌判定為
負載混淆假陽性 → **實質 2/7，未達門檻**。

### 4.2 與 DV 的相關
**▶ 結果**：`FR_Coulomb` ρ=0.778、`STIFF_TorqueSlope` ρ=−0.508（均 p<0.001）。

### 4.3 可分性
原案：若任一 BL 特徵 AUC(LN vs LO) > 0.75，即為實質貢獻。
**▶ 結果**：**未達成**，LN vs LO 最高僅 0.556。全部特徵（含對照組）皆貼近 0.5。

### 4.4 健康基線 + 複合異常分數
**▶ 結果**：觸發率 LN 12.5% / LO 12.0% / MED 33.0% / **HI 90.0%**。
HI 達設計理想值；LN 誤報率高於理想的 10%；LO 與 LN 無法區分。

### 4.5 增量資訊消融
**▶ 結果（Phase 3）**：base(21 維) 0.8187 複現基準；變體 B(23 維) 0.8124（−0.006，持平）、
變體 A(32 維) 0.7581（−0.061，維度噪音）。**結論：其資訊已被既有 21 維涵蓋。**

---

## 5. 產出清單

| 原案產出 | 修訂後狀態 |
|---|---|
| ~~圖 1：LN vs HI 的 demand–actual 遲滯迴圈疊圖。**最直觀的一張**，肉眼可見迴圈變胖 = 背隙~~ | **▶ 經 Phase 0–1 證偽**：此承諾基於**連續軌跡激勵**假設；FMCRD 指令為分段常數階梯（每 run 299,950 個差分僅 4 個非零），demand–actual 平面呈階梯狀路徑而非迴圈。原圖降級為附錄 `figA_phase_plot_test.png`，作為「迴圈類特徵為何不適用」的直接證據 |
| — | **▶ 新主圖**：`fig1_stribeck_test.png`（torque–velocity 平面，低速帶 \|v\|<100 標色），畫的是資料實際證明的東西；副圖 `fig1b_fe_torque_slope_test.png` 視覺化 `STIFF_TorqueSlope` |
| 圖 2：箱型圖 | 依原案產出 |
| 圖 3：特徵 vs DV 散佈 | 依原案產出 |
| 圖 4：z-score 根因排名 | 依原案產出 |
| 表 1 / 表 2 + `metrics_s18.json` | 依原案產出 |

---

## 6. 驗收判定

原案三格出口。**▶ 落點（Phase 2–3）**：第二格與第三格之混合——

- **STIFF/FR 族**：於真實資料驗證成功（within-noisy 軌無負載混淆）。
- **BL 族**：不成立，歸因為**激勵條件不支持**。
- **LN/LO 邊界**：未突破，與 repo E1–E5 之類別不平衡實驗結論互相印證。

**不硬拗**：情境 18 的背隙路徑於本資料集**未獲驗證**；獲得驗證的是摩擦/剛性路徑。

---

## 7. 規格與實作落差（本實驗新增之方法論章節）

`dsp_analytics.py` 的四項函數在原專案中**從未被真實資料檢驗過**，本實驗逐一暴露：

1. `dead_zone_width` 窗參數寫死 20 samples，隱含 1 kHz 假設，在 50 kHz 失效。
2. `force_displacement_slope` 名實不符（未用到 torque）。
3. `stribeck_friction_parameters` 高速帶門檻 `|v|>500` 對本資料（|v| ≤ 183）恆不命中。
4. `del_pos` 語意誤判（非逐點差分）。

因原函數簽名無法支援取樣率換算與事件型別篩選，採**參數化重實作並以逐位元等價測試錨定**
（`tests/test_s18_features.py`，涵蓋 3 個全長 LN run）。該等價測試同時替原函數補上了
它從未有過的測試。

**方法論範例（混淆控制）**：`BL_ReversalErr_zc` 在全類別檢定上 p=0.0005 看似鐵證，
偏相關控制 `n_zero_crossings` 後仍為 0.237（p<0.001）——**偏相關並未拆穿它**。
真正拆穿它的是 within-noisy **分軌檢定**（ρ=0.075, p=0.0665）。教訓：混淆控制不是控制
你猜到的變因（事件數），而是控制資料結構裡真實存在的變因（負載條件）。

---

## 8. 對 40 情境框架的輸入

本實驗把「背隙特徵需反轉/連續激勵方可觀測」由**設計假設**升級為**實證命題**。
**建議：激勵條件應納入情境可行性分級維度** —— 特徵可用性不只取決於感測器與訊號，
還取決於工況是否提供該特徵所需的激勵。
