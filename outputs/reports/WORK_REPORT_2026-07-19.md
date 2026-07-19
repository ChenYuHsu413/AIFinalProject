# 工作報告 — 2026-07-19

> **狀態（2026-07-19）**：**情境 18（滾珠絲槓）× 真實 FMCRD 實驗線 Phase 0–4 全數完成並收官**。
> 重點：(1) 分支清理與 main 同步（清 8 個殘留分支）；(2) **Phase 0 前置檢核**——揪出
> `dsp_analytics` 三項規格與實作落差；(3) **Phase 1 特徵表**（train 665 段）+ 逐位元等價測試；
> (4) **Phase 2 test 評估（只跑一次）**——摩擦/剛性通過預註冊複現檢驗、背隙族確認無訊號；
> (5) **Phase 3 消融**——兩變體皆無提升，如實報告；(6) **Phase 4 diagnosis JSON + 合約 v1.1 提案七條**
> （含 hash chain 驗證失效之功能缺陷）。全部為新增檔案，**repo 既有模組零改動**，199 tests 全過。
> 詳見 [`../s18_experiment/`](../s18_experiment/)、
> [`../../docs/s18_experiment_design.md`](../../docs/s18_experiment_design.md)、
> [`../../docs/contracts/`](../../docs/contracts/)。

## 0. 分支清理（暖身）

- `git fetch --prune` 後發現遠端已隨 PR 合併自動刪除 13 個分支。以 `git cherry`（patch-id）
  逐一比對本地 8 個殘留分支，確認內容全部已在 main（含 3 個 `claude/*`）後刪除。
- 過程更正一次判斷：`claude/wizardly-lewin` 的 `count==0` guard 一度被 `git cherry` 標為未合併，
  實際上 main 已於同日以不同寫法（迴圈 vs 展開）獨立實作同一修正，patch-id 因排版不同而對不上。
- 本地 main 由 `e000a28` 快轉至 `3a6627a`。

## 1. Phase 0 — 前置檢核（設計書 §3）

抽 3 個 LN run，`src/s18/phase0_precheck.py`：

- **取樣率 50,000 Hz**（dt=20 µs 均勻）、299,951 samples / 5.999 s per run，與設計書一致。
- **方向反轉存在**（顯著反轉 7 次/run），BL 族不需啟動降級方案。
- 揪出**三項 `dsp_analytics` 規格與實作落差**：
  1. `dead_zone_width` 窗參數寫死 20 samples（隱含 1 kHz 假設），50 kHz 下僅 0.4 ms → 恆回傳預設常數 `0.05`；
  2. `force_displacement_slope` **名實不符**——實作為 `polyfit(fe, pos)`，**完全未用到 torque**；
  3. `stribeck_friction_parameters` 高速帶門檻 `|v|>500` 對本資料（|v|≤183）**命中 0 筆** → 黏滯項恆為 `0.001`；
  4. `del_pos` 語意誤判——實為「相對 run 起點的**指令**位移」，非逐點差分（設計書 §1 鐵律 4 後半句作廢）。

## 2. Phase 1 — 特徵表與等價測試

- `s18_features.py`：因原函數簽名無法支援取樣率換算與事件型別篩選，採**參數化重實作**，
  並以**逐位元等價測試**錨定（`tests/test_s18_features.py`，涵蓋 3 個全長 LN run）——
  原參數設定下必須與 `dsp_analytics` bit-identical，任何行為漂移即測試失敗。`dsp_analytics.py` 原檔不動。
- **兩遍掃描架構**：`build_s18_features.py` 只讀訊號欄、`build_s18_labels.py` 只讀 DV/ylabel，
  分析階段才 join——防洩漏鐵律 1 的**結構性**保證（非靠自律）。
- **零事件輸出 NaN，絕不落回預設常數**。
- train 665 段（LN 200 / **LO 65** / MED 200 / HI 200）跑通。依預註冊出口標兩個 N/A：
  `BL_DeadZone`（結構性——階梯指令下死區位移恆為 0，10/20/40 ms 敏感度同結果）、
  `FR_Viscous`（穩定性判準未過，且係數為負而黏滯係數物理上須非負）。
- 用資料檢驗事件數的跨等級可比性：`n_cmd_reversals` 一致（1.40–1.53），
  但 `n_zero_crossings` 在 LO 異常偏高（9.82±9.10，max 52）——組成效應確實存在。

## 3. Phase 2 — test 評估（800 段，只跑一次）

- **預註冊複現檢驗全數通過**：`FR_Coulomb` ρ 0.784→**0.768**、AUC(HI) 0.985→**0.991**；
  `STIFF_TorqueSlope` ρ −0.562→**−0.499**。未觸發過擬合警示。
- **主要推論軌 within-noisy**（LO/MED/HI，負載條件一致，無混淆）：
  `FR_Coulomb` ρ=0.794、LO vs HI AUC **0.983**；`STIFF_TorqueSlope` ρ=−0.496、AUC 0.161（等效 0.839）。
- **背隙族四項全部無訊號**。對照組 `FE_RMS`/`FE_Max` 全程無訊號（|ρ|≤0.02），
  證明獲勝特徵**非追隨誤差之換皮**。
- **within-noisy 分軌當場識破一個假陽性**：`BL_ReversalErr_zc` 在全類別檢定 p=0.0005 看似鐵證，
  分軌後 ρ=0.075（p=0.067）——訊號全來自 LN（零負載）與其餘三類的對比。
  值得注意的是**偏相關控制 `n_zero_crossings` 後仍為 0.237（p<0.001），並未拆穿它**；
  真正拆穿的是分軌檢定。教訓：混淆控制要控制資料結構裡真實存在的變因（負載條件），
  而非事前猜到的變因（事件數）。
- **發現負載混淆**：LN 僅存在於零負載檔案（`*_load0`），故所有 torque 基特徵與 LN 錨定指標
  皆含退化與負載的混合效應。推論框架據此重排：主要推論改 within-noisy，LN 錨定指標降為
  「操作性偵測指標」。
- 觸發率 LN 12.5% / LO 12.0% / MED 33.0% / **HI 90.0%**。
- **圖 1 更換主圖**：原設計書承諾的「遲滯迴圈變胖」經證偽（階梯指令下 demand–actual 為階梯路徑
  非迴圈），改為 **Stribeck 平面**（torque–velocity，低速帶標色），畫資料實際證明的東西；
  原圖降為附錄，作為「迴圈類特徵為何不適用」的直接證據。

## 4. Phase 3 — 增量資訊消融（設計書 §4.5）

沿用 `train_servo.run(out_dir=)` + `validation_gate.run_gate`，**repo 既有模組零改動**
（變體以執行期 config 覆寫 + `FEATURE_SETS` 僅新增鍵實現）。模型家族釘死 LR（依設計書基準）。

| 變體 | 維度 | test macro-F1 | Δ | DV R² | 閘門 |
|---|---|---|---|---|---|
| base（repo full） | 21 | **0.8187** | — | 0.9444 | PASS |
| A（全部候選） | 32 | 0.7581 | −0.0606 | 0.9298 | FAIL（契約不符） |
| B（train 診斷通過者） | 23 | 0.8124 | −0.0063 | 0.9415 | FAIL（契約不符） |

**結論（依預註冊）**：兩變體皆無提升 → s18 特徵的資訊**已被既有 21 維涵蓋**；
其價值在**物理可解釋性與根因排名**，不在新增判別力。閘門攔下契約不符為正確行為，
建議不變更特徵契約。

**過程中自捕兩個錯誤**（均如實記錄）：

1. **join key 錯位造成自製洩漏**——repo 表的 `run_index` 是排序後重編的全域列號而非原始 run 編號，
   直接對接導致 665 列（全部 train）配不上 → 中位數填補 → `_isna` 指示欄**恰好等於「是否為 train」**，
   test 分布完全錯位（A=0.479、B=0.337）。修正後加兩道強制驗證：join 後 `ylabel` 全對、
   且 `FE_RMS` 必須恆等於 `position_error_rms`（實測最大偏差 1.78e-15）。
2. **模型家族未釘死**——首次讓 CV 自由搜尋，變體 B 選到 MLP，比較變成「LR vs MLP」。
   釘回 LR 是**回歸設計書原文的協定修正**，非依結果調整。

兩個錯誤都由「base 必須複現 0.819」這個錨點暴露。**可複現的已知基準是接線檢查的最強工具。**

## 5. Phase 4 — diagnosis JSON 與合約 v1.1 提案

- **severity 門檻**取 train LN 複合分數經驗分位數：watch **P90=1.7701** / warning **P95=1.9437**
  / critical **P99=2.5048**；`is_anomaly` 綁 P95 與 Phase 2 觸發率同源。
  選同源分位數而非倍數法的理由：答辯只需辯護一次。
- **diagnosis JSON 兩筆**：test run 708（HI，critical，`STIFF_TorqueSlope` **z=−11.6**）
  + run 429（MED，warning）。內容一律反映實證：`BL_DeadZone` 不出現於根因、
  背隙補償 P2-60 不產出（改列 `not_estimable`）、`drive_adjustments` 為通用參數群 + advisory。
- **新增 `not_estimable` 欄位**（超出合約 schema 的設計貢獻）：明示「量不到」與「量到但正常」的差別。
  沒有這個第三態，`BL_DeadZone` 只能在「假裝有值」與「無聲消失」之間二選一——
  前者不誠實，後者讓維護人員誤以為背隙已排除。
- **`DATA_CONTRACT.md` verbatim 收錄**進 `docs/contracts/DATA_CONTRACT_v1.0.md`
  （sha256 `cf830718a7dbb990…`，10,108 bytes，`cp` 後逐位元比對）。
  收進版控的理由：合約先前只存在於交付附件，導致實作時「找不到檔案」。
- **v1.1 修訂建議書七條**（獨立提案檔，不改 v1.0），其中
  **⑥ hash chain 驗證失效**屬功能性缺陷：`verify_chain` 的 `prev_hash` 只賦值從未比對，
  故僅驗單筆完整性、**完全未驗鏈結**——刪除中間任意筆或重排順序皆可通過，
  影響 IEC 61508 合規主張。已附修正碼。

## 6. 情境 18 最終定位

> **情境 18：機械傳動層——真實資料部分驗證（摩擦/剛性 ✅；背隙：激勵條件不支持 ⚠️）**

- **成立**：`FR_Coulomb`、`STIFF_TorqueSlope`（within-noisy，p<0.001），HI 偵測率 90.0%。
- **不成立**：背隙族——歸因為**激勵條件不支持**而非訊號微弱，由**四個機制獨立的證據三角化**
  （窗長敏感度、遲滯迴圈不存在、指令換向僅 0–3 次、within-noisy 無訊號）。
- **未突破**：LN/LO 早期邊界（AUC≤0.556，四次獨立確認），與 repo E1–E5 之類別不平衡結論互證。

此結果把「背隙特徵需反轉/連續激勵方可觀測」由**設計假設**升級為**實證命題**，
並回饋 40 情境框架：**激勵條件應納入情境可行性分級維度**。

## 7. Git

四次推送至 main（`3a6627a` → `64f76cf`），CI 與 HF Space 部署全綠：

| commit | 內容 |
|---|---|
| `87145aa` | `feat(s18)` 特徵管線與分析工具 |
| `b1707b2` | `chore(s18)` 參數與預註冊條文 |
| `84c4661` | `docs(s18)` 設計書劃線修訂版 |
| `55e3040` | `chore(s18)` 實驗產出 |
| `64f76cf` | `feat(s18)` Phase 4 — diagnosis JSON、合約收錄與 v1.1 提案 |

順修 `.gitignore` 一個經典陷阱：**行內註解無效**（整行含 `#` 被當 pattern），
三條規則靜默失效——與 join key 錯位同族的「看起來生效實際沒生效」問題。

## 8. 未竟事項

1. **FMCRD 資料集出處與授權待確認（最高優先）**。`docs/DATA_PROVENANCE.md` 有 CRC32 指紋，
   但**無來源 URL、無授權條款、無引用格式**；repo 為 PUBLIC + MIT 且已公開四份衍生特徵表
   （早於本實驗）。建議新增「來源與授權」節，四資料集各記出處/條款/引用/再散布狀態，
   Kaggle 鏡像與原始發布方條款分開記。
2. **DATA_CONTRACT v1.1 待 zip 專案維護方採納**，⑥ 建議優先。
3. `FR_Viscous` 的 regime 切分留待未來重新定義後檢驗（本次依預註冊標 N/A，不翻案）。
4. **背隙路徑需另尋工況**（需連續軌跡激勵或足量方向反轉的資料集）。
5. contract-matched 驗證閘門未執行（需重建 smoke 固件）。
6. **情境 34 縮小版**（選做）：THD/DQ 特徵已在 `dsp_analytics`，管線與分析套件已建好，
   約為 18 的三分之一工作量；不做則維持「方法就緒 + 資料缺口如實標註」。
