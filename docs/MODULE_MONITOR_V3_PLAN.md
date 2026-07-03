# Live Monitor v3 規劃 — 即時監控雷達（合成 demo 軌）

> **定位**：一條**獨立的合成資料 demo 軌**，把 AI 生成的 *Servo AI Dataset v3.0*
> 逐幀回放成「即時監控雷達」，用**六角子系統雷達**一眼看出哪個子系統在惡化，並以
> **早期預警模型**在真正告警前先亮燈。**不是**真實產線遙測，**與真實 PHM 伺服主線
> （模組 Servo）分開、不取代之**，也不佔 A/B/B+/C 的模組字母（避免暗示研究等級）。
> 相關：真實主線見 [`MODULE_SERVO_PLAN.md`](MODULE_SERVO_PLAN.md)、資料溯源見
> [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md)。
>
> **狀態（2026-07-03）**：**建置腳本 → 後端 → 前端全部完成並跑通**；並已升級為
> **產生器單一源 + 即時感測器串流**。
> - **Vendored 產生器**（`src/monitor/servo_v3_generator.py` + streaming 版）為**唯一資料源**，
>   一切從程式重現，**不再需要 480MB 原始檔**。統一為產生器的 **100 欄 schema**。
> - **回放模式**：30 情境 × 600 frames（40 Hz）回放包；早期預警（留出情境）**F1 = 1.00**、
>   **中位提前量 = 5.0s**；六角雷達 argmax 物理對應合理。
> - **即時模式（新）**：後端 **SSE `/monitor/stream`** 用產生器**即時生成 + 即時推論**，
>   **隨機注入** 30 種故障（每條連線各自隨機、不連續重複）；前端 `/monitor` 加「即時 ⇄ 回放」切換，即時模式有 ● LIVE、
>   往左捲動的即時趨勢、事件串流（注入→模型預警→告警），驗證通過（t+ 時鐘持續前進、跨情境循環）。

---

## 0. 一句話定位

> 用 AI 合成的伺服 PLC/Drive 訊號，做一個「會動的監控台」：雷達顯示子系統健康、
> 模型在退化早期就預警、時間軸標出「模型示警 vs 真實告警」的提前量。**強項**是即時視覺
> 與提前量概念；**弱點**是資料為合成、退化平滑易分離，故指標偏高，非困難基準。

---

## 1. 資料事實

| 項目 | 值 |
| --- | --- |
| 來源 | **Vendored 產生器**（`src/monitor/servo_v3_generator.py`，seed 42）—— *Servo AI Dataset v3.0* 的公開產生器 |
| 規模 | **30 情境 × 15,000 列（15 秒 @ 1000 Hz）= 45 萬列**，**100 欄**（產生器 schema）|
| 情境 | 1 healthy baseline + 29 故障，每情境一種故障、完整走 `normal → early_degradation → warning → alarm → trip` |
| 故障類別 | 13 類（TEMPERATURE / ENCODER / MECHANICAL / CURRENT / MOTION / POWER / COMMUNICATION / CONTROL / SAFETY / VIBRATION / …），部分類別**僅單一情境** |
| 內建標籤 | `fault_stage`、`warning/alarm/trip`、`health_index`、`failure_probability`、`rul_sec`、`root_cause`、`maintenance_action` |
| 原始檔依賴 | **無**。一切由 vendored 產生器重現（原始 30 檔企業版 zip 為 138 欄，已不需要）。|

### 誠實性紅線（務必遵守）
- 這是 **AI 合成資料**：欄位含 `line_id=TSMC_ASE_AUTO_LINE_SIM_V3`、`drive_model=Mitsubishi MR-J5-G simulated`、
  alarm code `MITSUBISHI_SIM_*` —— 皆為**模擬/泛化標籤**，README 亦註明不複製任何 OEM/廠商實資料。
  **不得**呈現為真實產線遙測，**不得**凌駕真實 PHM 主線。
- **即時串流是「模擬感測器」**：資料由產生器即時產生、非真實硬體 sensor。介面（SSE frames）設計成未來可把
  來源換成 ESP32/PLC 而前端不變，但現階段就是模擬。
- **標籤是注入的**：`fault_stage / health / rul / failure_probability` 皆為一條 severity 斜坡
  （階段轉換比例）的確定性函數，感測欄位也是同一條 severity + 小雜訊 → **這就是模型 F1=1.00 的原因**。
  本軌價值是**即時視覺與流程**，非難度基準；再多資料也不會變難（產生器結構決定）。
- **提前量的變異來自時間抖動**：離線建置（回放包）用**固定**階段比例（53%/73%/88%）→ 中位提前量恆為 5.0s；
  **即時串流則每次注入隨機抖動 warning/alarm/trip 的時間**（`_jittered_stage_fracs`）→ 提前量自然散開（實測 ~3.5–7s）。
  這是為了讓「每次剛好 5.0s」的合成破綻更貼近真實（故障演進本有快慢），仍為合成、非真實硬體量測。
- **公開產生器只有 100 欄**（比企業版 138 欄少 igbt_temp / encoder_error_count / crc / following_error）；
  雷達的編碼器軸改用 `position_error + digital_twin_pos_residual` 等效表示。
- **類別標註為 in-distribution 參考**（單一情境類別無法留出泛化），非泛化宣稱。

---

## 2. 系統設計

單一資料源 = **vendored 產生器**；共用 `schema.py`（雷達軸 + 特徵 + 計算）於離線建置與即時串流。

```
src/monitor/servo_v3_generator.py（vendored，唯一資料源，進 git）
   │
   ├─ (A) 離線回放：① build_servo_v3.py 生成 30 情境 → 降頻 40Hz + 特徵 → 訓 2 模型
   │        ├─ data/processed/servo_v3/scenario_XX.json + manifest.json（進 git）
   │        ├─ outputs/models/monitor_v3_clf.joblib（進 git，即時推論也用）
   │        └─ outputs/metrics/monitor_v3_eval.json
   │      ② 後端 /monitor/scenarios、/monitor/scenario/{id}（讀回放包）
   │      ③ 前端「回放」模式：雷達 + 健康環 + RUL + 告警燈 + 時間軸 + 播放/調速/scrub
   │
   └─ (B) 即時串流：④ live_stream.py 即時生成情境 → 降頻 → 跑模型 → 逐幀
          ⑤ 後端 SSE /monitor/stream（async，隨機注入 30 故障、不連續重複，?speed=）
          ⑥ 前端「即時」模式：EventSource → canvas 捲動趨勢（含**示警→告警 gap 標線 + 提前秒數**）
             + 量條/健康/告警即時 + 事件串流 + **AI 提前告警 KPI**（本次/平均，量化價值）
             + 全頁告警紅閃 + 開始/暫停
```

- **回放包自包含**：模型預測在建置時烤進 JSON。**即時串流**則在後端載入
  `monitor_v3_clf.joblib` 做**即時推論**（故該模型進 git）。
- **降頻/調速**：建置端 40 Hz；回放用 `setInterval`（隱藏分頁不凍結）+ 0.5×–8× + scrub；
  即時串流後端 20 Hz、`?speed=` 調倍率、SSE 用 `asyncio.sleep` 控節奏、情境生成 off-loop（`asyncio.to_thread`）。

### 2.1 六角雷達（透明、非模型）
六軸 = **溫度 / 電流 / 振動 / 編碼器 / 運動追隨 / 通訊**，各軸嚴重度 = 相對 healthy baseline
（scenario_01）的**穩健偏離度**（`|x−μ|/σ/Z_CAP`，flag 類欄位改用絕對偏離），clip 到 0–1。
純資料計算、可解釋，哪角爆紅一眼看出。（`vibration_kurtosis` 因合成資料普遍抬升會汙染雷達，
**排除於雷達軸、僅留作模型特徵**。POWER/CONTROL 無專屬軸，經由下游 current/encoder 反映。）

### 2.2 早期預警模型（ML 貢獻）
- **輸入**：**僅物理遙測**滑動窗特徵（mean/std/max，102 維），**不含**任何標籤衍生欄位。
- **目標**：`warning-or-worse within horizon`（前瞻 2s）→ 模型能在真實 warning flag **之前**亮燈。
- **評估**：**留出整條情境**（每 6 條留 1）→ 跨情境測試；headline 指標為
  **提前量（lead-to-alarm）= 真實 alarm 時間 − 模型示警時間**。
- **疑似原因**：另一顆分類器用**全部情境**訓練（in-distribution 參考），標「哪個子系統」文字；
  視覺上的「哪裡壞」以透明雷達 argmax 為主。

---

## 3. 結果（2026-07-03，`monitor_v3_eval.json`）

| 指標 | 值 |
| --- | --- |
| 早期預警 F1（留出情境，horizon 2s） | **1.000** |
| 中位提前量（預警 → 真實告警） | **5.0 s**（29/30 情境有告警）|
| 雷達 argmax 對應真實故障子系統 | 30/30 物理合理（多數命中，少數映到相鄰子系統如扭矩→電流、軸承磨耗→溫度）|
| 特徵維度 / 回放 frames | 102 / 每情境 600（40 Hz）|
| 即時串流 | 10–20 Hz、隨機注入 30 故障；事件串流呈現「注入→模型預警→告警」（提前 ~5s）|

> 指標偏高源於合成資料易分離（見 §1 誠實性）。本軌定位為 **demo**，非基準比較。

---

## 4. 檔案清單

| 類型 | 路徑 |
| --- | --- |
| 產生器（vendored，唯一源）| `src/monitor/servo_v3_generator.py`、`servo_v3_streaming_generator.py`（可生成百萬~千萬列分片）|
| 共用 schema | `src/monitor/schema.py`（雷達軸 / 特徵 / 計算，離線+即時共用）|
| 離線建置 | `src/monitor/build_servo_v3.py`（`python -m src.monitor.build_servo_v3`）|
| 即時串流 | `src/monitor/live_stream.py` |
| 設定 | `config.yaml::monitor` |
| 後端 | `app/backend/main.py`（`/monitor/scenarios`、`/monitor/scenario/{id}`、`/monitor/stream`）、`services.py`（`monitor_*`）|
| 前端 | `web/src/app/monitor/page.tsx`（回放）、`components/monitor/{LiveView,shared}.tsx`、`lib/nav.ts`、`lib/api.ts` |
| 產物（committed） | `data/processed/servo_v3/*.json`、`outputs/metrics/monitor_v3_eval.json`、`outputs/models/monitor_v3_clf.joblib` |
| 測試 | `tests/test_backend_api.py::test_monitor_*` |

---

## 5. 重建 / 執行

```bash
# 1) 建置回放包 + 模型 + 評估（從 vendored 產生器，無需原始檔）
python -m src.monitor.build_servo_v3            # 可調 --out-hz / --win-ms / --horizon-s / --holdout
# 2) 後端 + 前端
uvicorn app.backend.main:app --port 8000
cd web && NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev   # → /monitor（預設即時模式）
# 3)（可選）用 streaming 產生器擴充大資料
python src/monitor/servo_v3_streaming_generator.py --rows 10000000 --scenario mixed \
    --format parquet --output_dir servo_10M_parquet --chunk_rows 100000 --split_files
```

---

## 6. 未來工作（未做）

- 把即時串流的來源從產生器換成 **ESP32 / PLC 真 sensor**（SSE frame 介面不變）。
- 多軸/多情境**同框機群雷達牆**（目前一次一情境）。
- 用 streaming 產生器 + 多 seed 做真正的深度時序（LSTM/Transformer/AE）與跨軌泛化評估。
- 把此 demo 接到真實主線的 `/servo/fleet`（目前刻意分開、不取代）。
