# AI SERVO Analysis Engine — 資料輸出規格書

> **給前端團隊（Stage 1-2 Dashboard / HMI）與後端團隊（Stage 8-9 日誌 / 資料庫）的資料合約**
>
> 分析引擎（Stage 3-7）產出以下四種資料，前端/後端各自訂閱需要的部分。

---

## 目錄

1. [即時推理結果（L1）](#1-即時推理結果-l1)
2. [診斷與回饋映射（Stage 7）](#2-診斷與回饋映射-stage-7)
3. [Shadow Mode 結果（Stage 6）](#3-shadow-mode-結果-stage-6)
4. [合規稽核日誌（Stage 9）](#4-合規稽核日誌-stage-9)
5. [傳輸方式](#5-傳輸方式)
6. [附錄：完整 JSON Schema](#6-附錄完整-json-schema)

---

## 1. 即時推理結果（L1）

**用途：** HMI 顯示即時異常分數、Dashboard 趨勢圖、PLC 控制模式切換

**觸發時機：** 每 0.02ms（50kHz）或降採樣後每 1ms

### 輸出格式

```json
{
  "type": "inference",
  "scenario_id": "01_Pick_and_Place",
  "timestamp": "2026-07-17T14:30:00.123Z",
  "anomaly_score": 0.42,
  "is_anomaly": false,
  "severity": "watch",
  "latency_us": 312,
  "sample": {
    "fe": -10.62,
    "vel": 0.0015,
    "acc": -0.0032,
    "torque": 0.48,
    "id": 0.012,
    "iq": 0.51
  }
}
```

### 欄位說明

| 欄位 | 型態 | 範圍 | 說明 |
|------|------|------|------|
| `type` | string | `inference` | 固定值，供前端過濾 |
| `scenario_id` | string | `01_Pick_and_Place` / `18_Ball_Screw` / `34_Rotor_Demagnetization` | 當前工況 |
| `timestamp` | string | ISO 8601 | UTC 時間 |
| `anomaly_score` | float | 0 ~ N (0=健康) | 偏離健康 baseline 的程度 |
| `is_anomaly` | bool | true/false | `anomaly_score > threshold` |
| `severity` | string | `normal` / `watch` / `warning` / `critical` | 對應 HMI 顏色：綠/黃/橙/紅 |
| `latency_us` | int | 0~1000 | 推理延遲（微秒），>1000 表示 L1 超時 |
| `sample` | object | — | 原始 6 個即時值（可選，前端需要才送） |

### 前端使用方式

```
severity = "normal"   → HMI 綠色燈號，無動作
severity = "watch"    → HMI 黃色燈號，記錄趨勢
severity = "warning"  → HMI 橙色燈號，觸發診斷
severity = "critical" → HMI 紅色燈號，降速運行 + 通知工程師
```

---

## 2. 診斷與回饋映射（Stage 7）

**用途：** HMI 顯示根因排名 + 建議調整參數、PLC 寫入驅動器參數、機械保養建議

**觸發時機：** 當 `severity >= "warning"` 時自動觸發，或前端手動呼叫

### 輸出格式

```json
{
  "type": "diagnosis",
  "scenario_id": "18_Ball_Screw",
  "timestamp": "2026-07-17T14:30:05.000Z",
  "anomaly_score": 0.73,
  "severity": {
    "label": "warning",
    "action": "adjust_parameters",
    "hmi_color": "orange"
  },
  "root_cause_ranking": [
    {"rank": 1, "feature": "BL_DeadZone", "z_score": 4.2, "meaning": "方向反轉時無響應區變大 — 背隙增加"},
    {"rank": 2, "feature": "STIFF_Slope", "z_score": 3.5, "meaning": "剛性下降（torque/position 斜率降低）"},
    {"rank": 3, "feature": "FR_Coulomb", "z_score": 2.8, "meaning": "庫倫摩擦力增加"}
  ],
  "drive_adjustments": [
    {"param": "P2-60", "action": "set_from_feature_value", "value": 0.12, "unit": "mm", "label": "Backlash Compensation"},
    {"param": "P1-40", "action": "increase_pct", "value": 15, "label": "Position Loop Gain"},
    {"param": "P1-50", "action": "increase_pct", "value": 10, "label": "Bandwidth"}
  ],
  "plc_adjustments": [
    {"address": "D2000", "action": "set_from_feature_value", "value": 0.12, "label": "Backlash Comp Pulse"},
    {"address": "D2020", "action": "decrease_pct", "value": 20, "label": "Torque Limit"}
  ],
  "mechanical_checks": [
    "調整螺帽預壓、檢查滾珠螺桿磨損",
    "潤滑螺桿、檢查支撐軸承預壓"
  ],
  "hmi_display": {
    "color": "orange",
    "alarm_lines": [
      "BL_DeadZone (z=4.2): 方向反轉時無響應區變大",
      "STIFF_Slope (z=3.5): 剛性下降"
    ],
    "action": "adjust_parameters"
  },
  "fallback": {
    "level": "warning",
    "chain": ["model_output", "previous_value"],
    "trigger_alert": false
  }
}
```

### 各 Scenario 的 root cause 特徵

| Scenario | 可能根因（依 z_score 排序） | 對應驅動器參數 | 對應 PLC 位址 |
|----------|--------------------------|---------------|--------------|
| **01 Pick & Place** | FE_RMS, FE_Peak, Pos_SettlingTime, FE_MainFreq, Vel_Std | P1-40, P1-42, P1-50, P2-22, P2-10 | D1300, D1310, D1320, D1330 |
| **18 Ball Screw** | BL_DeadZone, BL_ReversalErr, STIFF_Slope, FR_Coulomb | P2-60, P2-62, P1-40, P1-50, P2-80 | D2000, D2010, D2020, D2030 |
| **34 Rotor Demag** | CS_THD, DQ_IdShift, TR_RippleFactor, Current_RMS | P3-10, P3-12, P3-20, P3-30, P3-40 | D3000, D3010, D3020, D3030 |

---

## 3. Shadow Mode 結果（Stage 6）

**用途：** 後端記錄模型版本切換、Dashboard 顯示 A/B test 進度

**觸發時機：** L3 重訓完成後自動執行

### 輸出格式

```json
{
  "type": "shadow_result",
  "scenario_id": "01_Pick_and_Place",
  "timestamp": "2026-07-17T18:00:00.000Z",
  "new_model_version": "20260717_180000",
  "old_model_version": "20260717_140000",
  "new_model_rmse": 0.023,
  "old_model_rmse": 0.031,
  "improvement_pct": 25.8,
  "shadow_duration_sec": 600,
  "passed": true,
  "deployment_decision": "deploy_new_model"
}
```

---

## 4. 合規稽核日誌（Stage 9）

**用途：** IEC 61508 合規、事後稽核、故障追蹤

**儲存方式：** Append-only JSON lines 檔案 + SHA256 hash chain
**路徑：** `data/compliance_logs/audit_YYYY-MM-DD.jsonl`

### 每筆日誌格式

```json
{
  "event_type": "inference",
  "timestamp": "2026-07-17T14:30:00.123Z",
  "scenario_id": "01_Pick_and_Place",
  "model_version": "l1_20260717",
  "anomaly_score": 0.42,
  "severity": "watch",
  "action_taken": "record_trend",
  "duration_ms": 0.312,
  "fallback_triggered": false,
  "prev_hash": "a1b2c3d4e5f6...",
  "sha256_hash": "f6e5d4c3b2a1..."
}
```

### 日誌 chain 驗證方式（後端負責）

```python
import hashlib, json

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
                return False  # 資料被篡改
            prev_hash = stored_hash
    return True
```

### 事件類型列表

| event_type | 觸發時機 | 記錄內容 |
|-----------|---------|---------|
| `inference` | 每次 L1 推理 | anomaly_score, latency, severity |
| `diagnosis` | 每次 Stage 7 診斷 | top_cause, drive_adjustments |
| `finetune` | 每次 L2 微調觸發 | rolling stat 更新 |
| `retrain` | 每次 L3 重訓完成 | 模型版本, training duration |
| `shadow` | 每次 Shadow A/B test | new/old model RMSE, passed/failed |
| `fallback` | fallback 鏈被觸發 | fallback level, chain step |
| `model_switch` | 模型版本切換 | old→new version, reason |
| `alert` | severity="critical" | anomaly_score, root cause |

---

## 5. 傳輸方式

分析引擎提供三種輸出方式，前端/後端團隊選一種即可：

### 方式 A：Shared JSON Directory（最簡單，推薦）

分析引擎每產生一筆結果就寫入 `output/` 目錄：

```
output/
├── inference/          ← 即時推理結果
│   ├── 2026-07-17/
│   │   ├── 14_30_00_123.json
│   │   └── 14_30_00_456.json
├── diagnosis/          ← 診斷結果
│   └── 2026-07-17/
└── shadow/             ← Shadow 結果
    └── 2026-07-17/
```

前端用 `fs.watch` 或 `inotify` 監聽目錄變化；後端定期搬檔案入 DB。

### 方式 B：Redis Pub/Sub（即時需求高）

| Channel | 內容 | 適合 |
|---------|------|------|
| `ai_servo:inference` | L1 即時結果 | 前端即時顯示 |
| `ai_servo:diagnosis` | Stage 7 診斷 | HMI 警報顯示 |
| `ai_servo:shadow` | Shadow 結果 | 後端模型管理 |
| `ai_servo:compliance` | 合規日誌（也可讀 file） | 後端稽核系統 |

```python
# 分析引擎發送
import redis
r = redis.Redis()
r.publish("ai_servo:inference", json.dumps(data))

# 前端訂閱
import asyncio
import aioredis
pubsub = aioredis.subscribe("ai_servo:inference")
async for msg in pubsub:
    update_dashboard(json.loads(msg))
```

### 方式 C：REST API（FastAPI，已實作）

分析引擎內建 FastAPI server，提供前述所有格式的 HTTP 端點：
- `POST /api/v1/predict` → L1 結果
- `POST /api/v1/diagnose` → 診斷結果
- `GET /api/v1/logs` → 合規日誌

前後端團隊直接 HTTP 呼叫即可。

---

## 6. 附錄：完整 JSON Schema

### L1 Inference

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "L1Inference",
  "type": "object",
  "required": ["type", "scenario_id", "timestamp", "anomaly_score", "is_anomaly", "severity", "latency_us"],
  "properties": {
    "type": {"type": "string", "enum": ["inference"]},
    "scenario_id": {"type": "string", "enum": ["01_Pick_and_Place", "18_Ball_Screw", "34_Rotor_Demagnetization"]},
    "timestamp": {"type": "string", "format": "date-time"},
    "anomaly_score": {"type": "number", "minimum": 0},
    "is_anomaly": {"type": "boolean"},
    "severity": {"type": "string", "enum": ["normal", "watch", "warning", "critical"]},
    "latency_us": {"type": "integer", "minimum": 0, "maximum": 10000},
    "sample": {
      "type": "object",
      "properties": {
        "fe": {"type": "number"},
        "vel": {"type": "number"},
        "acc": {"type": "number"},
        "torque": {"type": "number"},
        "id": {"type": "number"},
        "iq": {"type": "number"}
      }
    }
  }
}
```

### Diagnosis

完整 schema 見 `api/schemas.py` 中的 `DiagnosisResponse` class。

---

> **版本：** v1.0 | **最後更新：** 2026-07-17
>
> 有任何欄位異動會更新此文件並標記版本號。前端/後端團隊請以此文件為準進行開發。
