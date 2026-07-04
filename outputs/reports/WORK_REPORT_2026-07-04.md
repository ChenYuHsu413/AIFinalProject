# 工作報告 — 2026-07-04

> **狀態（2026-07-04）**：Live Monitor 軌**資料升級 v4.1 + 大量呈現/分析強化**，並修一個雷達 bug。
> 重點：(1) 產生器物理升級 **v4.1 Enterprise（142 欄、每情境獨立設備 profile）**；(2) 回放雷達外圈固定 1、
> 即時串流「分格 6 圖」切換 + 顯示 `equipment_profile`；(3) v4.1 合成資料**分析報告**（EDA/故障簽章/模型診斷，
> 誠實揭露為何 F1≈1.0）；(4) 真實 PHM **1D-CNN 資料增強實驗**（誠實負面/穩定性結果，訓練模擬器面板）；
> (5) **1D-CNN 特徵萃取動畫教學**（真實激活、健康度切換、偵測器對應標記）；(6) **修雷達 bug**：高負載情境
> 電流/溫度軸在 normal 階段被釘在 1.0。定位/誠實性見
> [`PROJECT_POSITIONING.md`](PROJECT_POSITIONING.md)、[`MONITOR_V4_DATA_ANALYSIS.md`](MONITOR_V4_DATA_ANALYSIS.md)、
> [`../../docs/MODULE_MONITOR_V3_PLAN.md`](../../docs/MODULE_MONITOR_V3_PLAN.md)。

## 1. 資料升級 v4.1（Vendored 生成器）

- `src/monitor/servo_v3_generator.py` 物理升級至 **Servo AI Dataset v4.1 Enterprise**（檔名保留 v3）：
  每情境獨立 `equipment_profile`/`operation_profile` 動力學、**142 欄**（補回 `encoder_error_count`/
  `igbt_temp_c`/`following_error_abs_pulse`/`bearing_bsf,ftf_amp`/`resonance_amp`）；保留 `stage_fracs`
  抖動與 `SCENARIOS`/CLI 契約。`schema.py` 編碼器軸改用真正 `encoder_error_count`。
- 重建 30 回放包 + 重訓模型 + eval：留出 **F1≈1.00、中位提前量 5.0s、特徵維度 120**。
- 修好 `servo_v3_streaming_generator.py` 壞掉的 import（接上 vendored v4.1，可生成千萬列分片）。

## 2. Monitor 呈現強化

- **回放雷達**：`PolarRadiusAxis domain=[0,1]`，外圈固定嚴重度 1（不再隨資料自動縮放）。
- **即時串流**：新增「**分格 6 圖 / 全部疊圖 / 單一子系統**」切換；狀態列顯示 v4 `equipment_profile`。

## 3. v4.1 合成資料分析（誠實）

- `src/monitor/analyze_v4.py` → [`MONITOR_V4_DATA_ANALYSIS.md`](MONITOR_V4_DATA_ANALYSIS.md) + 6 張圖：
  EDA（階段演進/相關/缺失）、**故障簽章熱圖**（29×6，物理對應合理）、PCA 可分性、
  **模型診斷**（單一特徵 `vibration_kurtosis` AUC=1.0、ECDF/同斜坡圖）——量化說明**為何 F1≈1.0**
  是合成可分性（標籤洩漏），非基準難度、加資料不會變難。
- 順修一個 bug：`radar_severity` 改 `np.fmax`，避免 scenario_06 訊號遺失的 by-design NaN 汙染編碼器軸。

## 4. 真實 PHM 1D-CNN 資料增強實驗（誠實結果）

- `src/models/servo_cnn_augment.py`：對真實波形窗做 **train-only** 增強（jitter/scaling/magnitude-warp），
  5 seeds 比較 baseline / 激進 / 溫和+類別平衡（LO 65→80）。**test 不增強、train-only 標準化、依檔切分（無洩漏）、
  所有配置並列未挑選。**
- **結論（誠實）**：macro-F1 **無穩健提升**（0.655→0.633，落在 seed 變異內），但**溫和+類別平衡降低 seed 變異
  （std 0.049→0.036）**、模型更穩定。呈現於訓練模擬器「資料增強實驗」面板（`/servo/augment_results`）。

## 5. 1D-CNN 特徵萃取動畫教學

- `src/models/servo_cnn_featmap.py` 抽**真實**第一層卷積激活（4 健康等級 LN/LO/MED/HI）→ `web/src/lib/cnnFeatmap.ts`。
- `web/src/components/servo/CnnFeatureSlider.tsx`：canvas 動畫——卷積核在真實波形滑動、6 偵測器特徵圖逐格點亮、
  **健康度切換**（退化越重亮章越多，實測亮格 70→121）、波形上 **①~⑥ 標偵測器最活躍位置** + 中文樣式描述、
  白話「圖章」比喻。放在訓練模擬器折疊面板。

## 6. 修 bug：雷達電流/溫度軸在 normal 階段被釘在 1.0

- **症狀**：即時串流切標籤/暫停播放時，電流（黃線）有時固定為 1.0。
- **根因**：v4.1 每情境設備負載不同（load 15–80），但雷達仍以 **scenario_01**（低負載）為共用基準——
  高負載情境（over_current/torque/brake，load 70–80）的 normal 階段電流遠高於 scenario_01 → z-score 爆表 clip 1.0。
- **修法**：新增 `schema.scenario_baseline(df)`，雷達改用**各情境自身 normal 階段**為基準；
  `build_servo_v3` / `live_stream` 同步。
- **驗證**：修後各情境 current@normal ~0.14（原高負載情境 =1.0），故障軸仍正確升高（over_current alarm=1.0、
  vibration/bearing 不受影響）；殘餘僅 1/4500 幀雜訊尖峰（非持續釘死）。模型/F1 不受影響（雷達為顯示層）。

## 7. 部署

- GitHub main 全程同步。**HF Space** 重新部署至 v4.1 資料 + 本次雷達修正（見 `DEPLOYMENT.md` §9.1）；
  前端純前端變更隨 push 由 **Vercel 自動 redeploy**。
