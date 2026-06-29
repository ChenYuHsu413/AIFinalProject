# 工作報告 — 2026-06-29

> **狀態（2026-06-29）**：完成兩件事——(1) 跨報告/前端**淡化模組 A**、新增「失效模式 × 感測模態」
> 相關性圖、descope CE2/CE3 與模組 B 1D-CNN；(2) 修復 CI py3.12 的 **pandas 3.0.4 段錯誤**並留下除錯記錄。
> PR：[#14](https://github.com/ChenYuHsu413/AIFinalProject/pull/14)（branch `docs/de-emphasize-module-a`）。
> 相關文件：[`../../docs/FINAL_REPORT.md`](../../docs/FINAL_REPORT.md)、[`REPORT_OUTLINE.md`](REPORT_OUTLINE.md)。

## 1. 模組定位重整（淡化 A）

主線 Servo 已是**真實 PHM FMCRD**，四條對照軌的承重下降。依與伺服馬達的貼近程度確立分級：
**Servo（核心）> C（馬達電流 MCSA，最貼近馬達）> B / B+（軸承，機械失效模式）> A（AI4I 合成，最不貼近，僅方法基礎）**。

- **新增圖**：[`../figures/servo_modality_matrix.svg`](../figures/servo_modality_matrix.svg)（+ `.png`）——
  「失效模式 × 感測模態」對應，A 灰階淡化，註明「軸承故障可由振動(B/B+) 與電流(C) 兩模態互補診斷」。README 主圖改用此圖。
- **報告/文件**：`FINAL_REPORT.md` §1/§2/§12/§15、`README.md`（定位段 + 四軌架構註）、`REPORT_OUTLINE.md` 對齊；A 結果保留不刪、僅降為方法基礎對照。
- **前端**（`nav.ts` / 首頁 Legacy / `about`）：A 移到對照模組最後、改 slate 灰階、標「合成·基礎」；順序統一 C→B→B+→A。
- **descope（確定不做）**：CE2（MCSA 頻譜邊帶）、CE3（全 4 工況跨工況泛化）、模組 B 1D-CNN AE，於各規劃文件標「不採用 + 2026-06-29 + 理由」，規劃內容保留供參考。

## 2. 除錯記錄 — CI py3.12 `pandas.date_range` 段錯誤（exit 139）

### 症狀
- PR 的 CI：`lint + tests (py3.12)` ❌ **exit code 139（SIGSEGV）**；`py3.11` / `web` ✅。
- 觸發 commit 是**純文檔 + 圖檔**，照理不可能讓 Python 測試崩潰。

### 診斷過程（測量，非猜測）
1. **先排除 flaky**：`gh run rerun --failed` 重跑 → **同一步又 exit 139**，確認是**確定性**失敗、非隨機。
2. **抓 log 定位崩潰點**：
   ```
   Fatal Python error: Segmentation fault
     File ".../pandas/core/indexes/datetimes.py", line 1442 in date_range
     File "tests/test_ims_features.py", line 56 in test_add_rul_health_is_monotonic_and_bounded
   ```
   → 崩在 `pandas.date_range`（原生 C extension）。
3. **diff 安裝版本**（py3.11 通過 vs py3.12 崩潰）：唯一差異 `scipy 1.17.1` vs `1.18.0`。
4. **再 diff 06-28 綠燈的 py3.12**：scipy 同為 `1.18.0`（**排除 scipy**），唯一變動是 **pandas `3.0.3 → 3.0.4`**。

| 環境 | pandas | scipy | numpy | 結果 |
| --- | --- | --- | --- | --- |
| 06-28 py3.12 | **3.0.3** | 1.18.0 | 2.4.6 | ✅ |
| 06-29 py3.12 | **3.0.4** | 1.18.0 | 2.4.6 | ❌ SIGSEGV |
| 06-29 py3.11 | 3.0.4 | 1.17.1 | 2.4.6 | ✅ |

### 根因
**`pandas 3.0.4` 的 py3.12 wheel 在 `date_range` 段錯誤**（py3.11 的 3.0.4、py3.12 的 3.0.3 皆正常）。
因專案依賴**未鎖版本**，06-28 後 pip 解析到新發布的壞 wheel。**與當日的文檔/前端整理無關。**

### 修復
`requirements.txt`：`pandas>=2.0` → **`pandas>=2.0,!=3.0.4`**（排除壞版本、保留下限、待 3.0.5+ 自動採用）。
同時讓雲端 Streamlit / HF 後端避開壞版本。

### 驗證
推送後新 CI run **全綠**（py3.11 / py3.12 / web / docker build smoke test 皆過）。

### 教訓 / 後續
- 依賴**全部未鎖版本** → CI 不可重現，這次踩 pandas、下次可能換別的套件。
- 後續可考慮加 `constraints.txt` 或鎖定整套科學運算堆疊版本（獨立議題，非緊急）。

## 工具備忘
- 本機 `gh` 安裝後 PATH 未即時更新；以登錄檔 Machine+User PATH 重載即可使用：
  `$env:Path = "$([Environment]::GetEnvironmentVariable('Path','Machine'));$([Environment]::GetEnvironmentVariable('Path','User'))"`。
