# 工作報告 — 2026-06-25

> **狀態（2026-06-25）**：主線重構第一輪完成（分支 `feat/servo-mainline`）。以 PHM 伺服馬達
> 滾珠螺桿退化資料為新主線；Model A/B/B+/C 降為補充模組，未刪除。全部以 placeholder
> 合成資料運作，端到端可跑（Streamlit 各頁、FastAPI、pytest 51 passed）。

## 本次完成

1. **模組 Servo 主線**
   - 特徵層 `src/features/servo_features.py`（逐時序 → 依 `run_index` 聚合；5 種特徵組）。
   - 資料建構 `src/data/{load_servo,build_servo_dataset}.py`：無原始資料時產生 **placeholder
     合成特徵表**（類別重疊以利教學），輸出 parquet / demo csv / 樣本筆。
   - Reference Model `src/models/train_servo.py`：分類（CV 選最佳）+ DV 回歸，匯出
     model/feature_config/metrics。placeholder 下 macro-F1≈0.75、R²≈0.73。
   - 推論 `src/models/servo_predict.py`：結構化輸出（健康狀態 / DV / 風險 / top_features /
     信心 / 建議）。
2. **AI 訓練模擬器** `src/models/servo_simulator.py`：5 演算法 × 資料量 × 特徵組，
   時間 / 指標 / 混淆矩陣 / 與 Reference 對照 + 文字解釋。
3. **應用層**
   - 馬達欄位解釋 `src/servo/field_glossary.py`。
   - LLM 維護助理 `src/llm/maintenance_assistant.py`（Anthropic SDK + 離線 fallback；
     保守措辭；模型預設 `claude-opus-4-8`，此層級不送 temperature）。
   - 維修知識庫 / RAG `src/knowledge/*`（TF-IDF，sklearn 無新依賴；5 份種子文件；
     選用白名單爬蟲）。
4. **深度學習（第二部分，離線唯讀）** `src/models/servo_dl.py`：MLP baseline + PCA 重建
   誤差（隨退化上升）。Dashboard 唯讀顯示。
5. **UI** `src/ui/servo_views.py` + 改寫 `app/streamlit_app.py`（導覽置頂 Servo 主線、
   首頁改 Servo、A/B/C 收進「補充模組」、HEROES/KPI/action 對應）。
6. **API** `app/backend`：新增 `GET /servo/model_info`、`POST /servo/predict`。
7. **文件 / 設定**：`docs/MODULE_SERVO_PLAN.md`、README / data/README 主線化、
   `config.yaml` 新增 `servo`/`llm`/`knowledge`、`.gitignore` 白名單 Servo 產物、
   `requirements-dev.txt` 加 anthropic/requests/bs4。
8. **測試**：新增 `tests/test_servo_features.py`、`tests/test_servo_predict.py`（含 RAG /
   LLM fallback）；全套 **51 passed**。

## 誠實性

- Servo 為**模擬資料**、非真實實機；`run_index` 非 RUL；**不宣稱 RUL**。
- 目前產物為 **placeholder 合成資料**（`config.yaml::servo.placeholder: true`），僅供流程展示。
- 真正 1D-CNN / Autoencoder 需離線 torch 與真實時序資料，列為後續工作。

## 待真實 PHM 資料下載後

放入 `data/raw/servo/` → 重跑 `build_servo_dataset` / `train_servo` / `servo_dl` →
設 `placeholder: false` → 依真實 DV 分布重校風險帶。見
[`../../docs/MODULE_SERVO_PLAN.md`](../../docs/MODULE_SERVO_PLAN.md) 第 10 節。
