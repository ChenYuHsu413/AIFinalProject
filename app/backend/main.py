"""FastAPI entry point.

Run::

    uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Make ``src.*`` and ``app.*`` resolvable even when this file is launched
# directly (uvicorn handles it via the dotted-module form, but we want a
# robust default).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.backend import services
from src.utils.paths import load_config, resolve
from app.backend.schemas import (
    AssistantQARequest,
    AssistantReportRequest,
    BatchPredictResponse,
    FailureTypeMetricsResponse,
    FullPredictResponse,
    HealthResponse,
    MaintenanceAdviceRequest,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ServoSimulateRequest,
)

app = FastAPI(
    title="AI 伺服馬達健康狀態估測與智慧維護助理 API",
    version="0.2.0",
    description=(
        "主線：以 PHM 伺服馬達滾珠螺桿退化（模擬）資料估測健康狀態與退化值"
        "（/servo/*）。補充：UCI AI4I 2020 合成資料的靜態故障風險（/predict 等）。"
        "本服務僅作為維護決策輔助，不會直接控制馬達。"
    ),
)

# Permissive CORS so a separately hosted frontend (Nginx / React / Streamlit)
# can call the API during development.  Tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return services.health()


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info():
    try:
        return services.model_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return services.predict_one(req.to_raw_record())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="請上傳 .csv 檔案。")
    content = await file.read()
    try:
        results = services.predict_batch_from_csv(content)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"count": len(results), "results": results}


@app.post("/predict/explain")
def predict_explain(req: PredictRequest):
    """單筆 SHAP 特徵貢獻解釋；非樹模型回 {"supported": false}。"""
    try:
        return services.predict_explain(req.to_raw_record())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return {"rows": services.comparison_metrics()}


@app.get("/metrics/test_predictions")
def metrics_test_predictions():
    """測試集 y_true / y_proba 陣列（門檻調整器用，混淆矩陣運算在前端）。"""
    return services.test_predictions()


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(records: List[PredictRequest]):
    """JSON 批次預測（What-if 1D/2D sweep、風險地景格網用，取代多次單點呼叫）。"""
    raw = [r.to_raw_record() for r in records]
    try:
        results = services.predict_batch_records(raw)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"count": len(results), "results": results}


@app.post("/predict_full", response_model=FullPredictResponse)
def predict_full(req: PredictRequest):
    """單筆預測 + 第二階段故障類型分析。

    回傳第一階段的故障機率／健康分數／維護建議，並附上每種故障類型
    （TWF / HDF / PWF / OSF / RNF）的機率與顯著類型清單。
    """
    try:
        return services.predict_one_full(req.to_raw_record())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/failure_type_metrics", response_model=FailureTypeMetricsResponse)
def failure_type_metrics():
    return {"rows": services.failure_type_metrics()}


@app.get("/metrics/summary")
def metrics_summary(module: str):
    """各模組總覽 KPI 所需的原始指標 JSON。

    ``module`` 取值：``servo`` / ``B`` / ``Bplus`` / ``C``。
    （模組 A 的 KPI 由 ``/metrics`` 與 ``/model_info`` 提供。）
    """
    try:
        return services.metrics_summary(module)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/paderborn/eval")
def paderborn_eval():
    """模組 C 完整評估：baseline CV vs 人工→真實泛化 + 混淆矩陣（含 summary）。"""
    return services.paderborn_eval()


@app.get("/paderborn/samples")
def paderborn_samples():
    """即時推論用的代表性量測（每種損傷來源/故障類別數筆，含真實標籤與特徵）。"""
    return services.paderborn_samples()


@app.get("/paderborn/domain_adapt")
def paderborn_domain_adapt():
    """CE1 領域自適應消融：baseline / CORAL / 工況感知標準化 / few-shot 學習曲線。"""
    return services.paderborn_domain_adapt()


from pydantic import BaseModel  # noqa: E402


class PaderbornPredictRequest(BaseModel):
    """One Paderborn feature row to classify (see /paderborn/samples for shape)."""
    features: dict


@app.post("/paderborn/predict")
def paderborn_predict(req: PaderbornPredictRequest):
    """以已訓練的 Paderborn 分類器即時推論一筆量測：回預測類別 + 各類機率。"""
    try:
        return services.paderborn_predict_one(req.features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/xjtu/generalization")
def xjtu_generalization():
    """模組 B+ 多軌跡退化偵測（固定參數 FPT，per-bearing + aggregate）。"""
    return services.xjtu_generalization()


@app.get("/xjtu/lobo_loco")
def xjtu_lobo_loco():
    """模組 B+ 監督式 RUL 泛化：留一軸承（LOBO）+ 留一工況（LOCO）。"""
    return services.xjtu_lobo_loco()


@app.get("/xjtu/domain_adapt")
def xjtu_domain_adapt():
    """模組 B+ E1 跨工況域適應消融（results / summary）。"""
    return services.xjtu_domain_adapt()


@app.get("/xjtu/rul_predictions")
def xjtu_rul_predictions():
    """模組 B+ E2 每軸承 RUL/健康預測表（維護建議輸入）。"""
    return services.xjtu_rul_predictions()


@app.get("/xjtu/health_overlay")
def xjtu_health_overlay():
    """模組 B+ 15 軸承健康指標疊圖（vs 壽命%）+ FPT 標記；特徵表缺回 available:false。"""
    return services.xjtu_health_overlay()


@app.get("/xjtu/replay/{condition}/{bearing}")
def xjtu_replay(condition: str, bearing: str):
    """模組 B+ E3 單軸承串流回放預算 frames（≤100）；找不到軌跡回 404。"""
    try:
        return services.xjtu_replay(condition, bearing)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/maintenance/advice")
def maintenance_advice(req: MaintenanceAdviceRequest):
    """依當下 health / RUL / FPT 給出風險等級、建議維護時窗與理由（含可選成本對照）。"""
    return services.maintenance_advice(
        req.health, req.rul_hours, req.past_fpt,
        alarm_health=req.alarm_health, safety_margin=req.safety_margin,
        cost_unplanned=req.cost_unplanned, cost_planned=req.cost_planned,
    )


@app.get("/ims/metrics")
def ims_metrics():
    """模組 B（IMS）RUL 指標/中繼資料（健康指標、FPT、提前量、近失效誤差）。"""
    return services.ims_metrics()


@app.get("/ims/health_curve")
def ims_health_curve():
    """模組 B（IMS）健康/RUL 時間序列；中繼資料另見 /ims/metrics。"""
    return services.ims_health_curve()


@app.get("/ims/health_indicator")
def ims_health_indicator(indicator: str | None = None):
    """切換健康指標即時重算健康曲線與退化起點（FPT）；未知指標回 400。"""
    try:
        return services.ims_health_indicator(indicator)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ims/snapshot/{index}")
def ims_snapshot(index: int):
    """單一快照原始振動波形 + FFT 頻譜；無原始資料時回 available:false。"""
    try:
        return services.ims_snapshot(index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/knowledge/documents")
def knowledge_documents():
    """維修知識庫文件清單（source / title / preview / chars）。"""
    return services.knowledge_documents()


@app.get("/knowledge/search")
def knowledge_search(q: str, top_k: int | None = None):
    """知識庫 TF-IDF 關鍵字搜尋；每筆結果含 text / score / source / title / topic。"""
    return services.knowledge_search(q, top_k=top_k)


# --- Module Servo (project main line) ----------------------------------------
from pydantic import BaseModel  # noqa: E402


class ServoPredictRequest(BaseModel):
    """Aggregated feature row for the reference health/DV models.

    ``features`` maps each required feature column (see /servo/model_info) to a
    numeric value.
    """
    features: dict


@app.get("/servo/model_info")
def servo_model_info():
    try:
        return services.servo_model_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/servo/provenance")
def servo_provenance():
    """資料溯源：真實 PHM FMCRD 指紋 + 聚合統計 + 留出測試指標（證明非 placeholder）。"""
    return services.servo_provenance()


@app.post("/servo/predict")
def servo_predict(req: ServoPredictRequest):
    """伺服馬達健康狀態估測（健康狀態分類 + DV 退化回歸）。"""
    try:
        return services.servo_predict_one(req.features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/servo/glossary")
def servo_glossary():
    """馬達欄位辭典（每筆 name / zh / desc / meaning / anomaly）。"""
    return services.servo_glossary()


@app.get("/servo/feature_sets")
def servo_feature_sets():
    """可用特徵組（key -> label / desc / columns）。"""
    return services.servo_feature_sets()


@app.get("/servo/samples")
def servo_samples():
    """儀表板可選的 demo 樣本列（特徵 + ylabel / DV）。"""
    return services.servo_samples()


@app.get("/servo/fleet")
def servo_fleet():
    """Command Center 機群：合成設備識別 + 真實參考模型在代表性 demo 運轉段上的健康輸出。"""
    try:
        return services.servo_fleet()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/servo/alerts")
def servo_alerts():
    """機群告警：由真實模型驅動的機群（風險/狀態/異常特徵）衍生。"""
    try:
        return services.servo_alerts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/servo/work_orders")
def servo_work_orders():
    """維修工單：由機群告警衍生。"""
    try:
        return services.servo_work_orders()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/servo/reference_metrics")
def servo_reference_metrics():
    """訓練模擬器的對照指標：clf / reg / dl（離線 baseline）。"""
    return services.servo_reference_metrics()


@app.get("/servo/cnn_results")
def servo_cnn_results():
    """Phase B 離線 1D-CNN（原始波形能量包絡）：分類指標 + conv-AE 重建誤差；未建置回 {}。"""
    return services.servo_cnn_results()


@app.get("/servo/simulate/options")
def servo_simulate_options():
    """訓練模擬器可選的演算法（分類 / 回歸名稱 + 中文標籤）。"""
    return services.servo_simulate_options()


@app.post("/servo/simulate")
def servo_simulate(req: ServoSimulateRequest):
    """瀏覽器端小模型即時訓練（同步，<0.4s），回指標、混淆矩陣與說明。"""
    try:
        return services.servo_simulate(req.task, req.feature_set, req.algo, req.n)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/servo/assistant/providers")
def servo_assistant_providers():
    """已設定的 LLM 供應商（依序嘗試；全空則用離線確定性範本）。"""
    return services.assistant_providers()


@app.post("/servo/assistant/report")
def servo_assistant_report(req: AssistantReportRequest):
    """依結構化預測 + 知識庫檢索產生完整維護報告；回 {text, source}。"""
    return services.assistant_report(req.prediction)


@app.post("/servo/assistant/qa")
def servo_assistant_qa(req: AssistantQARequest):
    """依結構化預測 + 知識庫檢索回答維修問題（不輸出整份報告）；回 {text, source}。"""
    return services.assistant_qa(req.question, req.prediction)


# --- Static figures ----------------------------------------------------------
# Serve the pre-rendered training / evaluation figures (outputs/figures/*.png)
# as ``GET /figures/{name}`` so the frontend can ``<img src>`` them directly
# instead of reading the filesystem.
_figures_dir = resolve(load_config()["paths"]["outputs_figures"])
_figures_dir.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=_figures_dir), name="figures")
