"""FastAPI entry point.

Run::

    uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

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
    BatchPredictResponse,
    FailureTypeMetricsResponse,
    FullPredictResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
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


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    return {"rows": services.comparison_metrics()}


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


@app.post("/servo/predict")
def servo_predict(req: ServoPredictRequest):
    """伺服馬達健康狀態估測（健康狀態分類 + DV 退化回歸）。"""
    try:
        return services.servo_predict_one(req.features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Static figures ----------------------------------------------------------
# Serve the pre-rendered training / evaluation figures (outputs/figures/*.png)
# as ``GET /figures/{name}`` so the frontend can ``<img src>`` them directly
# instead of reading the filesystem.
_figures_dir = resolve(load_config()["paths"]["outputs_figures"])
_figures_dir.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=_figures_dir), name="figures")
