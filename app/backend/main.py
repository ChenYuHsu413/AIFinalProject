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

from app.backend import services
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
    title="AI 伺服馬達預測性維護 API",
    version="0.1.0",
    description=(
        "以 UCI AI4I 2020（合成資料集）建立的預測性維護原型 API。"
        "提供故障機率預測、健康分數，以及依規則產生的維護建議。"
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
