"""Pydantic schemas for the FastAPI request / response models."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    type: Literal["L", "M", "H"] = Field(..., description="Product quality variant.")
    air_temperature_K: float = Field(..., ge=270, le=320)
    process_temperature_K: float = Field(..., ge=270, le=330)
    rotational_speed_rpm: float = Field(..., gt=0)
    torque_Nm: float = Field(..., ge=0)
    tool_wear_min: float = Field(..., ge=0)

    def to_raw_record(self) -> dict:
        """Return a dict matching the column names used by the training data."""
        return {
            "Type": self.type,
            "Air temperature [K]": self.air_temperature_K,
            "Process temperature [K]": self.process_temperature_K,
            "Rotational speed [rpm]": self.rotational_speed_rpm,
            "Torque [Nm]": self.torque_Nm,
            "Tool wear [min]": self.tool_wear_min,
        }


class PredictResponse(BaseModel):
    failure_probability: float
    predicted_class: int
    health_score: float
    risk_level: Literal["Low", "Medium", "High"]
    maintenance_advice: List[str]


class FullPredictResponse(PredictResponse):
    failure_type_probabilities: dict
    likely_failure_types: List[str]
    failure_type_notes: List[str]


class FailureTypeMetricsResponse(BaseModel):
    rows: List[dict]


class BatchPredictResponse(BaseModel):
    count: int
    results: List[PredictResponse]


class ModelInfoResponse(BaseModel):
    model_name: str
    feature_set: str
    feature_columns: List[str]
    metrics: dict


class MetricsResponse(BaseModel):
    rows: List[dict]


class HealthResponse(BaseModel):
    status: Literal["ok", "model_missing"]
    model_loaded: bool
    message: str | None = None


class ServoSimulateRequest(BaseModel):
    task: Literal["clf", "reg"]
    feature_set: str
    algo: str
    n: int = Field(..., gt=0, description="Sample size to train on (clamped to the table size).")


class MaintenanceAdviceRequest(BaseModel):
    health: float = Field(..., description="0..100 health score (100 = baseline-healthy).")
    rul_hours: float | None = Field(None, description="Remaining-useful-life estimate; None if not estimable.")
    past_fpt: bool = Field(..., description="Whether degradation onset (FPT) has been detected.")
    alarm_health: float = 30.0
    safety_margin: float = Field(0.3, ge=0.0, lt=1.0)
    cost_unplanned: float | None = None
    cost_planned: float | None = None
