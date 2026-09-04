"""API response envelopes."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    final_probability: float
    lstm_probability: float
    prophet_probability: float
    anomaly_probability: float
    risk_level: str                  # GREEN / YELLOW / RED
    horizon_minutes: int = 120
    recommendation: str
    timestamp: str
    demo: bool = False
    models_loaded: Optional[dict] = None


class BankHealthEntry(BaseModel):
    score: float
    status: str
    updated_at: Optional[str] = None
    source: str = "default"


class BankHealthResponse(BaseModel):
    bank_health_scores: dict
    best_available_banks: list
    critical_banks: list
    warning_banks: list
    timestamp: str


class ComponentStatus(BaseModel):
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    status: str          # ok / degraded / error
    components: dict
    demo_mode: bool
    timestamp: str


class MessageResponse(BaseModel):
    message: str
    status: str = "ok"
