"""Prediction endpoints — ensemble outage probability + history."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from loguru import logger

from api.schemas.response import PredictionResponse
from config import audit_path
from models.ensemble import EnsemblePredictor

router = APIRouter()

_predictor: Optional[EnsemblePredictor] = None


def get_predictor() -> EnsemblePredictor:
    global _predictor
    if _predictor is None:
        _predictor = EnsemblePredictor()
    return _predictor


@router.get("/outage", response_model=PredictionResponse,
            summary="Current outage probability (2h horizon)")
async def predict_outage() -> dict:
    """Ensemble forecast for the next 120 minutes across all PSPs."""
    predictor = get_predictor()
    result = predictor.get_mock_prediction()
    try:
        result = predictor.predict()
    except Exception as exc:
        logger.warning(f"Prediction endpoint fell back to demo signal: {exc}")

    prob = float(result["final_probability"])
    result["recommendation"] = EnsemblePredictor.recommendation(
        prob, result.get("risk_level"))
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.get("/history", summary="Recent model predictions")
async def predict_history(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Last N predictions — DB first, then the JSON audit log."""
    records = _history_from_db(limit)
    source = "db"
    if not records:
        records = _history_from_log(limit)
        source = "audit_log"
    return {"source": source, "count": len(records), "predictions": records}


def _history_from_db(limit: int) -> list[dict]:
    try:
        from db.models import ModelPrediction, session_scope

        with session_scope() as session:
            rows = (session.query(ModelPrediction)
                    .order_by(ModelPrediction.predicted_at.desc())
                    .limit(limit).all())
        return [
            {"predicted_at": row.predicted_at.isoformat() if row.predicted_at else None,
             "outage_probability": float(row.outage_probability or 0),
             "lstm_prob": float(row.lstm_prob or 0),
             "prophet_prob": float(row.prophet_prob or 0),
             "anomaly_score": float(row.anomaly_score or 0),
             "risk_level": ("RED" if float(row.outage_probability or 0) >= 0.7
                            else "YELLOW" if float(row.outage_probability or 0) >= 0.3
                            else "GREEN")}
            for row in rows]
    except Exception:
        return []


def _history_from_log(limit: int) -> list[dict]:
    import json

    path = audit_path("model_predictions.log")
    if not path.exists():
        return []
    try:
        lines = [line for line in path.read_text().splitlines()
                 if line.strip()][-limit:]
        return [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError):
        return []
