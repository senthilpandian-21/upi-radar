"""Bank health endpoints — real-time (or simulated) health per bank."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException

from api.schemas.response import BankHealthResponse
from models.bank_health_scorer.scorer import get_shared_scorer

router = APIRouter()


def get_scorer():
    """Shared scorer — the same cache instance the agents update."""
    return get_shared_scorer()


def _warm_scores() -> dict:
    """Return cached scores; seed a realistic simulation when the cache is
    still empty (first boot / demo)."""
    scorer = get_scorer()
    scores = scorer.get_all_bank_scores()
    if any(data.get("source") == "default" for data in scores.values()):
        scores = scorer.refresh_from_simulation()
    return scores


@router.get("/health", response_model=BankHealthResponse,
            summary="Health of all banks (0-100)")
async def get_all_bank_health() -> dict:
    scores = _warm_scores()
    best = scorer_healthiest(scores)
    critical = [b for b, d in scores.items()
                if float(d.get("score", 100)) < 30.0]
    warning = [b for b, d in scores.items()
               if 30.0 <= float(d.get("score", 100)) < 70.0]
    return {
        "bank_health_scores": scores,
        "best_available_banks": [b for b, _ in best],
        "critical_banks": critical,
        "warning_banks": warning,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/health/{bank_name}", summary="Single bank health detail")
async def get_bank_health(bank_name: str) -> dict:
    from models.bank_health_scorer.bank_config import ALL_BANKS

    scorer = get_scorer()
    bank = bank_name.upper()
    data = scorer.get_bank_score(bank)
    if data.get("source") == "default":
        if bank not in ALL_BANKS:
            raise HTTPException(status_code=404,
                                detail=f"No telemetry for bank '{bank}'")
        # known bank, cache not warm yet (fresh boot) → seed simulation
        _warm_scores()
        data = scorer.get_bank_score(bank)
    return {"bank": bank, **data,
            "timestamp": datetime.now().isoformat()}


def scorer_healthiest(scores: dict, min_score: float = 50.0) -> list:
    healthy = sorted(
        ((b, float(d.get("score", 0))) for b, d in scores.items()
         if float(d.get("score", 0)) >= min_score),
        key=lambda item: item[1], reverse=True)
    return healthy
