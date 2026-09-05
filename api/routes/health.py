"""Health endpoint — component-level liveness for SRE."""
from __future__ import annotations

import socket
from datetime import datetime

from fastapi import APIRouter

from api.schemas.response import HealthResponse
from config import is_demo_mode, kafka_bootstrap_servers, models_available

router = APIRouter()


@router.get("", response_model=HealthResponse, summary="System health")
@router.get("/", response_model=HealthResponse, include_in_schema=False)
async def health() -> dict:
    components = {
        "postgres": _db_component(),
        "redis": _redis_component(),
        "kafka": _kafka_component(),
        "models": _models_component(),
    }
    demo = is_demo_mode()
    core_down = [name for name, c in components.items()
                 if name in ("postgres", "redis", "kafka") and not c["ok"]]
    status = "ok"
    if core_down and demo:
        status = "degraded"  # demo mode tolerates missing infra
    elif core_down:
        status = "error"
    return {
        "status": status,
        "components": components,
        "demo_mode": demo,
        "timestamp": datetime.now().isoformat(),
    }


def _db_component() -> dict:
    try:
        from db.models import get_engine, is_postgres

        engine = get_engine()
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return {"ok": True,
                "detail": "PostgreSQL" if is_postgres() else "SQLite (demo fallback)"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)[:200]}


def _redis_component() -> dict:
    try:
        from models.bank_health_scorer.scorer import BankHealthScorer

        scorer = BankHealthScorer()
        if scorer.connected:
            return {"ok": True, "detail": "connected"}
        return {"ok": False, "detail": "in-process cache (demo fallback)"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)[:200]}


def _kafka_component() -> dict:
    try:
        host_port = kafka_bootstrap_servers().split(",")[0].strip()
        host, _, port = host_port.partition(":")
        port = int(port or 9092)
        with socket.create_connection((host, port), timeout=1.0):
            return {"ok": True, "detail": f"{host}:{port} reachable"}
    except Exception:
        return {"ok": False, "detail": "no broker — spool/demo stream fallback"}


def _models_component() -> dict:
    status = {kind: models_available(kind)
              for kind in ("lstm", "prophet", "anomaly")}
    return {"ok": any(status.values()) or is_demo_mode(),
            "detail": status}
