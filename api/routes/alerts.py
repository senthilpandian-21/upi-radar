"""Alert endpoints — active alerts + history."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from config import audit_path

router = APIRouter()


@router.get("/active", summary="Unresolved alerts")
async def active_alerts() -> dict:
    """Alerts not yet resolved (from DB; falls back to recent audit log)."""
    from db.models import AlertLog, session_scope

    try:
        with session_scope() as session:
            rows = (session.query(AlertLog)
                    .filter(AlertLog.resolved_at.is_(None))
                    .order_by(AlertLog.sent_at.desc()).limit(20).all())
        if rows:
            return {"source": "db", "count": len(rows),
                    "alerts": [row_as_dict(r) for r in rows]}
    except Exception:
        pass

    recent = _from_log()
    # in demo/audit mode treat SRE alerts from the last 2 hours as active
    active = [a for a in recent if a.get("type") == "SRE"
              and _within_minutes(a.get("timestamp"), minutes=120)]
    return {"source": "audit_log", "count": len(active),
            "alerts": active[:20]}


@router.get("/history", summary="Alert history")
async def alert_history(limit: int = Query(50, ge=1, le=500)) -> dict:
    from db.models import AlertLog, session_scope

    try:
        with session_scope() as session:
            rows = (session.query(AlertLog)
                    .order_by(AlertLog.sent_at.desc()).limit(limit).all())
        if rows:
            return {"source": "db", "count": len(rows),
                    "alerts": [row_as_dict(r) for r in rows]}
    except Exception:
        pass
    logs = _from_log()[-limit:]
    return {"source": "audit_log", "count": len(logs), "alerts": logs}


def row_as_dict(row) -> dict:
    return {"id": row.id, "alert_type": row.alert_type,
            "severity": row.severity, "message": row.message,
            "sent_to": row.sent_to,
            "outage_prob": float(row.outage_prob or 0),
            "bank_affected": row.bank_affected,
            "method_affected": row.method_affected,
            "sent_at": row.sent_at.isoformat() if row.sent_at else None,
            "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None}


def _from_log() -> list[dict]:
    path = audit_path("alerts_sent.log")
    if not path.exists():
        return []
    try:
        lines = [line for line in path.read_text().splitlines()
                 if line.strip()]
        return [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError):
        return []


def _within_minutes(iso: Optional[str], minutes: int = 120) -> bool:
    if not iso:
        return False
    try:
        stamp = datetime.fromisoformat(iso)
        return datetime.now() - stamp < timedelta(minutes=minutes)
    except (ValueError, TypeError):
        return False
