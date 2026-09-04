"""Shared pytest fixtures for UPI Radar.

Fixtures redirect audit logs + DB to temp dirs, so tests never touch
real project state. Services (Kafka/Redis/Postgres) are never required.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Demo mode ON: all tests run against built-in fallbacks
os.environ["DEMO_MODE"] = "true"


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Redirect audit logs & reports to a temp dir for every test."""
    monkeypatch.setenv("RADAR_AUDIT_DIR", str(tmp_path / "audit_logs"))
    monkeypatch.setenv("RADAR_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("DATABASE_URL",
                       f"sqlite:///{tmp_path / 'test_radar.db'}")
    # force fresh DB engine per test
    from db.models import reset_engine
    reset_engine()
    from db.models import init_db
    init_db()
    yield tmp_path


@pytest.fixture
def sample_transaction() -> dict:
    return {"id": "txn_test_001", "transaction_id": "txn_test_001",
            "amount": 500000, "currency": "INR", "method": "upi",
            "bank": "SBI", "customer_id": "cust_test"}


@pytest.fixture
def sample_bank_scores() -> dict:
    return {
        "SBI":   {"score": 25.0, "status": "CRITICAL"},
        "HDFC":  {"score": 96.0, "status": "EXCELLENT"},
        "ICICI": {"score": 88.0, "status": "EXCELLENT"},
        "AXIS":  {"score": 91.0, "status": "EXCELLENT"},
        "PNB":   {"score": 61.0, "status": "DEGRADED"},
        "BOB":   {"score": 54.0, "status": "DEGRADED"},
        "KOTAK": {"score": 93.0, "status": "EXCELLENT"},
        "YES":   {"score": 89.0, "status": "EXCELLENT"},
    }


@pytest.fixture
def sample_prediction() -> dict:
    return {
        "final_probability": 0.82,
        "lstm_probability": 0.85,
        "prophet_probability": 0.71,
        "anomaly_probability": 0.88,
        "risk_level": "RED",
        "horizon_minutes": 120,
        "predicted_at": "2026-09-05T12:00:00",
        "models_loaded": {"lstm": False, "prophet": False,
                          "isolation": False},
        "demo": True,
    }


@pytest.fixture
def mock_db_session(isolated_environment):
    """SQLAlchemy session bound to the temp sqlite DB."""
    from db.models import get_session

    session = get_session()
    yield session
    session.close()


@pytest.fixture
def test_client(isolated_environment):
    """FastAPI TestClient (httpx-based)."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        yield client
