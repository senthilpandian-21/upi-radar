"""SQLAlchemy ORM models for UPI Radar.

PostgreSQL in production; if no PostgreSQL is reachable we transparently
fall back to a local SQLite file (demo mode) — the application code never
changes. Tests may point DATABASE_URL at a temp sqlite file.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Index,
                        Integer, Numeric, String, Text, create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import database_url

Base = declarative_base()

_engine = None
_SessionLocal = None
_engine_kind = {"postgres": False}


def _build_engine():
    """Create engine; prefers DATABASE_URL, auto-falls back to SQLite."""
    url = database_url()
    global _engine, _SessionLocal
    kind = "postgres" if url.startswith("postgresql") else "sqlite"
    if kind == "postgres":
        try:
            engine = create_engine(url, pool_pre_ping=True,
                                   pool_size=5, max_overflow=5)
            with engine.connect() as conn:  # eager connectivity check
                conn.exec_driver_sql("SELECT 1")
            _engine_kind["postgres"] = True
        except Exception as exc:
            logger.warning(f"PostgreSQL unreachable ({exc}) — "
                           f"falling back to SQLite (demo mode)")
            url = "sqlite:///" + str(
                Path(__file__).resolve().parents[1] / "data" / "upi_radar_demo.db")
            Path(url.split(":///")[1]).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        if url.startswith("sqlite:///"):
            Path(url.split(":///")[1]).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False}
                               if url.startswith("sqlite") else {})
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return engine


def get_engine():
    if _engine is None:
        _build_engine()
    return _engine


def is_postgres() -> bool:
    get_engine()
    return _engine_kind["postgres"]


def reset_engine() -> None:
    """Test hook: forces engine recreation (new DATABASE_URL)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    _engine_kind["postgres"] = False


def get_session():
    if _SessionLocal is None:
        _build_engine()
    return _SessionLocal()


@contextmanager
def session_scope():
    """Transactional session context manager (commit or rollback)."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables (SQLAlchemy DDL — idempotent)."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info(f"Database initialised ({'PostgreSQL' if is_postgres() else 'SQLite'})")


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Tables ───────────────────────────────────────────────────────────
class BankTransactionHourly(Base):
    __tablename__ = "bank_transactions_hourly"
    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, nullable=False, default=now_utc)
    bank_name = Column(String(50), nullable=False)
    hour_of_day = Column(Integer)
    day_of_week = Column(Integer)
    day_of_month = Column(Integer)
    month = Column(Integer)
    year = Column(Integer)
    is_salary_day = Column(Boolean, default=False)
    is_fy_end = Column(Boolean, default=False)
    is_month_end = Column(Boolean, default=False)
    is_peak_hour = Column(Boolean, default=False)
    is_festival_day = Column(Boolean, default=False)
    total_volume = Column(BigInteger)
    success_count = Column(BigInteger)
    failure_count = Column(BigInteger)
    failure_rate = Column(Numeric(10, 6))
    technical_decline_pct = Column(Numeric(10, 6))
    business_decline_pct = Column(Numeric(10, 6))
    bank_health_score = Column(Numeric(5, 2))
    complaint_volume = Column(Integer, default=0)
    created_at = Column(DateTime, default=now_utc)

    __table_args__ = (Index("ix_bth_bank_time", "bank_name", "recorded_at"),)


class DowntimeEvent(Base):
    __tablename__ = "razorpay_downtimes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    downtime_id = Column(String(50), unique=True)
    method = Column(String(50))
    bank = Column(String(100))
    severity = Column(String(20))
    status = Column(String(20))
    is_scheduled = Column(Boolean)
    begin_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_mins = Column(Integer)
    created_at = Column(DateTime, default=now_utc)

    @classmethod
    def from_dict(cls, data: dict) -> "DowntimeEvent":
        def _parse(value):
            if isinstance(value, str) and value:
                try:
                    return datetime.fromisoformat(value.replace("Z", ""))
                except ValueError:
                    return None
            return value

        return cls(
            downtime_id=str(data.get("downtime_id"))[:50],
            method=data.get("method"),
            bank=data.get("bank"),
            severity=data.get("severity"),
            status=data.get("status"),
            is_scheduled=data.get("is_scheduled"),
            begin_time=_parse(data.get("begin_time")),
            end_time=_parse(data.get("end_time")),
            duration_mins=data.get("duration_mins"),
        )


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    predicted_at = Column(DateTime, nullable=False, default=now_utc)
    prediction_horizon = Column(Integer, default=120)
    outage_probability = Column(Numeric(5, 4))
    lstm_prob = Column(Numeric(5, 4))
    prophet_prob = Column(Numeric(5, 4))
    anomaly_score = Column(Numeric(5, 4))
    actual_outage = Column(Boolean, default=None)
    prediction_correct = Column(Boolean, default=None)
    created_at = Column(DateTime, default=now_utc)

    __table_args__ = (Index("ix_mp_predicted_at", "predicted_at"),)


class RoutingDecisionRow(Base):
    __tablename__ = "routing_decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100))
    decided_at = Column(DateTime, nullable=False, default=now_utc)
    original_bank = Column(String(50))
    routed_to_bank = Column(String(50))
    original_method = Column(String(50))
    routed_to_method = Column(String(50))
    reason = Column(Text)
    outage_prob_at_time = Column(Numeric(5, 4))
    bank_health_score = Column(Numeric(5, 2))
    success = Column(Boolean)
    created_at = Column(DateTime, default=now_utc)

    __table_args__ = (Index("ix_rd_decided_at", "decided_at"),)


class AlertLog(Base):
    __tablename__ = "alerts_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50))
    severity = Column(String(20))
    message = Column(Text)
    sent_to = Column(String(200))
    outage_prob = Column(Numeric(5, 4))
    bank_affected = Column(String(100))
    method_affected = Column(String(50))
    sent_at = Column(DateTime, default=now_utc)
    resolved_at = Column(DateTime, default=None)


class QueuedTransaction(Base):
    __tablename__ = "transaction_queue"
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True)
    amount = Column(BigInteger)
    currency = Column(String(10), default="INR")
    original_method = Column(String(50))
    original_bank = Column(String(50))
    queued_at = Column(DateTime, default=now_utc)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=5)
    status = Column(String(20), default="queued")  # queued/processing/success/failed
    processed_at = Column(DateTime, default=None)
    error_message = Column(Text, default=None)
