"""Central configuration for UPI Radar.

Every module reads its paths / env vars from here so that:
  * Demo mode works out of the box (safe defaults everywhere), and
  * tests can redirect audit logs / reports / DB to temp locations.

Env vars are resolved at call time (get_env), which lets pytest
override them with ``monkeypatch.setenv`` between tests.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

APP_NAME = "UPI Radar"
APP_VERSION = "1.0.0"


# ── env helpers ────────────────────────────────────────────────────────
def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def get_int(key: str, default: int) -> int:
    try:
        return int(float(get_env(key, str(default))))
    except (TypeError, ValueError):
        return default


def get_float(key: str, default: float) -> float:
    try:
        return float(get_env(key, str(default)))
    except (TypeError, ValueError):
        return default


def get_bool(key: str, default: bool = False) -> bool:
    val = get_env(key, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def is_demo_mode() -> bool:
    """Demo mode = everything must run without external infra/keys."""
    return get_bool("DEMO_MODE", default=True)


# ── directories (call-time so tests can redirect them) ────────────────
def _ensure(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:  # read-only FS edge cases
        pass
    return path


def root_dir() -> Path:
    return PROJECT_ROOT


def audit_dir() -> Path:
    return _ensure(Path(get_env("RADAR_AUDIT_DIR", str(PROJECT_ROOT / "audit_logs"))))


def audit_path(name: str) -> Path:
    return audit_dir() / name


def reports_dir() -> Path:
    return _ensure(Path(get_env("RADAR_REPORTS_DIR", str(PROJECT_ROOT / "reports"))))


def data_dir(*parts: str) -> Path:
    return _ensure(PROJECT_ROOT.joinpath("data", *parts))


# ── file paths ─────────────────────────────────────────────────────────
SYNTHETIC_CSV = data_dir("synthetic", "generated_data.csv")
PROCESSED_DIR = data_dir("processed")

MODEL_ROOT = PROJECT_ROOT / "models"
LSTM_MODEL_DIR = MODEL_ROOT / "lstm_forecaster" / "saved_model"
PROPHET_MODEL_DIR = MODEL_ROOT / "prophet_forecaster" / "saved_model"
ISO_MODEL_DIR = MODEL_ROOT / "anomaly_detector" / "saved_model"

LSTM_MODEL_PATH = LSTM_MODEL_DIR / "lstm_outage_model.h5"
LSTM_SCALER_PATH = LSTM_MODEL_DIR / "scaler.pkl"
PROPHET_MODEL_PATH = PROPHET_MODEL_DIR / "prophet_model.pkl"
ISO_MODEL_PATH = ISO_MODEL_DIR / "isolation_forest.pkl"


def models_available(model_kind: str = "lstm") -> bool:
    return {"lstm": LSTM_MODEL_PATH, "prophet": PROPHET_MODEL_PATH,
            "anomaly": ISO_MODEL_PATH}.get(model_kind, LSTM_MODEL_PATH).exists()


# ── infra connection strings ───────────────────────────────────────────
def database_url() -> str:
    """PostgreSQL DSN from env; falls back to SQLite for demo/tests."""
    return get_env(
        "DATABASE_URL",
        f"postgresql://{get_env('POSTGRES_USER', 'admin')}:{get_env('POSTGRES_PASSWORD', 'password123')}"
        f"@{get_env('POSTGRES_HOST', 'localhost')}:{get_int('POSTGRES_PORT', 5432)}"
        f"/{get_env('POSTGRES_DB', 'upi_radar')}",
    )


def kafka_bootstrap_servers() -> str:
    return get_env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


def kafka_topics() -> dict:
    return {
        "transactions": get_env("KAFKA_TOPIC_TRANSACTIONS", "upi_transactions"),
        "alerts": get_env("KAFKA_TOPIC_ALERTS", "upi_alerts"),
    }


def redis_config() -> dict:
    return {
        "host": get_env("REDIS_HOST", "localhost"),
        "port": get_int("REDIS_PORT", 6379),
        "db": get_int("REDIS_DB", 0),
        "password": get_env("REDIS_PASSWORD") or None,
        "socket_timeout": 2,
    }


# ── thresholds (env-tunable, matching .env.example) ────────────────────
def outage_threshold() -> float:
    return get_float("OUTAGE_PROBABILITY_THRESHOLD", 0.70)


def critical_health_threshold() -> float:
    return get_float("BANK_HEALTH_CRITICAL_THRESHOLD", 30.0)


def warning_health_threshold() -> float:
    return get_float("BANK_HEALTH_WARNING_THRESHOLD", 70.0)


def queue_max_wait_seconds() -> int:
    return get_int("QUEUE_MAX_WAIT_SECONDS", 120)


def prediction_interval_seconds() -> int:
    return get_int("PREDICTION_INTERVAL_SECONDS", 300)


def queue_process_interval_seconds() -> int:
    return get_int("QUEUE_PROCESS_INTERVAL_SECONDS", 30)


# ── logging bootstrap ──────────────────────────────────────────────────
def setup_logging(name: str = "upi_radar") -> None:
    logger.remove()
    logger.add(
        audit_dir() / f"{name}.log",
        rotation="10 MB",
        retention="7 days",
        level=get_env("LOG_LEVEL", "INFO"),
        enqueue=True,
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=get_env("LOG_LEVEL", "INFO"),
        enqueue=True,
    )
