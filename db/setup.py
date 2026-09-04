"""Database setup script.

    python db/setup.py

* PostgreSQL + DATABASE_URL configured → applies the raw SQL migration
  (db/migrations/001_initial_schema.sql) and prints per-table status.
* Otherwise → initialises the SQLite demo DB via SQLAlchemy DDL.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
for _ in range(4):
    if (ROOT / "config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from config import PROJECT_ROOT
from db.models import Base, get_engine, init_db, is_postgres, session_scope

MIGRATION_SQL = PROJECT_ROOT / "db" / "migrations" / "001_initial_schema.sql"

EXPECTED_TABLES = [
    "bank_transactions_hourly",
    "razorpay_downtimes",
    "model_predictions",
    "routing_decisions",
    "alerts_log",
    "transaction_queue",
]


def create_tables() -> list[str]:
    """Create all tables. Returns the list of created table names."""
    engine = get_engine()

    if is_postgres() and MIGRATION_SQL.exists():
        # Apply the canonical migration file (indexes included)
        with engine.begin() as conn:
            conn.exec_driver_sql(MIGRATION_SQL.read_text())
        created = _verify_tables()
        logger.success(f"Migration applied via SQL: {MIGRATION_SQL.name}")
        for table in created:
            logger.info(f"  ✓ {table}")
        return created

    init_db()
    created = _verify_tables()
    logger.info("SQLite demo database ready (PostgreSQL not configured)")
    return created


def _verify_tables() -> list[str]:
    from sqlalchemy import inspect

    inspector = inspect(get_engine())
    existing = set(inspector.get_table_names())
    return [table for table in EXPECTED_TABLES if table in existing]


def seed_demo_data() -> None:
    """Idempotent demo seed: a few alerts/routing rows so dashboards and
    API history endpoints show something on first boot."""
    try:
        from datetime import timedelta

        from db.models import AlertLog, RoutingDecisionRow

        with session_scope() as session:
            if session.query(RoutingDecisionRow).count() == 0:
                session.add_all([
                    RoutingDecisionRow(
                        transaction_id="txn_seed_001", decided_at=now_minus(2),
                        original_bank="SBI", routed_to_bank="HDFC",
                        original_method="upi", routed_to_method="upi",
                        reason="RULE B4: SBI health score 63.4% < 70% "
                               "→ route via HDFC",
                        outage_prob_at_time=0.46, bank_health_score=63.4,
                        success=True),
                    RoutingDecisionRow(
                        transaction_id="txn_seed_002", decided_at=now_minus(5),
                        original_bank="BOB", routed_to_bank="KOTAK",
                        original_method="card", routed_to_method="card",
                        reason="RULE B1: BOB health score 21.0% < 30% "
                               "critical threshold",
                        outage_prob_at_time=0.81, bank_health_score=21.0,
                        success=True),
                ])
            if session.query(AlertLog).count() == 0:
                session.add(AlertLog(
                    alert_type="SRE", severity="HIGH",
                    message="RULE B1: BOB health score 21.0% — critical",
                    sent_to="sre-team@razorpay.com", outage_prob=0.81,
                    bank_affected="BOB", method_affected="upi"))
        logger.info("Demo seed data inserted (alerts/routing rows)")
    except Exception as exc:
        logger.warning(f"Demo seed skipped: {exc}")


def now_minus(hours: float):
    from datetime import datetime, timedelta

    return datetime.now() - timedelta(hours=hours)


def main() -> None:
    created = create_tables()
    print(f"Tables ready ({len(created)}): {', '.join(created)}")
    if not is_postgres():
        print("Note: running on SQLite demo DB — set DATABASE_URL for PostgreSQL.")


if __name__ == "__main__":
    main()
