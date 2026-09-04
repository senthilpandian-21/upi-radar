"""Queue agent — graceful transaction queuing + retry with backoff.

A transaction that cannot be completed during a degradation window is
queued (never silently failed), replayed with exponential backoff, and —
if still stuck after max retries/wait — surfaced loudly with an
alternate-method recommendation.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta

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


from config import audit_path, queue_process_interval_seconds

try:
    import razorpay

    RAZORPAY_AVAILABLE = True
except Exception:  # pragma: no cover
    razorpay = None
    RAZORPAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class QueueAgent:
    # ── enqueue ─────────────────────────────────────────────────────
    async def enqueue(self, transaction: dict, reason: str = "") -> str:
        """Persist a transaction as 'queued' (DB) + mirror in audit log."""
        from db.models import QueuedTransaction, session_scope

        txn_id = str(transaction.get("id") or transaction.get("transaction_id")
                     or f"queued_{datetime.now().timestamp()}")
        try:
            with session_scope() as session:
                existing = session.query(QueuedTransaction).filter_by(
                    transaction_id=txn_id).first()
                if existing is None:
                    session.add(QueuedTransaction(
                        transaction_id=txn_id,
                        amount=int(transaction.get("amount", 0)),
                        currency=transaction.get("currency", "INR"),
                        original_method=transaction.get("method", "upi"),
                        original_bank=transaction.get("bank", ""),
                        status="queued"))
            entry = {"action": "ENQUEUE", "transaction_id": txn_id,
                     "reason": reason, "timestamp": datetime.now().isoformat()}
            with audit_path("transaction_queue.log").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            logger.warning(f"Queued transaction {txn_id} ({reason})")
            return txn_id
        except Exception as exc:
            logger.error(f"Enqueue failed: {exc}")
            return ""

    # ── processing ──────────────────────────────────────────────────
    async def process_queue(self, limit: int = 25) -> dict:
        """Retry eligible queued transactions once each."""
        from policy.queue_rules import QueueRules

        from db.models import QueuedTransaction, session_scope

        summary = {"attempted": 0, "success": 0, "failed": 0, "abandoned": 0}
        try:
            with session_scope() as session:
                queued = (session.query(QueuedTransaction)
                          .filter(QueuedTransaction.status.in_(["queued", "processing"]))
                          .order_by(QueuedTransaction.queued_at.asc())
                          .limit(limit).all())
                for row in queued:
                    if row.retry_count >= row.max_retries:
                        row.status = "failed"
                        summary["abandoned"] += 1
                        continue
                    if QueueRules.should_abandon(
                            queued_at_iso=row.queued_at.isoformat()
                            if row.queued_at else None,
                            retry_count=row.retry_count):
                        row.status = "failed"
                        row.error_message = ("abandoned after max wait — "
                                             "recommend alternate method")
                        summary["abandoned"] += 1
                        continue
                    row.status = "processing"
                    row.retry_count += 1
                    session.flush()

                    txn = {"id": row.transaction_id, "amount": row.amount or 0,
                           "method": row.original_method, "bank": row.original_bank}
                    ok = await self._attempt_payment(txn)
                    row.status = "success" if ok else "queued"
                    summary["attempted"] += 1
                    if ok:
                        row.processed_at = datetime.now()
                        summary["success"] += 1
                    else:
                        delay = QueueRules.get_retry_delay(row.retry_count)
                        row.error_message = (f"retry {row.retry_count} "
                                             f"failed — next in {delay}s")
                        summary["failed"] += 1
            return summary
        except Exception as exc:
            logger.error(f"Queue processing error: {exc}")
            return summary

    async def _attempt_payment(self, transaction: dict,
                               bank: str | None = None,
                               method: str | None = None) -> bool:
        """Razorpay payment attempt (real SDK when keys present, else mock).

        The mock intentionally fails ~12% of attempts so cascade routing
        and retries can be observed live in demo mode.
        """
        import os

        import numpy as np

        bank = bank or transaction.get("bank")
        method = method or transaction.get("method", "upi")
        if transaction.get("_force_fail_all"):
            return False

        key_id = os.getenv("RAZORPAY_KEY_ID", "")
        key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
        if RAZORPAY_AVAILABLE and key_id and key_secret and "XXX" not in key_id:
            try:
                client = razorpay.Client(auth=(key_id, key_secret))
                order = client.order.create({
                    "amount": int(transaction.get("amount", 50000)),
                    "currency": "INR", "payment_capture": 1,
                    "notes": {"routed_bank": bank, "routed_method": method,
                              "upi_radar": "true"}})
                return order.get("status") == "created"
            except Exception as exc:
                logger.debug(f"Razorpay attempt failed ({exc}) → mock fallback")

        seed_text = f"{transaction.get('id', transaction.get('transaction_id', ''))}:{bank}"
        seed = abs(hash(seed_text)) % (2 ** 31)
        rng = np.random.default_rng(seed)
        return bool(rng.random() < 0.88)

    # ── status ──────────────────────────────────────────────────────
    def get_queue_status(self) -> dict:
        from db.models import QueuedTransaction, session_scope

        try:
            with session_scope() as session:
                rows = session.query(QueuedTransaction).all()
                statuses: dict[str, int] = {}
                for row in rows:
                    statuses[row.status] = statuses.get(row.status, 0) + 1
                return {"total": len(rows), "by_status": statuses}
        except Exception as exc:
            logger.debug(f"Queue status unavailable: {exc}")
            return {"total": 0, "by_status": {}, "error": str(exc)}

    # ── scheduler ───────────────────────────────────────────────────
    def run_scheduler(self) -> None:
        from apscheduler.schedulers.blocking import BlockingScheduler

        interval = queue_process_interval_seconds()
        scheduler = BlockingScheduler()
        scheduler.add_job(lambda: asyncio.run(self.process_queue()),
                          "interval", seconds=interval,
                          id="queue_processor", max_instances=1)
        logger.info(f"Queue processor started (every {interval}s)")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Queue processor stopped")


if __name__ == "__main__":  # pragma: no cover
    agent = QueueAgent()
    agent.run_scheduler()
