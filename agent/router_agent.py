"""Router agent — executes smart routing with cascade fallback.

Every routing attempt is written to:
  1. PostgreSQL ``routing_decisions`` table   (when available)
  2. audit_logs/routing_decisions.log         (always, JSON lines)

Cascade: primary target → next-best banks (max 3 attempts) → queue.
Everything stays inside Razorpay's ecosystem.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

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


from config import audit_path

logger = logging.getLogger(__name__)


class RouterAgent:
    MAX_CASCADE_ATTEMPTS = 3

    # ── entry point ─────────────────────────────────────────────────
    async def route(self, transaction: dict,
                    bank_decision=None, gateway_decision=None) -> dict:
        """
        Resolve final (bank, method) from both decision layers, execute
        cascade routing and persist the audit trail.
        Returns {success, bank_used, method_used, queued, attempts, reason}.
        """
        from models.bank_health_scorer.scorer import BankHealthScorer

        original_bank = str(transaction.get("bank", "UNKNOWN")).upper()
        original_method = transaction.get("method", "upi")

        final_bank = getattr(bank_decision, "target_bank", None) or original_bank
        final_method = (getattr(gateway_decision, "suggested_method", None)
                        or original_method)
        reason = (getattr(bank_decision, "reason", "") or "smart routing") + \
                 (" | " + gateway_decision.reason if gateway_decision is not None
                  and getattr(gateway_decision, "reason", "") else "")

        scorer = BankHealthScorer()
        bank_health_at_time = scorer.get_bank_score(original_bank).get("score")

        if final_bank == "QUEUE":
            queued = await self._queue_transaction(transaction, reason)
            result = {"success": False, "bank_used": None, "method_used": None,
                      "queued": queued, "attempts": [], "reason": reason}
            self._log_routing_decision(
                transaction_id=transaction.get("id") or transaction.get("transaction_id"),
                original_bank=original_bank, routed_to_bank=None,
                original_method=original_method, routed_to_method=final_method,
                reason=reason, success=False, queued=queued,
                bank_health_score=bank_health_at_time)
            return result

        attempt = await self._cascade_route(
            transaction=transaction, target_bank=final_bank,
            target_method=final_method)

        self._log_routing_decision(
            transaction_id=transaction.get("id") or transaction.get("transaction_id"),
            original_bank=original_bank,
            routed_to_bank=attempt.get("bank_used"),
            original_method=original_method,
            routed_to_method=attempt.get("method_used", final_method),
            reason=reason, success=attempt.get("success", False),
            queued=attempt.get("queued", False),
            bank_health_score=bank_health_at_time)
        return {**attempt, "reason": reason}

    # ── cascade ─────────────────────────────────────────────────────
    async def _cascade_route(self, transaction: dict,
                             target_bank: str | None,
                             target_method: str) -> dict:
        """
        Try: target_bank → next healthiest banks (≤ MAX_CASCADE_ATTEMPTS).
        If every attempt fails → queue the transaction for later retry.
        """
        from models.bank_health_scorer.scorer import BankHealthScorer

        scorer = BankHealthScorer()
        original_bank = str(transaction.get("bank", "UNKNOWN")).upper()
        healthy = scorer.get_best_available_banks(min_score=50.0,
                                                  exclude=[original_bank])
        ordered = [target_bank] if target_bank else []
        ordered += [bank for bank, _ in healthy if bank not in ordered]

        attempts = []
        for bank in ordered[: self.MAX_CASCADE_ATTEMPTS]:
            success = await self._attempt_payment(transaction, bank=bank,
                                                  method=target_method)
            attempts.append({"bank": bank, "method": target_method,
                             "success": bool(success)})
            if success:
                logger.info(f"✅ Routed {transaction.get('id', '?')} via "
                            f"{bank} ({target_method})")
                return {"success": True, "bank_used": bank,
                        "method_used": target_method, "queued": False,
                        "attempts": attempts}

        logger.error(f"All cascade routes failed for "
                     f"{transaction.get('id', '?')} — queueing")
        queued = await self._queue_transaction(transaction, "cascade exhausted")
        return {"success": False, "bank_used": None, "method_used": None,
                "queued": queued, "attempts": attempts}

    # ── payment attempt ─────────────────────────────────────────────
    async def _attempt_payment(self, transaction: dict,
                               bank: str, method: str) -> bool:
        """Delegate to QueueAgent's attempt (real Razorpay or mock)."""
        from agent.queue_agent import QueueAgent

        # demo-script hook: force a specific bank to fail (cascade demo)
        forced_fail = transaction.get("_force_fail_banks") or []
        if bank in forced_fail:
            return False
        return await QueueAgent()._attempt_payment(transaction, bank=bank,
                                                   method=method)

    async def _queue_transaction(self, transaction: dict,
                                 reason: str = "") -> bool:
        from agent.queue_agent import QueueAgent

        txn_id = await QueueAgent().enqueue(transaction, reason=reason)
        return bool(txn_id)

    # ── audit trail ─────────────────────────────────────────────────
    def _log_routing_decision(self, **kwargs) -> None:
        entry = {"timestamp": datetime.now().isoformat(),
                 "transaction_id": kwargs.get("transaction_id"),
                 "decided_at": datetime.now().isoformat(),
                 "original_bank": kwargs.get("original_bank"),
                 "routed_to_bank": kwargs.get("routed_to_bank"),
                 "original_method": kwargs.get("original_method"),
                 "routed_to_method": kwargs.get("routed_to_method"),
                 "reason": kwargs.get("reason"),
                 "outage_prob_at_time": kwargs.get("outage_prob_at_time"),
                 "bank_health_score": kwargs.get("bank_health_score"),
                 "success": kwargs.get("success"),
                 "queued": kwargs.get("queued", False)}
        # 1) JSON audit file (always)
        try:
            with audit_path("routing_decisions.log").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        # 2) DB (best-effort)
        try:
            from db.models import RoutingDecisionRow, session_scope

            with session_scope() as session:
                session.add(RoutingDecisionRow(
                    transaction_id=str(kwargs.get("transaction_id"))[:100],
                    original_bank=kwargs.get("original_bank"),
                    routed_to_bank=kwargs.get("routed_to_bank"),
                    original_method=kwargs.get("original_method"),
                    routed_to_method=kwargs.get("routed_to_method"),
                    reason=kwargs.get("reason"),
                    outage_prob_at_time=kwargs.get("outage_prob_at_time"),
                    bank_health_score=kwargs.get("bank_health_score"),
                    success=kwargs.get("success")))
        except Exception as exc:
            logger.debug(f"Routing decision DB log skipped: {exc}")
