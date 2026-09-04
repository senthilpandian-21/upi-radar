"""Router-agent integration tests (async, temp DB + audit dir)."""
from __future__ import annotations

import asyncio
import json

import pytest

from config import audit_path


class TestRouterAgent:
    def test_route_logs_decision(self, isolated_environment, sample_transaction):
        from agent.router_agent import RouterAgent
        from policy.bank_rules import RoutingDecision

        decision = RoutingDecision(action="ROUTE", target_bank="HDFC",
                                   target_method="upi",
                                   reason="RULE B4: degraded bank",
                                   severity="MEDIUM",
                                   requires_sre_alert=False, rule_id="B4")
        result = asyncio.run(RouterAgent().route(sample_transaction,
                                                 bank_decision=decision))

        assert result["success"] is True
        assert result["bank_used"] == "HDFC"

        log_path = audit_path("routing_decisions.log")
        lines = [json.loads(l) for l in log_path.read_text().splitlines()
                 if l.strip()]
        assert any(line["transaction_id"] == "txn_test_001"
                   and line["routed_to_bank"] == "HDFC" for line in lines)

    def test_cascade_falls_through_to_second_bank(self, isolated_environment,
                                                  sample_transaction):
        """Primary fails → router falls back to the next healthiest bank."""
        from agent.router_agent import RouterAgent
        from models.bank_health_scorer.scorer import BankHealthScorer

        scorer = BankHealthScorer(use_redis=False)
        scores = {"SBI": 95, "HDFC": 96, "ICICI": 88, "AXIS": 91,
                  "PNB": 61, "BOB": 54, "KOTAK": 93, "YES": 89}
        for bank, score in scores.items():
            scorer.update_bank_score(bank, score)

        transaction = dict(sample_transaction, bank="SBI",
                           _force_fail_banks=["HDFC"])
        result = asyncio.run(RouterAgent()._cascade_route(
            transaction, target_bank="HDFC", target_method="upi"))

        assert result["success"] is True
        assert result["bank_used"] != "HDFC"
        assert any(a["bank"] == "HDFC" and not a["success"]
                   for a in result["attempts"])

    def test_all_banks_fail_then_queue(self, isolated_environment,
                                       sample_transaction):
        from agent.router_agent import RouterAgent

        transaction = dict(sample_transaction, _force_fail_all=True)
        result = asyncio.run(RouterAgent()._cascade_route(
            transaction, target_bank="HDFC", target_method="upi"))
        assert result["success"] is False
        assert result["queued"] is True

    def test_attempt_payment_mock_distribution(self):
        from agent.router_agent import RouterAgent

        async def probe():
            ok = 0
            for i in range(60):
                transaction = {"id": f"mock_{i}", "amount": 50000}
                if await RouterAgent()._attempt_payment(transaction,
                                                        bank="HDFC",
                                                        method="upi"):
                    ok += 1
            return ok

        successes = asyncio.run(probe())
        # ~88% success rate with deterministic per-id seeds
        assert 40 <= successes <= 60


class TestQueueAgent:
    def test_enqueue_and_process(self, isolated_environment):
        from agent.queue_agent import QueueAgent
        from db.models import QueuedTransaction, session_scope

        transaction = {"id": "txn_queue_1", "amount": 40000,
                       "method": "upi", "bank": "BOB"}
        txn_id = asyncio.run(QueueAgent().enqueue(transaction,
                                                  reason="cascade exhausted"))
        assert txn_id == "txn_queue_1"

        with session_scope() as session:
            row = session.query(QueuedTransaction).filter_by(
                transaction_id="txn_queue_1").first()
            assert row is not None
            assert row.status == "queued"

        summary = asyncio.run(QueueAgent().process_queue())
        assert summary["attempted"] >= 1

        status = QueueAgent().get_queue_status()
        assert status["total"] >= 1

    def test_retry_increments(self, isolated_environment, monkeypatch):
        from agent.queue_agent import QueueAgent

        async def always_fail(_self, transaction, bank=None, method=None):
            return False

        monkeypatch.setattr(QueueAgent, "_attempt_payment", always_fail)

        transaction = {"id": "txn_queue_retry", "amount": 100,
                       "method": "upi", "bank": "SBI"}
        asyncio.run(QueueAgent().enqueue(transaction, reason="test"))
        summary = asyncio.run(QueueAgent().process_queue())
        from db.models import QueuedTransaction, session_scope

        with session_scope() as session:
            row = session.query(QueuedTransaction).filter_by(
                transaction_id="txn_queue_retry").first()
            assert row.retry_count == 1
            assert row.status == "queued"  # failed retry → requeued
        assert summary["attempted"] == 1
