"""Policy-engine tests — deterministic rules (no infra required)."""
from __future__ import annotations

import pytest

from policy.bank_rules import BankRulesEngine
from policy.gateway_rules import GatewayRulesEngine
from policy.queue_rules import QueueRules
from policy.sre_alert_rules import SREAlertRules


# ── bank rules ───────────────────────────────────────────────────────
class TestBankRules:
    @pytest.fixture
    def engine(self):
        return BankRulesEngine()

    def test_critical_bank_routes(self, engine, sample_transaction,
                                  sample_bank_scores):
        decision = engine.evaluate(sample_transaction, sample_bank_scores,
                                   outage_probability=0.2)
        assert decision.action == "ROUTE"
        assert decision.target_bank != "SBI"
        assert decision.severity == "HIGH"
        assert decision.requires_sre_alert is True
        assert decision.rule_id == "B1"

    def test_healthy_bank_proceeds(self, engine, sample_transaction,
                                   sample_bank_scores):
        transaction = dict(sample_transaction, bank="HDFC")
        decision = engine.evaluate(transaction, sample_bank_scores,
                                   outage_probability=0.1)
        assert decision.action == "PROCEED"
        assert decision.target_bank == "HDFC"
        assert decision.rule_id == "DEFAULT"

    def test_high_outage_routes(self, engine, sample_transaction,
                                sample_bank_scores):
        decision = engine.evaluate(sample_transaction, sample_bank_scores,
                                   outage_probability=0.9)
        assert decision.action == "ROUTE"

    def test_anomaly_routes(self, engine, sample_transaction,
                            sample_bank_scores):
        transaction = dict(sample_transaction, bank="HDFC")
        decision = engine.evaluate(transaction, sample_bank_scores,
                                   outage_probability=0.1,
                                   anomaly_detected=True)
        assert decision.action == "ROUTE"
        assert decision.rule_id == "B3"

    def test_degraded_bank_routes_medium(self, engine, sample_transaction,
                                        sample_bank_scores):
        transaction = dict(sample_transaction, bank="PNB")
        decision = engine.evaluate(transaction, sample_bank_scores,
                                   outage_probability=0.1)
        assert decision.action == "ROUTE"
        assert decision.severity == "MEDIUM"
        assert decision.rule_id == "B4"

    def test_moderate_probability_routes(self, engine, sample_transaction,
                                   sample_bank_scores):
        transaction = dict(sample_transaction, bank="HDFC")
        decision = engine.evaluate(transaction, sample_bank_scores,
                                   outage_probability=0.5)
        assert decision.action == "ROUTE"
        assert decision.rule_id == "B5"

    def test_no_healthy_bank_queues(self, engine, sample_transaction,
                                       sample_bank_scores):
        scores = {bank: {"score": 10.0, "status": "CRITICAL"}
                  for bank in sample_bank_scores}
        decision = engine.evaluate(sample_transaction, scores,
                                   outage_probability=0.9)
        assert decision.action == "QUEUE"

    def test_high_value_uses_neft_imps(self, engine, sample_transaction,
                                   sample_bank_scores):
        transaction = dict(sample_transaction, amount=2_000_000, bank="HDFC")
        decision = engine.evaluate(transaction, sample_bank_scores,
                                   outage_probability=0.8)
        assert decision.target_method == "neft_imps"
        assert decision.rule_id == "B2"


# ── gateway rules ────────────────────────────────────────────────────
class TestGatewayRules:
    @pytest.fixture
    def engine(self):
        return GatewayRulesEngine()

    @pytest.fixture
    def healthy_methods(self):
        return {m: {"score": 95.0} for m in
                ["upi", "card", "netbanking", "wallet", "neft_imps"]}

    def test_high_outage_switches_upi_to_card(self, engine, healthy_methods):
        decision = engine.evaluate("upi", healthy_methods,
                                   outage_probability=0.75)
        assert decision.action == "SWITCH_METHOD"
        assert decision.suggested_method == "card"

    def test_all_healthy_proceeds(self, engine, healthy_methods):
        decision = engine.evaluate("upi", healthy_methods,
                                   outage_probability=0.1)
        assert decision.action == "PROCEED"
        assert decision.suggested_method == "upi"

    def test_critical_method_switches(self, engine):
        methods = {m: {"score": 95.0} for m in
                   ["upi", "card", "netbanking", "wallet", "neft_imps"]}
        methods["upi"] = {"score": 25.0}
        decision = engine.evaluate("upi", methods, outage_probability=0.1)
        assert decision.action == "SWITCH_METHOD"
        assert decision.suggested_method == "card"

    def test_extreme_probability_forces_neft_imps(self, engine,
                                                  healthy_methods):
        decision = engine.evaluate("card", healthy_methods,
                                   outage_probability=0.9)
        assert decision.suggested_method == "neft_imps"


# ── SRE alert rules ──────────────────────────────────────────────────
class TestSREAlertRules:
    def test_extreme_probability_alerts(self):
        alert, reason = SREAlertRules().should_alert_sre(
            outage_probability=0.9, bank_health_scores={})
        assert alert is True
        assert "85%" in reason

    def test_multi_bank_degradation_alerts(self, sample_bank_scores):
        scores = dict(sample_bank_scores)
        for bank in ["SBI", "PNB", "BOB"]:
            scores[bank] = {"score": 40.0}
        alert, reason = SREAlertRules().should_alert_sre(
            outage_probability=0.3, bank_health_scores=scores)
        assert alert is True
        assert "Multiple banks" in reason

    def test_sustained_high_risk_alerts(self, sample_bank_scores):
        alert, reason = SREAlertRules().should_alert_sre(
            outage_probability=0.65, bank_health_scores=sample_bank_scores,
            consecutive_high_prob_count=6)
        assert alert is True
        assert "30 minutes" in reason

    def test_calm_situation_no_alert(self, sample_bank_scores):
        alert, reason = SREAlertRules().should_alert_sre(
            outage_probability=0.2, bank_health_scores=sample_bank_scores,
            consecutive_high_prob_count=0)
        assert alert is False
        assert reason == ""


# ── queue rules ──────────────────────────────────────────────────────
class TestQueueRules:
    def test_slow_gateway_queues(self):
        assert QueueRules.should_queue(razorpay_response_time_ms=6000) is True
        assert QueueRules.should_queue(razorpay_response_time_ms=1000) is False

    def test_exponential_backoff(self):
        assert QueueRules.get_retry_delay(0) == 10
        assert QueueRules.get_retry_delay(1) == 20
        assert QueueRules.get_retry_delay(4) == 120

    def test_abandon_after_retries(self):
        assert QueueRules.should_abandon(retry_count=5) is True
        assert QueueRules.should_abandon(retry_count=1) is False
