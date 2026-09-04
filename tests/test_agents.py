"""Agent behaviour tests (alert / report / retrainer / monitor helpers)."""
from __future__ import annotations

import asyncio
import json

import pytest


class TestAlertAgent:
    def test_sre_alert_logged(self, isolated_environment):
        from agent.alert_agent import AlertAgent

        alert = AlertAgent()
        alert.demo_mode = True  # never actually sends mail in tests
        sent = asyncio.run(alert.send_sre_alert(
            probability=0.9, reason="TEST CRITICAL",
            prediction_details={"risk_level": "RED",
                                "lstm_probability": 0.8,
                                "prophet_probability": 0.7,
                                "anomaly_probability": 0.9}))
        assert sent is True

        from config import audit_path

        lines = [json.loads(l)
                 for l in audit_path("alerts_sent.log")
                 .read_text().splitlines() if l.strip()]
        assert lines[-1]["type"] == "SRE"
        assert lines[-1]["probability"] == pytest.approx(0.9)

    def test_merchant_alert_logged(self, isolated_environment):
        from agent.alert_agent import AlertAgent

        asyncio.run(AlertAgent().send_merchant_alert(
            transaction_id="txn_m_1"))
        from config import audit_path

        lines = [json.loads(l)
                 for l in audit_path("alerts_sent.log")
                 .read_text().splitlines() if l.strip()]
        assert lines[-1]["type"] == "MERCHANT"


class TestReportAgent:
    def test_generate_post_outage_pdf(self, isolated_environment):
        from datetime import datetime, timedelta

        from agent.report_agent import ReportAgent

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        decisions = [
            {"transaction_id": "t1", "original_bank": "SBI",
             "routed_to_bank": "HDFC", "success": True,
             "timestamp": end.isoformat()},
            {"transaction_id": "t2", "original_bank": "BOB",
             "routed_to_bank": "KOTAK", "success": True,
             "timestamp": end.isoformat()},
        ]
        path = ReportAgent().generate_post_outage_report(
            start, end, affected_banks=["SBI", "BOB"], routing_decisions=decisions)
        assert path.endswith(".pdf")
        assert ReportAgent().list_reports()  # non-empty

    def test_generate_daily_report(self, isolated_environment):
        from agent.report_agent import ReportAgent

        path = ReportAgent().generate_daily_report()
        assert path.endswith(".pdf")


class TestRetrainerAgent:
    def test_accuracy_needs_samples(self, isolated_environment):
        from agent.retrainer_agent import RetrainerAgent

        status = RetrainerAgent().check_model_accuracy(min_samples=5)
        assert status["needs_retrain"] is False
        assert status["accuracy"] is None

    def test_labelling_updates_accuracy(self, isolated_environment):
        from agent.retrainer_agent import RetrainerAgent
        from datetime import datetime, timedelta

        from db.models import ModelPrediction, session_scope

        # seed labelled history
        with session_scope() as session:
            session.add(ModelPrediction(
                predicted_at=datetime.now() - timedelta(days=2),
                outage_probability=0.9, prediction_correct=True,
                actual_outage=True))
            session.add(ModelPrediction(
                predicted_at=datetime.now() - timedelta(days=2),
                outage_probability=0.2, prediction_correct=True,
                actual_outage=False))
            session.add(ModelPrediction(
                predicted_at=datetime.now() - timedelta(days=2),
                outage_probability=0.8, prediction_correct=False,
                actual_outage=False))

        status = RetrainerAgent().check_model_accuracy(min_samples=2)
        assert status["accuracy"] == pytest.approx(2 / 3, abs=0.001)
        assert status["needs_retrain"] is True  # 0.67 < 0.80

    def test_label_recent_predictions(self, isolated_environment):
        from agent.retrainer_agent import RetrainerAgent
        from datetime import datetime, timedelta

        from db.models import ModelPrediction, session_scope

        with session_scope() as session:
            session.add(ModelPrediction(
                predicted_at=datetime.now() - timedelta(hours=5),
                outage_probability=0.9))
        updated = RetrainerAgent().label_recent_predictions(
            observed_outage=True)
        assert updated == 1


class TestMonitorAgent:
    def test_policy_pipeline_on_transaction(self, isolated_environment,
                                            sample_transaction):
        """Full monitor decision path: critical bank → ROUTE + audit."""
        from agent.monitor_agent import MonitorAgent

        monitor = MonitorAgent(demo_mode=True)
        # force a critical scenario
        monitor.health_scorer.update_bank_score("SBI", 22.0)
        monitor.health_scorer.update_bank_score("HDFC", 96.0)
        monitor.current_outage_prob = 0.62
        asyncio.run(monitor._process_transaction(sample_transaction))

        from config import audit_path

        log_lines = [json.loads(l) for l in
                     audit_path("routing_decisions.log")
                     .read_text().splitlines() if l.strip()]
        assert any(line["transaction_id"] == "txn_test_001"
                   and line["success"] is True for line in log_lines)

    def test_prediction_loop_logs_once(self, isolated_environment):
        from agent.monitor_agent import MonitorAgent

        monitor = MonitorAgent(demo_mode=True)
        prediction = asyncio.run(monitor._run_prediction())
        assert "final_probability" in prediction
        monitor._log_prediction(prediction)

        from config import audit_path

        path = audit_path("model_predictions.log")
        assert len([l for l in path.read_text().splitlines() if l.strip()]) == 1
