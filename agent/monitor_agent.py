"""Monitor agent — the system's conductor.

Runs two concurrent loops:
  1. PREDICTION LOOP  — every N seconds/minutes: ensemble forecast →
     SRE alert rules → page SRE → persist prediction.
  2. STREAM LOOP      — continuous: consume Kafka (or demo stream) →
     per-transaction policy evaluation → router/alert/queue actions.

Also refreshes simulated bank health in demo mode and runs a short
self-demonstrating scenario on boot so dashboards/audit logs show life
immediately.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
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


from config import (audit_path, is_demo_mode, prediction_interval_seconds)

logger = logging.getLogger(__name__)


class MonitorAgent:
    def __init__(self, demo_mode: bool | None = None):
        self.demo_mode = is_demo_mode() if demo_mode is None else demo_mode
        from agent.alert_agent import AlertAgent
        from agent.router_agent import RouterAgent
        from models.bank_health_scorer.scorer import BankHealthScorer
        from models.ensemble import EnsemblePredictor
        from policy.bank_rules import BankRulesEngine
        from policy.gateway_rules import GatewayRulesEngine
        from policy.sre_alert_rules import SREAlertRules

        self.predictor = EnsemblePredictor()
        self.health_scorer = BankHealthScorer()
        self.bank_rules = BankRulesEngine()
        self.gateway_rules = GatewayRulesEngine()
        self.sre_rules = SREAlertRules()
        self.alert_agent = AlertAgent()
        self.router_agent = RouterAgent()

        self.consecutive_high_prob = 0
        self.current_outage_prob = 0.0
        self.running = True
        self._boot_done = False

        logger.info("MonitorAgent initialised "
                    f"(demo_mode={self.demo_mode})")

    # ── lifecycle ───────────────────────────────────────────────────
    async def run(self) -> None:
        logger.info("MonitorAgent starting…")
        await asyncio.gather(self._prediction_loop(), self._stream_monitor_loop())

    def stop(self) -> None:
        self.running = False

    # ── prediction loop ─────────────────────────────────────────────
    async def _prediction_loop(self) -> None:
        interval = 10 if self.demo_mode else prediction_interval_seconds()
        while self.running:
            try:
                prediction = await self._run_prediction()
                self.current_outage_prob = float(prediction["final_probability"])

                # sustained-risk bookkeeping (per-sample ≈ 1)
                if self.current_outage_prob > 0.60:
                    self.consecutive_high_prob += 1
                else:
                    self.consecutive_high_prob = 0

                bank_scores = self.health_scorer.get_all_bank_scores()
                should_alert, reason = self.sre_rules.should_alert_sre(
                    outage_probability=self.current_outage_prob,
                    bank_health_scores=bank_scores,
                    consecutive_high_prob_count=self.consecutive_high_prob)

                if should_alert:
                    await self.alert_agent.send_sre_alert(
                        probability=self.current_outage_prob,
                        reason=reason, prediction_details=prediction)

                self._log_prediction(prediction)
                logger.info(f"Prediction: {self.current_outage_prob:.2%} "
                            f"[{prediction.get('risk_level')}]")

                if self.demo_mode and not self._boot_done:
                    await self._demo_boot_scenario()
                    self._boot_done = True
            except Exception as exc:
                logger.error(f"Prediction loop error: {exc}")
            await asyncio.sleep(interval)

    async def _run_prediction(self) -> dict:
        prediction = self.predictor.get_mock_prediction()
        try:
            if any(self.predictor.models_loaded.values()):
                prediction = self.predictor.predict(
                    sequence_data=self._get_last_24hr_sequence(),
                    realtime_features=self._get_realtime_features())
        except Exception as exc:
            logger.warning(f"Live prediction failed ({exc}) — demo signal used")
        return prediction

    # ── stream loop ─────────────────────────────────────────────────
    async def _stream_monitor_loop(self) -> None:
        from data_pipeline.kafka_consumer import UPIRadarConsumer

        consumer = UPIRadarConsumer()
        queue: asyncio.Queue = asyncio.Queue(maxsize=5000)

        def _feed() -> None:
            try:
                for record, source in consumer.stream():
                    if not self.running:
                        break
                    queue.put_nowait(record)
            except Exception as exc:
                logger.error(f"Stream feed died: {exc}")

        thread = threading.Thread(target=_feed, daemon=True)
        thread.start()

        while self.running:
            try:
                record = await queue.get()
                asyncio.create_task(self._process_transaction(record))
            except Exception as exc:
                logger.error(f"Stream monitor error: {exc}")
                await asyncio.sleep(2)

    # ── transaction pipeline ────────────────────────────────────────
    async def _process_transaction(self, transaction: dict) -> None:
        try:
            bank_scores = self.health_scorer.get_all_bank_scores()
            transaction = dict(transaction)
            transaction.setdefault("bank", "SBI")
            transaction.setdefault("method", "upi")
            transaction.setdefault("amount", 50000)

            bank_decision = self.bank_rules.evaluate(
                transaction=transaction,
                bank_health_scores=bank_scores,
                outage_probability=self.current_outage_prob,
                anomaly_detected=self._is_anomalous(transaction))

            gateway_decision = self.gateway_rules.evaluate(
                current_method=transaction.get("method", "upi"),
                method_health_scores=self._get_method_health(),
                outage_probability=self.current_outage_prob)

            action_required = (bank_decision.action != "PROCEED"
                               or gateway_decision.action != "PROCEED")
            routed = None
            if action_required:
                routed = await self.router_agent.route(
                    transaction=transaction,
                    bank_decision=bank_decision,
                    gateway_decision=gateway_decision)
                if bank_decision.requires_sre_alert \
                        or gateway_decision.requires_sre_alert:
                    await self.alert_agent.send_merchant_alert(
                        transaction_id=transaction.get("id")
                        or transaction.get("transaction_id"),
                        bank_decision=bank_decision,
                        gateway_decision=gateway_decision)
            else:
                self._log_stream_entry(transaction, "PROCEED", "healthy path")

            if routed is not None:
                self._log_stream_entry(
                    transaction, bank_decision.action,
                    bank_decision.reason, extra={"routing_result": routed})
        except Exception as exc:
            logger.error(f"Transaction pipeline error: {exc}")

    # ── demo helpers ────────────────────────────────────────────────
    def _get_method_health(self) -> dict:
        """Razorpay rail health — real API in prod; time-aware mock here."""
        import numpy as np

        seed = int(datetime.now().timestamp() // 60)
        rng = np.random.default_rng(seed)
        health = {"upi": 95.0, "card": 98.0, "netbanking": 92.0,
                  "wallet": 97.0, "neft_imps": 99.0}
        for method in health:
            health[method] = round(max(0, health[method] + rng.uniform(-4, 4)), 1)
        # when model risk is high, UPI rail health drops (leading indicator)
        if self.current_outage_prob > 0.55:
            health["upi"] = round(health["upi"] * 0.55, 1)
        return {k: {"score": v} for k, v in health.items()}

    def _is_anomalous(self, transaction: dict) -> bool:
        return bool(transaction.get("needs_attention", False)) \
            and self.current_outage_prob > 0.5

    def _get_last_24hr_sequence(self):
        """Production: last 24h from DB. Demo: simulated window."""
        import numpy as np

        rng = np.random.default_rng(0)
        return rng.normal(0, 1, (24, 18))

    def _get_realtime_features(self) -> dict:
        import numpy as np

        rng = np.random.default_rng(int(datetime.now().timestamp() // 60))
        return {"failure_rate": float(rng.uniform(0.0, 0.06)),
                "volume_vs_daily_avg": float(rng.uniform(0.3, 1.9)),
                "bank_health_score": float(rng.uniform(50, 100)),
                "failure_rate_1hr_avg": float(rng.uniform(0.0, 0.05)),
                "hour_of_day": datetime.now().hour}

    # ── audit ───────────────────────────────────────────────────────
    def _log_prediction(self, prediction: dict) -> None:
        entry = {"timestamp": datetime.now().isoformat(), **prediction}
        try:
            with audit_path("model_predictions.log").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        try:
            from db.models import ModelPrediction, session_scope

            with session_scope() as session:
                session.add(ModelPrediction(
                    prediction_horizon=int(prediction.get("horizon_minutes", 120)),
                    outage_probability=round(float(prediction["final_probability"]), 4),
                    lstm_prob=round(float(prediction.get("lstm_probability", 0)), 4),
                    prophet_prob=round(float(prediction.get("prophet_probability", 0)), 4),
                    anomaly_score=round(float(prediction.get("anomaly_probability", 0)), 4)))
        except Exception as exc:
            logger.debug(f"Prediction DB log skipped: {exc}")

    def _log_stream_entry(self, transaction: dict, action: str,
                          reason: str, extra: dict | None = None) -> None:
        entry = {"timestamp": datetime.now().isoformat(),
                 "transaction_id": transaction.get("id")
                 or transaction.get("transaction_id"),
                 "bank": transaction.get("bank"), "method": transaction.get("method"),
                 "amount": transaction.get("amount"), "action": action,
                 "reason": reason,
                 "outage_prob": round(self.current_outage_prob, 4),
                 **(extra or {})}
        try:
            with audit_path("stream_processing.log").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    # ── demo boot scenario ──────────────────────────────────────────
    async def _demo_boot_scenario(self) -> None:
        """Show a complete alert→route→cascade story right after boot."""
        logger.info("Demo boot scenario: injecting a pre-outage day…")
        # make SBI critical & BOB degraded like a real pre-outage hour
        self.health_scorer.update_bank_score("SBI", 21.5)
        self.health_scorer.update_bank_score("BOB", 44.0)
        self.health_scorer.update_bank_score("PNB", 58.0)
        self.health_scorer.update_bank_score("HDFC", 96.0)
        self.health_scorer.update_bank_score("ICICI", 88.0)
        self.health_scorer.update_bank_score("KOTAK", 93.0)

        scenario_txns = [
            # B1: critical bank → reroute + SRE alert
            {"id": "txn_demo_b1_001", "transaction_id": "txn_demo_b1_001",
             "amount": 500000, "method": "upi", "bank": "SBI"},
            # B4: degraded bank (PNB 58) → reroute to best (KOTAK/HDFC)
            {"id": "txn_demo_b4_001", "transaction_id": "txn_demo_b4_001",
             "amount": 150000, "method": "upi", "bank": "PNB"},
            # cascade: force HDFC fail → ICICI succeeds
            {"id": "txn_demo_cascade_1",
             "transaction_id": "txn_demo_cascade_1",
             "amount": 999000, "method": "card", "bank": "BOB",
             "_force_fail_banks": ["KOTAK"]},
            # healthy path: HDFC with low risk → PROCEED
            {"id": "txn_demo_ok_001", "transaction_id": "txn_demo_ok_001",
             "amount": 25000, "method": "upi", "bank": "HDFC"},
        ]
        for txn in scenario_txns:
            await self._process_transaction(txn)
            await asyncio.sleep(0.3)
        logger.success("Demo boot scenario complete — check "
                       "audit_logs/routing_decisions.log")


def main() -> None:
    """python -m agent.monitor_agent"""
    from config import setup_logging

    setup_logging("monitor_agent")
    agent = MonitorAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.stop()
        logger.info("MonitorAgent stopped")


if __name__ == "__main__":
    main()
