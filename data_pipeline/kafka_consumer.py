"""Kafka consumer — wraps kafka-python with a generator + demo fallback.

When Kafka (or the broker) is unavailable the consumer iterates over a
simulated transaction stream so agents can be demoed live. Callers simply
do ``for message in consumer.stream(): ...`` in either mode.
"""
from __future__ import annotations

import json
import time
import uuid
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


from config import is_demo_mode, kafka_bootstrap_servers, kafka_topics
from data.synthetic.synthetic_generator import BANK_BASE_TD

try:
    from kafka import KafkaConsumer as _KafkaConsumer

    KAFKA_AVAILABLE = True
except Exception:  # pragma: no cover
    _KafkaConsumer = None
    KAFKA_AVAILABLE = False

# probability a simulated transaction needs attention (demo stream)
DEMO_ROUTING_PROBABILITY = 0.35
DEMO_BANKS = list(BANK_BASE_TD.keys())
DEMO_METHODS = ["upi", "card", "netbanking", "wallet"]


class UPIRadarConsumer:
    def __init__(self, topic: str | None = None, group_id: str = "upi_radar_group"):
        self.topic = topic or kafka_topics()["transactions"]
        self.bootstrap_servers = kafka_bootstrap_servers()
        self.group_id = group_id
        self.demo_mode = is_demo_mode()
        self._consumer = None

    # ── connectivity ────────────────────────────────────────────────
    def _connect(self):
        if not KAFKA_AVAILABLE:
            return None
        try:
            return _KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                group_id=self.group_id,
                auto_offset_reset="latest",
                consumer_timeout_ms=4000,
                session_timeout_ms=10000,
                heartbeat_interval_ms=3000,
            )
        except Exception as exc:
            logger.debug(f"Kafka consumer unavailable ({exc}); demo stream active")
            return None

    def is_connected(self) -> bool:
        return KAFKA_AVAILABLE and self._connect() is not None

    # ── stream ──────────────────────────────────────────────────────
    def stream(self, timeout_ms: int = 3000):
        """
        Generator yielding (record: dict, source: str) tuples.
        source: 'kafka' | 'demo'
        """
        if self.demo_mode or not KAFKA_AVAILABLE:
            logger.info("Demo stream active (no Kafka broker required)")
            yield from self._demo_stream()
            return

        consumer = self._connect()
        if consumer is None:
            logger.info("Falling back to demo stream")
            yield from self._demo_stream()
            return

        try:
            while True:
                msg = consumer.poll(timeout_ms=timeout_ms, max_records=64)
                for records in msg.values():
                    for record in records:
                        if record.value:
                            yield record.value, "kafka"
        finally:
            try:
                consumer.close()
            except Exception:
                pass

    # ── demo stream ─────────────────────────────────────────────────
    def _demo_stream(self, max_records: int = 2000) -> iter:
        """Realistic simulated transactions with bursty volume."""
        rng = __import__("numpy").random.default_rng(int(time.time()) % 1000)
        count = 0
        while count < max_records:
            bank = rng.choice(DEMO_BANKS)
            method = rng.choice(DEMO_METHODS)
            hour = datetime.now().hour
            # evening peak → heavier stream
            if 18 <= hour <= 21:
                batch = rng.integers(3, 8)
            else:
                batch = rng.integers(1, 4)
            for _ in range(int(batch)):
                count += 1
                is_odd = rng.random() < DEMO_ROUTING_PROBABILITY
                yield {
                    "id": f"txn_{uuid.uuid4().hex[:12]}",
                    "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
                    "amount": int(rng.choice([25000, 50000, 120000, 999000, 1500000, 4000000])),
                    "currency": "INR",
                    "method": method,
                    "bank": bank,
                    "customer_id": f"cust_{rng.integers(1000, 9999)}",
                    "simulated": True,
                    "needs_attention": bool(is_odd),
                }, "demo"
                time.sleep(0.15)


if __name__ == "__main__":  # pragma: no cover
    consumer = UPIRadarConsumer()
    for i, (record, source) in zip(range(5), consumer.stream()):
        print(f"[{source}] {record['id']} {record['bank']} ₹{record['amount'] / 100:.2f}")
