"""Kafka producer — pushes transaction / telemetry records to Kafka.

Graceful degradation: if no broker is reachable, records are written
to a local JSONL spool file (``audit_logs/kafka_spool.jsonl``) so the
pipeline is still demonstrable end-to-end without Kafka.
"""
from __future__ import annotations

import json
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


from config import audit_path, is_demo_mode, kafka_bootstrap_servers, kafka_topics

try:
    from kafka import KafkaProducer

    KAFKA_AVAILABLE = True
except Exception:  # pragma: no cover
    KafkaProducer = None
    KAFKA_AVAILABLE = False

MAX_SPOOL_BYTES = 50 * 1024 * 1024  # rotate spool at 50 MB


class UPIRadarProducer:
    def __init__(self, bootstrap_servers: str | None = None):
        self.bootstrap_servers = bootstrap_servers or kafka_bootstrap_servers()
        self.topics = kafka_topics()
        self.demo_mode = is_demo_mode()
        self._producer = None

    # ── connection (lazy) ────────────────────────────────────────────
    def _get_producer(self):
        if self._producer is None:
            if not KAFKA_AVAILABLE:
                return None
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all", retries=1, request_timeout_ms=4000,
                    max_block_ms=3000,
                )
            except Exception as exc:
                logger.debug(f"Kafka producer unavailable: {exc}")
                self._producer = None
        return self._producer

    def is_connected(self) -> bool:
        return KAFKA_AVAILABLE and self._get_producer() is not None

    # ── send ─────────────────────────────────────────────────────────
    def send(self, topic_key: str, record: dict) -> bool:
        """Send a single record. topic_key ∈ {'transactions','alerts'}."""
        record = dict(record)
        record.setdefault("_sent_at", datetime.now().isoformat())
        producer = self._get_producer()
        if producer is not None:
            try:
                producer.send(self.topics[topic_key], value=record)
                producer.flush(timeout=3)
                return True
            except Exception as exc:
                logger.debug(f"Kafka send failed ({exc}) → spooling")
        return self._spool(topic_key, record)

    def send_batch(self, topic_key: str, records: list[dict]) -> int:
        sent = 0
        for record in records:
            sent += int(self.send(topic_key, record))
        return sent

    def _spool(self, topic_key: str, record: dict) -> bool:
        """Local JSONL fallback (demo mode)."""
        try:
            path = audit_path("kafka_spool.jsonl")
            if path.exists() and path.stat().st_size > MAX_SPOOL_BYTES:
                path.unlink()
            with path.open("a") as handle:
                handle.write(json.dumps({"topic": self.topics[topic_key], "record": record}) + "\n")
            return True
        except OSError:
            return False

    def close(self) -> None:
        if self._producer is not None:
            try:
                self._producer.close(timeout=3)
            except Exception:
                pass
            self._producer = None


# module-level convenience producer (agents reuse the same instance)
_singleton: UPIRadarProducer | None = None


def get_producer() -> UPIRadarProducer:
    global _singleton
    if _singleton is None:
        _singleton = UPIRadarProducer()
    return _singleton


if __name__ == "__main__":  # pragma: no cover
    producer = get_producer()
    ok = producer.send("transactions", {"id": "demo_1", "bank": "SBI", "amount": 50000})
    print(f"Kafka available: {producer.is_connected()} | sent={ok}")
    producer.close()
