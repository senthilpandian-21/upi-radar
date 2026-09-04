"""Razorpay downtime collector — polls GET /v1/payments/downtimes every 5 min.

Uses the official Razorpay Python SDK. In Test Mode the endpoint returns
sandbox downtimes; with no keys configured we fall back to a local cache
so the pipeline never crashes.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

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


from config import audit_dir

try:  # pragma: no cover - environment dependent
    import razorpay

    RAZORPAY_AVAILABLE = True
except Exception:  # pragma: no cover
    RAZORPAY_AVAILABLE = False

POLL_INTERVAL_MINUTES = 5


class RazorpayDowntimeCollector:
    def __init__(self, key_id: str | None = None, key_secret: str | None = None):
        self.key_id = key_id or os.getenv("RAZORPAY_KEY_ID", "")
        self.key_secret = key_secret or os.getenv("RAZORPAY_KEY_SECRET", "")
        self._client = None
        self._cache_file = audit_dir() / "razorpay_downtimes_cache.json"

    # ── client ───────────────────────────────────────────────────────
    @property
    def client(self):
        if self._client is None:
            if not RAZORPAY_AVAILABLE:
                raise RuntimeError("razorpay SDK not installed")
            if not (self.key_id and self.key_secret) or "XXX" in self.key_id:
                raise RuntimeError("Razorpay test keys not configured — run in demo mode")
            self._client = razorpay.Client(auth=(self.key_id, self.key_secret))
        return self._client

    # ── polling ──────────────────────────────────────────────────────
    def poll_downtimes(self) -> list[dict]:
        """
        Fetch current downtime events from the Razorpay API.
        Normalises each event to:
            id, method, bank, severity, status, is_scheduled, begin_time, end_time
        """
        try:
            raw = self.client.payment.downtimes.all()
            events = raw.get("items", []) if isinstance(raw, dict) else (raw or [])
            normalised = [self._normalise(e) for e in events]
            self._cache(normalised)
            logger.info(f"Polled Razorpay downtimes: {len(normalised)} event(s)")
            return normalised
        except Exception as exc:
            logger.warning(f"Razorpay downtime poll failed ({exc}); using local cache")
            return self._load_cache()

    def _normalise(self, event: dict) -> dict:
        return {
            "downtime_id": event.get("id") or event.get("downtime_id"),
            "method": event.get("method", "unknown"),
            "bank": event.get("bank", ""),
            "severity": event.get("severity", "medium"),
            "status": event.get("status", "started"),
            "is_scheduled": bool(event.get("is_scheduled", False)),
            "begin_time": event.get("begin_time"),
            "end_time": event.get("end_time"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── persistence ──────────────────────────────────────────────────
    def save_to_db(self, downtime: dict) -> bool:
        """Persist one event to the razorpay_downtimes table (if DB up)."""
        from db.models import DowntimeEvent, session_scope

        if not downtime.get("downtime_id"):
            return False
        try:
            with session_scope() as session:
                exists = session.query(DowntimeEvent).filter_by(
                    downtime_id=downtime["downtime_id"]).first()
                if exists:
                    for key, value in downtime.items():
                        if hasattr(exists, key) and value is not None:
                            setattr(exists, key, value)
                else:
                    session.add(DowntimeEvent.from_dict(downtime))
            logger.debug(f"Saved downtime {downtime['downtime_id']}")
            return True
        except Exception as exc:
            logger.debug(f"Could not persist downtime to DB ({exc})")
            return False

    def save_all_to_db(self, events: list[dict]) -> int:
        saved = 0
        for event in events:
            saved += int(self.save_to_db(event))
        return saved

    # ── local cache fallback ─────────────────────────────────────────
    def _cache(self, events: list[dict]) -> None:
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(events, indent=2))
        except OSError:
            pass

    def _load_cache(self) -> list[dict]:
        try:
            if self._cache_file.exists():
                return json.loads(self._cache_file.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        return []

    # ── scheduler ────────────────────────────────────────────────────
    def run_scheduler(self) -> None:
        """Poll every 5 minutes via APScheduler (blocking until Ctrl+C)."""
        from apscheduler.schedulers.blocking import BlockingScheduler

        scheduler = BlockingScheduler()
        scheduler.add_job(
            lambda: self.save_all_to_db(self.poll_downtimes()),
            "interval", minutes=POLL_INTERVAL_MINUTES, id="razorpay_downtime_poll",
        )
        logger.info(f"Razorpay downtime collector started (every {POLL_INTERVAL_MINUTES} min)")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Collector stopped")


if __name__ == "__main__":  # pragma: no cover
    collector = RazorpayDowntimeCollector()
    events = collector.poll_downtimes()
    collector.save_all_to_db(events)
    print(f"Fetched {len(events)} downtime events")
