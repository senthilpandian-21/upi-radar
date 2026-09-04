"""Queue rules — when to queue a transaction and how long to retry.

Graceful degradation: during short degradation windows a transaction is
queued (rather than failed), then replayed with exponential backoff.
"""
from __future__ import annotations

import time

from config import queue_max_wait_seconds

# A Razorpay payment call that does not respond within 5s is treated as
# "infrastructure under stress" → queue it instead of failing it.
RESPONSE_TIMEOUT_MS = 5000

MAX_RETRIES = 5

# exponential backoff: retry n (1-based) → delay seconds
BACKOFF_DELAYS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 120}


class QueueRules:
    @staticmethod
    def should_queue(transaction: dict | None = None,
                     razorpay_response_time_ms: float = 0.0,
                     outage_probability: float = 0.0) -> bool:
        """Queue when the gateway times out OR outage risk is very high."""
        if razorpay_response_time_ms > RESPONSE_TIMEOUT_MS:
            return True
        # brief severe-degradation window: queue low-priority volume
        if outage_probability > 0.90 and not transaction:
            return True
        if transaction:
            amount = float(transaction.get("amount", 0) or 0)
            method = str(transaction.get("method", "upi")).lower()
            if outage_probability > 0.90 and amount < 100_000:
                return True  # small txns can wait a few seconds
            if method == "neft_imps" and outage_probability > 0.95:
                return True
        return False

    @staticmethod
    def get_retry_delay(retry_count: int) -> int:
        """Exponential backoff delay in seconds (retry_count is 0-based)."""
        return BACKOFF_DELAYS.get(retry_count + 1, 120)

    @staticmethod
    def should_abandon(queued_at: float | None = None,
                       retry_count: int = 0,
                       max_retries: int = MAX_RETRIES,
                       queued_at_iso: str | None = None) -> bool:
        """Give up: queued beyond max-wait AND retries exhausted.

        After abandoning we recommend the merchant try an alternate method —
        the failure is surfaced loudly instead of silently dropped.
        """
        if retry_count >= max_retries:
            return True
        if queued_at_iso:
            try:
                from datetime import datetime

                queued_ts = datetime.fromisoformat(queued_at_iso).timestamp()
                queued_at = queued_ts
            except (ValueError, TypeError):
                pass
        if queued_at is not None:
            if (time.time() - queued_at) > queue_max_wait_seconds() \
                    and retry_count >= max_retries // 2:
                return True
        return False

    @staticmethod
    def recommend_alternate(transaction: dict) -> str:
        """After abandon: best alternate rail for the merchant."""
        method = str(transaction.get("method", "upi")).lower()
        alternates = [m for m in METHOD_ORDER_FALLBACK if m != method]
        return alternates[0] if alternates else "upi"


METHOD_ORDER_FALLBACK = ["card", "netbanking", "wallet", "neft_imps", "upi"]
