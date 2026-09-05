"""SRE alert rules — when to page the Razorpay SRE team.

A separate module (from bank/gateway rules) because alerting policy is
tuned differently: it is deliberately *trigger-happy* on multi-bank and
sustained signals, and always returns a human-readable reason.
"""
from __future__ import annotations

# a "high probability sample" every PREDICTION_INTERVAL_MINUTES minutes;
# ≥ 6 consecutive samples ≈ 30 minutes of sustained elevated risk.
PREDICTION_INTERVAL_MINUTES = 5
SUSTAINED_SAMPLE_COUNT = 6


class SREAlertRules:
    SRE_ALERT_THRESHOLD = 0.70
    MULTI_BANK_CRITICAL_SCORE = 50.0
    MULTI_BANK_COUNT = 3

    def should_alert_sre(self,
                         outage_probability: float,
                         bank_health_scores: dict | None = None,
                         consecutive_high_prob_count: int = 0,
                         window_minutes: int = PREDICTION_INTERVAL_MINUTES
                         ) -> tuple[bool, str]:
        """
        Returns (should_alert, reason). Reasons are descriptive on purpose —
        they go straight into the SRE email and the audit log.
        """
        bank_health_scores = bank_health_scores or {}

        # ── Rule S1: near-certain outage ────────────────────────────
        if outage_probability > 0.85:
            return True, (f"CRITICAL: outage probability "
                          f"{outage_probability:.1%} exceeds 85% "
                          f"— treat as imminent")

        # ── Rule S2: ≥3 banks degraded simultaneously ───────────────
        degraded_banks = [
            bank for bank, data in bank_health_scores.items()
            if float(data.get("score", 100) or 100) < self.MULTI_BANK_CRITICAL_SCORE
        ]
        if len(degraded_banks) >= self.MULTI_BANK_COUNT:
            return True, (f"Multiple banks critical/degraded: "
                          f"{', '.join(sorted(degraded_banks))}")

        # ── Rule S3: sustained high risk (≥30 minutes) ──────────────
        if (consecutive_high_prob_count >= SUSTAINED_SAMPLE_COUNT
                and outage_probability > 0.60):
            sustained_minutes = consecutive_high_prob_count * window_minutes
            return True, (f"Sustained high outage risk for {sustained_minutes} "
                          f"minutes ({outage_probability:.1%})")

        return False, ""
