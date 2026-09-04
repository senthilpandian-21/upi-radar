"""Bank routing rules — per-bank transaction routing decisions.

Pure deterministic rules, evaluated top-down; the FIRST match wins.
All routing stays WITHIN Razorpay's ecosystem (bank → bank, method → method).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import critical_health_threshold, warning_health_threshold

logger = logging.getLogger(__name__)

ACTION_PROCEED = "PROCEED"
ACTION_ROUTE = "ROUTE"
ACTION_QUEUE = "QUEUE"
ACTION_BLOCK = "BLOCK"

SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"


@dataclass
class RoutingDecision:
    action: str                            # PROCEED / ROUTE / QUEUE / BLOCK
    target_bank: Optional[str] = None
    target_method: Optional[str] = None
    reason: str = ""
    severity: str = SEVERITY_LOW           # LOW / MEDIUM / HIGH
    requires_sre_alert: bool = False
    rule_id: str = ""
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "target_bank": self.target_bank,
            "target_method": self.target_method,
            "reason": self.reason,
            "severity": self.severity,
            "requires_sre_alert": self.requires_sre_alert,
            "rule_id": self.rule_id,
        }


class BankRulesEngine:
    """Evaluate one transaction against all bank-level rules."""

    CRITICAL_HEALTH = critical_health_threshold()       # 30
    WARNING_HEALTH = warning_health_threshold()         # 70
    HIGH_VALUE_AMOUNT = 1_000_000                       # ₹10,000 in paise
    MIN_ALTERNATE_SCORE = 50.0

    def evaluate(self,
                 transaction: dict,
                 bank_health_scores: dict,
                 outage_probability: float,
                 anomaly_detected: bool = False) -> RoutingDecision:
        """Top-down rule evaluation — first matching rule wins."""
        original_bank = str(transaction.get("bank", "UNKNOWN")).upper()
        amount = float(transaction.get("amount", 0) or 0)
        method = transaction.get("method", "upi")
        bank_score = float(
            bank_health_scores.get(original_bank, {}).get("score", 100) or 100)

        # ── RULE B1: bank critical → reroute everything ─────────────
        if bank_score < self.CRITICAL_HEALTH:
            alt_bank = self._get_best_alternate_bank(original_bank,
                                                     bank_health_scores)
            reason = (f"RULE B1: {original_bank} health score {bank_score:.1f}% "
                      f"< {self.CRITICAL_HEALTH:.0f}% critical threshold")
            logger.warning(reason)
            return RoutingDecision(
                action=ACTION_QUEUE if alt_bank == "QUEUE" else ACTION_ROUTE,
                target_bank=None if alt_bank == "QUEUE" else alt_bank,
                target_method=method,
                reason=reason, severity=SEVERITY_HIGH,
                requires_sre_alert=True, rule_id="B1")

        # ── RULE B2: high outage prob + high value → NEFT/IMPS ─────
        if outage_probability > 0.70 and amount > self.HIGH_VALUE_AMOUNT:
            alt_bank = self._get_best_alternate_bank(original_bank,
                                                     bank_health_scores)
            reason = (f"RULE B2: outage probability {outage_probability:.0%} + "
                      f"high-value txn ₹{amount/100:,.0f}")
            logger.warning(reason)
            return RoutingDecision(
                action=ACTION_QUEUE if alt_bank == "QUEUE" else ACTION_ROUTE,
                target_bank=None if alt_bank == "QUEUE" else alt_bank,
                target_method="neft_imps",
                reason=reason, severity=SEVERITY_HIGH,
                requires_sre_alert=True, rule_id="B2")

        # ── RULE B3: live anomaly spike ─────────────────────────────
        if anomaly_detected:
            alt_bank = self._get_best_alternate_bank(original_bank,
                                                     bank_health_scores)
            reason = "RULE B3: real-time anomaly spike detected in stream"
            logger.warning(reason)
            return RoutingDecision(
                action=ACTION_QUEUE if alt_bank == "QUEUE" else ACTION_ROUTE,
                target_bank=None if alt_bank == "QUEUE" else alt_bank,
                target_method=method,
                reason=reason, severity=SEVERITY_HIGH,
                requires_sre_alert=True, rule_id="B3")

        # ── RULE B4: bank degraded (warning zone) ──────────────────
        if bank_score < self.WARNING_HEALTH:
            alt_bank = self._get_best_alternate_bank(original_bank,
                                                     bank_health_scores)
            reason = (f"RULE B4: {original_bank} health score {bank_score:.1f}% "
                      f"< {self.WARNING_HEALTH:.0f}% → route via {alt_bank}")
            logger.info(reason)
            return RoutingDecision(
                action=ACTION_QUEUE if alt_bank == "QUEUE" else ACTION_ROUTE,
                target_bank=None if alt_bank == "QUEUE" else alt_bank,
                target_method=method,
                reason=reason, severity=SEVERITY_MEDIUM,
                requires_sre_alert=False, rule_id="B4")

        # ── RULE B5: moderate outage probability ────────────────────
        if outage_probability > 0.40:
            alt_bank = self._get_best_alternate_bank(original_bank,
                                                     bank_health_scores)
            reason = (f"RULE B5: moderate outage probability "
                      f"{outage_probability:.0%} → route via {alt_bank}")
            logger.info(reason)
            return RoutingDecision(
                action=ACTION_QUEUE if alt_bank == "QUEUE" else ACTION_ROUTE,
                target_bank=None if alt_bank == "QUEUE" else alt_bank,
                target_method=method,
                reason=reason, severity=SEVERITY_MEDIUM,
                requires_sre_alert=False, rule_id="B5")

        # ── DEFAULT: healthy path ───────────────────────────────────
        return RoutingDecision(
            action=ACTION_PROCEED, target_bank=original_bank,
            target_method=method,
            reason=f"All health checks passed for {original_bank} "
                   f"(score {bank_score:.1f}%, P(outage) "
                   f"{outage_probability:.1%})",
            severity=SEVERITY_LOW, requires_sre_alert=False, rule_id="DEFAULT")

    # ── alternate bank selection ────────────────────────────────────
    def _get_best_alternate_bank(self, exclude_bank: str,
                                 scores: dict) -> str:
        """Highest-health bank ≠ current. 'QUEUE' if none is healthy."""
        candidates = [
            (bank, float(data.get("score", 0) or 0))
            for bank, data in scores.items()
            if bank != exclude_bank and float(data.get("score", 0) or 0) >= self.MIN_ALTERNATE_SCORE
        ]
        if not candidates:
            return "QUEUE"
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0]
