"""Gateway rules — Razorpay service-level routing (method switching).

These are Razorpay-INTERNAL rules: we may move a transaction between
payment rails (UPI ↔ card ↔ netbanking ↔ wallet ↔ NEFT/IMPS) and between
Razorpay-supported banks — never to a competitor gateway.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import critical_health_threshold, warning_health_threshold

METHOD_ORDER = ["card", "netbanking", "wallet", "neft_imps", "upi"]


@dataclass
class ServiceRoutingDecision:
    action: str                        # PROCEED / SWITCH_METHOD / QUEUE / SRE_ALERT
    suggested_method: str = "upi"
    reason: str = ""
    requires_sre_alert: bool = False
    rule_id: str = ""
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"action": self.action, "suggested_method": self.suggested_method,
                "reason": self.reason,
                "requires_sre_alert": self.requires_sre_alert,
                "rule_id": self.rule_id}


class GatewayRulesEngine:
    METHOD_CRITICAL = critical_health_threshold()    # 30
    METHOD_WARNING = warning_health_threshold()      # 70

    def evaluate(self,
                 current_method: str,
                 method_health_scores: dict,
                 outage_probability: float) -> ServiceRoutingDecision:
        current_method = (current_method or "upi").lower()
        current_health = float(
            method_health_scores.get(current_method,
                                     {"score": 100}).get("score", 100))

        # ── RULE G1: current rail critical → switch method ─────────
        if current_health < self.METHOD_CRITICAL:
            alternate = self._suggest_alternate_method(current_method,
                                                       method_health_scores)
            reason = (f"RULE G1: {current_method} health {current_health:.1f}% "
                      f"< {self.METHOD_CRITICAL:.0f}% → switch to {alternate}")
            return ServiceRoutingDecision(action="SWITCH_METHOD",
                                          suggested_method=alternate,
                                          reason=reason,
                                          requires_sre_alert=True, rule_id="G1")

        # ── RULE G2: UPI at risk → card fallback ───────────────────
        if outage_probability > 0.70 and current_method == "upi":
            card_health = float(method_health_scores.get("card",
                                                         {"score": 100}).get("score", 100))
            if card_health >= 50:
                reason = (f"RULE G2: UPI outage probability {outage_probability:.0%} "
                          f"> 70% → switch to card")
                return ServiceRoutingDecision(action="SWITCH_METHOD",
                                              suggested_method="card",
                                              reason=reason,
                                              requires_sre_alert=True, rule_id="G2")

        # ── RULE G3: near-certain outage → guaranteed NEFT/IMPS ────
        if outage_probability > 0.85:
            reason = (f"RULE G3: critical outage probability "
                      f"{outage_probability:.0%} > 85% → NEFT/IMPS")
            return ServiceRoutingDecision(action="SWITCH_METHOD",
                                          suggested_method="neft_imps",
                                          reason=reason,
                                          requires_sre_alert=True, rule_id="G3")

        # ── DEFAULT ─────────────────────────────────────────────────
        return ServiceRoutingDecision(
            action="PROCEED", suggested_method=current_method,
            reason=f"{current_method} rail healthy "
                   f"(score {current_health:.1f}%)",
            rule_id="DEFAULT")

    def _suggest_alternate_method(self, current: str,
                                  scores: dict) -> str:
        """Priority fallback order: card → netbanking → wallet → neft_imps → upi."""
        for method in METHOD_ORDER:
            if method == current:
                continue
            health = float(scores.get(method, {}).get("score", 0) or 0)
            if health >= 50:
                return method
        return "neft_imps"  # ultimate fallback — always inside Razorpay
