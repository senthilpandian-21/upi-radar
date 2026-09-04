"""Routing endpoints — smart routing decisions with full audit."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas.transaction import TransactionIn, TransactionOut
from config import audit_path
from models.bank_health_scorer.scorer import BankHealthScorer

router = APIRouter()

_scorer: Optional[BankHealthScorer] = None


def get_scorer() -> BankHealthScorer:
    global _scorer
    if _scorer is None:
        _scorer = BankHealthScorer()
    return _scorer


def warm_scores() -> dict:
    scorer = get_scorer()
    scores = scorer.get_all_bank_scores()
    if any(data.get("source") == "default" for data in scores.values()):
        scores = scorer.refresh_from_simulation()
    return scores


@router.post("/transaction", response_model=TransactionOut,
             summary="Route a transaction through the policy engine")
async def route_transaction(transaction: TransactionIn,
                            simulate: bool = False) -> TransactionOut:
    """Evaluates bank rules + gateway rules and executes cascade routing."""
    from models.ensemble import EnsemblePredictor
    from policy.bank_rules import BankRulesEngine
    from policy.gateway_rules import GatewayRulesEngine

    try:  # pydantic v2
        txn = transaction.model_dump()
    except AttributeError:  # pydantic v1
        txn = transaction.dict()
    bank_scores = warm_scores()

    predictor = EnsemblePredictor()
    prediction = predictor.get_mock_prediction()
    try:
        prediction = predictor.predict()
    except Exception:
        pass
    prob = float(prediction["final_probability"])

    bank_rules = BankRulesEngine()
    gateway_rules = GatewayRulesEngine()
    bank_decision = bank_rules.evaluate(txn, bank_scores, prob)
    method_health = {m: {"score": 100.0}
                     for m in ["upi", "card", "netbanking", "wallet", "neft_imps"]}
    gateway_decision = gateway_rules.evaluate(txn.get("method", "upi"),
                                              method_health, prob)

    decision = (bank_decision.action
                if bank_decision.action != "PROCEED"
                else gateway_decision.action)

    # execute
    success: Optional[bool] = None
    queued = False
    if decision == "QUEUE":
        from agent.queue_agent import QueueAgent

        txn_id = await QueueAgent().enqueue(txn, reason=bank_decision.reason)
        queued = bool(txn_id)
    elif decision != "PROCEED" and not simulate:
        from agent.router_agent import RouterAgent

        result = await RouterAgent().route(txn, bank_decision, gateway_decision)
        success = result.get("success")
        queued = result.get("queued", False)
        # pick the executed route for the response
        if result.get("bank_used"):
            # reconstruct a routed response
            return _build_out(transaction, bank_decision, gateway_decision,
                              prob, success, result)

    # log manual/PROCEED paths via router's audit for consistency
    if decision == "PROCEED" and not simulate:
        _log_manual_decision(txn, "PROCEED", bank_decision.reason,
                             prob, success=True)
        success = True

    return _build_out(transaction, bank_decision, gateway_decision,
                      prob, success, queued=queued)


@router.get("/history", summary="Recent routing decisions")
async def route_history(limit: int = Query(20, ge=1, le=200)) -> dict:
    records = _history_from_db(limit)
    source = "db"
    if not records:
        records = _history_from_log(limit)
        source = "audit_log"
    return {"source": source, "count": len(records), "decisions": records}


def _build_out(transaction: TransactionIn, bank_decision,
               gateway_decision, prob: float,
               success: Optional[bool] = None,
               result: Optional[dict] = None,
               queued: bool = False) -> TransactionOut:
    executed_bank = (result or {}).get("bank_used")
    executed_method = (result or {}).get("method_used")
    action = bank_decision.action
    if action == "PROCEED":
        action = gateway_decision.action if gateway_decision.action != "PROCEED" \
            else "PROCEED"
    return TransactionOut(
        transaction_id=transaction.transaction_id or f"txn_{datetime.now().timestamp():.0f}",
        original_bank=transaction.bank.upper(),
        original_method=transaction.method,
        action=action,
        routed_to_bank=executed_bank or bank_decision.target_bank,
        routed_to_method=executed_method or gateway_decision.suggested_method
        or transaction.method,
        reason=(bank_decision.reason or gateway_decision.reason or "proceed"),
        rule_id=bank_decision.rule_id or gateway_decision.rule_id,
        outage_probability=round(prob, 4),
        success=success,
        queued=queued,
        risk_level=_risk(prob),
        severity=bank_decision.severity,
        timestamp=datetime.now().isoformat(),
    )


def _risk(prob: float) -> str:
    return "RED" if prob >= 0.70 else ("YELLOW" if prob >= 0.30 else "GREEN")


def _log_manual_decision(txn: dict, action: str, reason: str,
                         prob: float, success: bool) -> None:
    from agent.router_agent import RouterAgent

    RouterAgent()._log_routing_decision(
        transaction_id=txn.get("transaction_id") or txn.get("id"),
        original_bank=txn.get("bank"), routed_to_bank=txn.get("bank"),
        original_method=txn.get("method"), routed_to_method=txn.get("method"),
        reason=reason, outage_prob_at_time=round(prob, 4),
        bank_health_score=None, success=success)


def _history_from_db(limit: int) -> list[dict]:
    try:
        from db.models import RoutingDecisionRow, session_scope

        with session_scope() as session:
            rows = (session.query(RoutingDecisionRow)
                    .order_by(RoutingDecisionRow.decided_at.desc())
                    .limit(limit).all())
        return [{"id": r.id, "transaction_id": r.transaction_id,
                 "decided_at": r.decided_at.isoformat() if r.decided_at else None,
                 "original_bank": r.original_bank,
                 "routed_to_bank": r.routed_to_bank,
                 "original_method": r.original_method,
                 "routed_to_method": r.routed_to_method,
                 "reason": r.reason,
                 "outage_prob_at_time": float(r.outage_prob_at_time or 0),
                 "bank_health_score": float(r.bank_health_score or 0)
                 if r.bank_health_score else None,
                 "success": r.success} for r in rows]
    except Exception:
        return []


def _history_from_log(limit: int) -> list[dict]:
    path = audit_path("routing_decisions.log")
    if not path.exists():
        return []
    try:
        lines = [line for line in path.read_text().splitlines()
                 if line.strip()][-limit:]
        return [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError):
        return []
