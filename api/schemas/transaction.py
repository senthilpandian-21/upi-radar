"""Transaction request/response models (pydantic v1 & v2 compatible)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

VALID_METHODS = {"upi", "card", "netbanking", "wallet", "neft_imps"}


class TransactionIn(BaseModel):
    """An incoming payment attempt that UPI Radar must route."""

    transaction_id: Optional[str] = Field(
        default=None, description="Merchant-generated transaction id")
    amount: int = Field(..., gt=0, le=100_000_000_000,
                        description="Amount in paise (1 INR = 100 paise)")
    currency: str = Field(default="INR", max_length=3)
    method: str = Field(default="upi",
                        description="upi | card | netbanking | wallet | neft_imps")
    bank: str = Field(..., max_length=50,
                      description="Remitter bank / PSP handle (e.g. SBI)")
    customer_id: Optional[str] = Field(default=None, max_length=100)

    # internal (demo) knobs — ignored by policy in production
    _force_fail_banks: Optional[list] = None


class TransactionOut(BaseModel):
    """Result of smart-routing a single transaction."""

    transaction_id: Optional[str] = None
    original_bank: str
    original_method: str
    action: str                      # PROCEED / ROUTE / QUEUE
    routed_to_bank: Optional[str] = None
    routed_to_method: Optional[str] = None
    reason: str
    rule_id: Optional[str] = None
    outage_probability: float = 0.0
    success: Optional[bool] = None
    queued: bool = False
    risk_level: Optional[str] = None
    severity: Optional[str] = None
    timestamp: str = ""


class RouteRequest(BaseModel):
    transaction: TransactionIn
    simulate: bool = Field(default=False,
                           description="Simulate routing without executing")
