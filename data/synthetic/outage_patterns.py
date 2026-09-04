"""Outage event definitions + calendar patterns used by the UPI simulator.

All numbers here mirror publicly reported UPI outage events and
NPCI calendar behaviour (salary days, month end, FY end, festivals).
"""
from __future__ import annotations

from datetime import date, datetime

# ── Real UPI outage days (March–April 2026, injected into synthetic data) ──
KNOWN_OUTAGE_DATES: list[datetime] = [
    datetime(2026, 3, 26),  # NPCI infrastructure failure — 550M vs 591M avg txns
    datetime(2026, 4, 1),   # FY-end load spike outage
    datetime(2026, 4, 7),   # peak-hour degradation (multiple PSPs)
    datetime(2026, 4, 12),  # third outage in 3 weeks
]

# ── National festival days (month, day) → elevated UPI volume ──────────
FESTIVAL_DAYS: set[tuple[int, int]] = {
    (1, 1),    # New Year
    (1, 26),   # Republic Day
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
    (11, 12),  # Diwali (2026 window)
    (11, 13),  # Diwali bank-holiday window
    (12, 25),  # Christmas
}

# ── Volume / failure multipliers (documented in README) ─────────────────
VOLUME_MULTIPLIERS = {
    "evening_peak": (range(18, 22), 1.8),   # 6 PM – 9 PM
    "lunch_peak":   (range(12, 15), 1.3),   # 12 PM – 2 PM
    "night_low":    (range(1, 6), 0.3),     # 1 AM – 5 AM
}

FAILURE_MULTIPLIERS = {
    "evening_peak": 1.5,
    "salary_day":   2.0,
    "fy_end":       5.0,
    "month_end":    1.8,
    "known_outage": 15.0,
    "random_outage": 20.0,
}

# Random non-calendar outage injection probability per bank-hour
RANDOM_OUTAGE_PROBABILITY = 0.002  # 0.2%


def is_known_outage(dt: datetime) -> bool:
    return any(dt.date() == od.date() for od in KNOWN_OUTAGE_DATES)


def is_festival_day(dt: datetime | date) -> bool:
    return (dt.month, dt.day) in FESTIVAL_DAYS


def is_salary_day(dt: datetime | date) -> bool:
    return dt.day in (1, 7, 15)


def is_month_end(dt: datetime | date) -> bool:
    return dt.day >= 28


def is_fy_end(dt: datetime | date) -> bool:
    """Indian FY ends Mar 31 → heavy load from Mar 25 to Apr 2."""
    return (dt.month == 3 and dt.day >= 25) or (dt.month == 4 and dt.day <= 2)


def is_monday_morning(dt: datetime) -> bool:
    return dt.weekday() == 0 and dt.hour == 9
