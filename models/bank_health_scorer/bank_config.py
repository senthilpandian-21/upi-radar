"""Per-bank baseline technical-decline rates + configuration.

Baselines come from NPCI monthly statistics (mean technical decline %)
and are consumed by the health scorer, simulator and routing rules.
"""
from __future__ import annotations

import os

# bank → baseline technical-decline rate (fraction, e.g. 0.009 = 0.9%)
BANK_BASE_TD: dict[str, float] = {
    "SBI": 0.009,
    "HDFC": 0.0013,
    "ICICI": 0.0101,
    "AXIS": 0.006,
    "PNB": 0.012,
    "BOB": 0.015,
    "KOTAK": 0.004,
    "YES": 0.003,
}

# NPCI member list — every UPI bank Razorpay can route to. Sorted by
# volume share (realistic). Everything stays WITHIN Razorpay's ecosystem.
BANK_VOLUME_SHARE: dict[str, float] = {
    "SBI": 0.20, "HDFC": 0.17, "ICICI": 0.15, "AXIS": 0.10,
    "PNB": 0.07, "BOB": 0.05, "KOTAK": 0.06, "YES": 0.03,
}

ALL_BANKS: list[str] = list(BANK_BASE_TD.keys())


def bank_config() -> dict:
    """Resolve base TD rates from NPCI CSV if present, else defaults."""
    try:
        from data_pipeline.npci_data_loader import NPCIDataLoader

        loader = NPCIDataLoader()
        if loader.list_files():
            rates = loader.bank_base_td_rates()
            if rates:
                merged = dict(BANK_BASE_TD)
                merged.update({k: v for k, v in rates.items() if k in merged})
                return merged
    except Exception:
        pass
    return dict(BANK_BASE_TD)


def get_base_td_rate(bank: str) -> float:
    return BANK_BASE_TD.get(bank.upper(), 0.01)


def is_test_mode() -> bool:
    return "XXX" in os.getenv("RAZORPAY_KEY_ID", "")
