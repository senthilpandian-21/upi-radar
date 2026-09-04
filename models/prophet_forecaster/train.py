"""Train Prophet seasonal models on synthetic (or real) hourly data.

    python -m models.prophet_forecaster.train --data data/synthetic/generated_data.csv

Trains one model per bank + the SYSTEM model → saved_model/prophet_model.pkl
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
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


from config import audit_path
from models.prophet_forecaster.model import ProphetOutageModel, prophet_required

DEFAULT_DATA = "data/synthetic/generated_data.csv"


def train(data_path: str = DEFAULT_DATA,
          banks: list[str] | None = None) -> ProphetOutageModel:
    prophet_required()
    logger.info(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    model = ProphetOutageModel().train(df, banks=banks, fit_system=True)
    path = model.save()
    logger.success(f"Prophet training complete → {path}")

    # quick self-check: next-day probability from SYSTEM model
    try:
        probe = model.probability_for_next_day("SYSTEM")
        logger.info(f"SYSTEM next-day P(outage) probe: {probe:.4f}")
    except Exception as exc:
        logger.warning(f"Probe failed: {exc}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Prophet forecaster")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--banks", nargs="*", default=None,
                        help="Restrict to specific banks (default: all)")
    args = parser.parse_args()
    train(data_path=args.data, banks=args.banks)


if __name__ == "__main__":
    main()
