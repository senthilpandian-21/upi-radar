"""Prophet inference CLI.

    python -m models.prophet_forecaster.predict --bank SYSTEM --hours 48
"""
from __future__ import annotations

import argparse

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
for _ in range(4):
    if (ROOT / "config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from models.prophet_forecaster.model import ProphetOutageModel


def predict(bank: str = "SYSTEM", hours: int = 48) -> dict:
    model = ProphetOutageModel.load()
    hourly = model.predict_hours(bank, future_periods_hours=hours)
    next_day_prob = model.probability_for_next_day(bank)
    return {
        "bank": bank,
        "horizon_hours": hours,
        "forecast_points": len(hourly),
        "next_day_outage_probability": next_day_prob,
        "latest_forecast_rows": hourly.tail(6).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prophet inference")
    parser.add_argument("--bank", default="SYSTEM")
    parser.add_argument("--hours", type=int, default=48)
    args = parser.parse_args()
    import json

    print(json.dumps(predict(args.bank, args.hours), indent=2, default=str))


if __name__ == "__main__":
    main()
