"""UPI Outage Simulator — generates 1 year of realistic per-bank hourly data.

Output: data/synthetic/generated_data.csv  (365 × 24 × 8 banks = 70,080 rows)

The simulator encodes everything we know about UPI failure dynamics:
  * per-bank technical-decline baselines (from NPCI monthly data),
  * calendar effects (salary days, month end, FY end, festivals, peaks),
  * real 2026 outage days, plus random small outage injections,
  * rolling failure-rate features used by the ML models,
  * a binary ``is_outage`` target.

Usage:
    python data/synthetic/synthetic_generator.py            # 1 year
    python data/synthetic/synthetic_generator.py --days 730
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow `python data/synthetic/synthetic_generator.py` from anywhere in the repo
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic.outage_patterns import (
    RANDOM_OUTAGE_PROBABILITY,
    is_festival_day,
    is_fy_end,
    is_known_outage,
    is_monday_morning,
    is_month_end,
    is_salary_day,
)

# Realistic per-bank technical-decline baselines (NPCI monthly data, 2025-26)
BANK_BASE_TD = {
    "SBI": 0.009,
    "HDFC": 0.0013,
    "ICICI": 0.0101,
    "AXIS": 0.006,
    "PNB": 0.012,
    "BOB": 0.015,
    "KOTAK": 0.004,
    "YES": 0.003,
}

AVG_DAILY_VOLUME = 591_000_000  # ~591M UPI transactions/day (May 2026)


class UPIOutageSimulator:
    """Simulate a full year of hourly per-bank UPI statistics."""

    def __init__(self, seed: int | None = 42):
        self.bank_base_td = dict(BANK_BASE_TD)
        self.avg_daily_volume = AVG_DAILY_VOLUME
        self.rng = np.random.default_rng(seed)

    # ── feature helpers ──────────────────────────────────────────────
    def _volume_multiplier(self, hour: int) -> float:
        """Time-of-day volume scaling."""
        if 18 <= hour <= 21:
            return 1.8
        if 12 <= hour <= 14:
            return 1.3
        if 1 <= hour <= 5:
            return 0.3
        return 1.0

    def _failure_multiplier(self, dt: pd.Timestamp, bank: str) -> float:
        """Calendar-driven failure-rate scaling (base TD × multiplier)."""
        mult = 1.0
        if 18 <= dt.hour <= 21:
            mult *= 1.5                                   # evening peak stress
        if is_salary_day(dt):
            mult *= 2.0                                   # 1st / 7th / 15th
        if is_fy_end(dt):
            mult *= 5.0                                   # FY-end massive load
        if is_month_end(dt):
            mult *= 1.8
        if is_known_outage(dt):
            mult *= 15.0                                  # real outage days
        return mult

    def generate_hourly_data(self, days: int = 365,
                             start_date: pd.Timestamp | None = None,
                             bank_wise: bool = True) -> pd.DataFrame:
        """
        Generate ``days`` of hourly rows per bank.

        start_date defaults to 2025-07-01 so the window covers the known
        March–April 2026 outage dates.
        """
        if start_date is None:
            start_date = pd.Timestamp("2025-07-01")
        start_date = pd.Timestamp(start_date)

        rows: list[dict] = []
        hourly_base = self.avg_daily_volume / 24.0

        for day_offset in range(days):
            day = start_date + pd.Timedelta(days=day_offset)
            for hour in range(24):
                dt = day.replace(hour=hour)

                # calendar volume factor
                date_volume_mult = 1.0
                if is_salary_day(dt):
                    date_volume_mult *= 1.5
                if is_month_end(dt):
                    date_volume_mult *= 1.3
                if is_fy_end(dt):
                    date_volume_mult *= 2.0
                if is_monday_morning(dt):
                    date_volume_mult *= 1.2

                for bank, base_td in self.bank_base_td.items():
                    hourly_volume = hourly_base * self._volume_multiplier(hour) * date_volume_mult
                    if is_festival_day(dt):
                        hourly_volume *= 1.4
                    if not bank_wise:
                        hourly_volume /= len(self.bank_base_td)

                    # random outage injection: realistic 0.2% per bank-hour
                    injected = self.rng.random() < RANDOM_OUTAGE_PROBABILITY

                    failure_rate = base_td * self._failure_multiplier(dt, bank)
                    if injected:
                        failure_rate = min(failure_rate * 20.0, 0.8)

                    is_outage = int(failure_rate > 0.05 or injected)

                    # gaussian noise on the observed failure rate
                    failure_rate *= self.rng.uniform(0.85, 1.15)
                    failure_rate = float(min(max(failure_rate, 0.0), 1.0))

                    failed = int(hourly_volume * failure_rate)
                    success = int(hourly_volume) - failed
                    health = max(0.0, min(100.0, 100.0 - (failure_rate * 1000.0)))

                    rows.append({
                        "datetime": dt,
                        "hour_of_day": hour,
                        "day_of_week": day.dayofweek,
                        "day_of_month": day.day,
                        "month": day.month,
                        "year": day.year,
                        # flags
                        "is_salary_day": int(is_salary_day(dt)),
                        "is_fy_end": int(is_fy_end(dt)),
                        "is_month_end": int(is_month_end(dt)),
                        "is_peak_hour": int(18 <= hour <= 21),
                        "is_festival_day": int(is_festival_day(dt)),
                        "is_monday_morning": int(is_monday_morning(dt)),
                        # bank identity
                        "bank": bank,
                        "base_td_rate": base_td,
                        # volume + failure metrics
                        "total_volume": int(hourly_volume),
                        "success_count": success,
                        "failure_count": failed,
                        "failure_rate": round(failure_rate, 6),
                        "bank_health_score": round(health, 2),
                        # TARGET VARIABLE
                        "is_outage": is_outage,
                    })

        df = pd.DataFrame(rows)
        df = self._add_rolling_features(df)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-bank rolling averages: 1hr, 3hr, 24hr failure rates."""
        df = df.sort_values(["bank", "datetime"]).reset_index(drop=True)
        df["failure_rate_1hr_avg"] = df.groupby("bank")["failure_rate"].transform(
            lambda x: x.rolling(1).mean())
        df["failure_rate_3hr_avg"] = df.groupby("bank")["failure_rate"].transform(
            lambda x: x.rolling(3).mean())
        df["failure_rate_24hr_avg"] = df.groupby("bank")["failure_rate"].transform(
            lambda x: x.rolling(24).mean())
        df["volume_vs_daily_avg"] = df["total_volume"] / (self.avg_daily_volume / 24)
        return df.fillna(0)

    # ── summary stats ────────────────────────────────────────────────
    @staticmethod
    def summarize(df: pd.DataFrame) -> str:
        lines = [
            f"Generated {len(df)} rows",
            f"Outage rate: {df['is_outage'].mean() * 100:.2f}%",
            f"Banks: {', '.join(df['bank'].unique())}",
            f"Date range: {df['datetime'].min()} → {df['datetime'].max()}",
            f"Avg hourly volume: {df['total_volume'].mean():,.0f}",
            f"Mean failure rate: {df['failure_rate'].mean():.5f}",
        ]
        if df["is_outage"].sum() > 0:
            outage_rows = df[df["is_outage"] == 1]
            lines.append(
                f"Outage rows per bank: "
                + ", ".join(f"{b}={n}" for b, n in outage_rows.groupby('bank').size().items())
            )
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic UPI outage data")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=str, default="data/synthetic/generated_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sim = UPIOutageSimulator(seed=args.seed)
    df = sim.generate_hourly_data(days=args.days)
    df.to_csv(args.output, index=False)
    if not args.quiet:
        print(UPIOutageSimulator.summarize(df))
    return df


if __name__ == "__main__":
    main()
