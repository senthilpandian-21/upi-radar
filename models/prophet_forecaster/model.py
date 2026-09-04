"""Prophet forecaster — seasonal (calendar) outage-risk signal.

Design: aggregate per-bank hourly failure rates to daily means, fit a
Prophet model per bank with extra regressors for salary-day / month-end /
FY-end / festival flags. The SYSTEM model is fitted on the cross-bank
daily mean and is used by the ensemble for the "seasonal component" of
the next-24h outage probability.

Conversion: a day's forecast failure rate is mapped to a probability by
linear interpolation between a bank's normal-day level and the outage
trigger (5% failure rate) — see ``failure_rate_to_probability``.
"""
from __future__ import annotations

import pickle
from datetime import timedelta

import pandas as pd
from loguru import logger

from config import PROPHET_MODEL_DIR, PROPHET_MODEL_PATH
from data.synthetic.outage_patterns import is_festival_day, is_fy_end, \
    is_month_end, is_salary_day

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except Exception:  # pragma: no cover
    Prophet = None
    PROPHET_AVAILABLE = False

OUTAGE_TRIGGER_RATE = 0.05  # hourly failure rate beyond which = outage
PROPHET_MODEL_SAVE_PATH = PROPHET_MODEL_PATH
FLAG_REGRESSORS = ["is_salary_day", "is_month_end", "is_fy_end", "is_festival_day"]


def prophet_required() -> None:
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet is not installed. Run: pip install -r requirements.txt"
        )


class ProphetOutageModel:
    """Container of per-bank Prophet models (+ a SYSTEM aggregate model)."""

    def __init__(self):
        self.models: dict[str, object] = {}
        self.meta: dict = {}

    # ── data prep ───────────────────────────────────────────────────
    @staticmethod
    def prepare_daily(df: pd.DataFrame, bank_col: str = "bank") -> pd.DataFrame:
        """Aggregate hourly rows → daily rows: ds, y (mean failure rate), bank, flags."""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["ds"] = df["datetime"].dt.normalize()
        out = (df.groupby(["bank", "ds"], as_index=False)
                 .agg(y=("failure_rate", "mean")))
        out["is_salary_day"] = out["ds"].map(lambda d: int(is_salary_day(d)))
        out["is_month_end"] = out["ds"].map(lambda d: int(is_month_end(d)))
        out["is_fy_end"] = out["ds"].map(lambda d: int(is_fy_end(d)))
        out["is_festival_day"] = out["ds"].map(lambda d: int(is_festival_day(d)))
        return out

    # ── build ───────────────────────────────────────────────────────
    def _build_one(self, series: pd.DataFrame) -> object:
        prophet_required()
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            interval_width=0.9,
        )
        for regressor in FLAG_REGRESSORS:
            model.add_regressor(regressor)
        model.fit(series[["ds", "y"] + FLAG_REGRESSORS])
        return model

    def train(self, df: pd.DataFrame, banks: list[str] | None = None,
              fit_system: bool = True) -> "ProphetOutageModel":
        daily = self.prepare_daily(df)
        banks = banks or list(daily["bank"].unique())
        for bank in banks:
            series = daily[daily["bank"] == bank].sort_values("ds")
            logger.info(f"Fitting Prophet for {bank} "
                        f"({len(series)} daily points)…")
            self.models[bank] = self._build_one(series)

        if fit_system:
            system = (daily.groupby("ds", as_index=False)["y"].mean())
            flags = (daily.groupby("ds", as_index=False)[FLAG_REGRESSORS]
                          .first())
            system = system.merge(flags, on="ds").sort_values("ds")
            logger.info(f"Fitting SYSTEM Prophet model ({len(system)} points)…")
            self.models["SYSTEM"] = self._build_one(system)

        self.meta = {"banks": banks, "fit_system": fit_system,
                     "points": len(daily)}
        return self

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: str | None = None) -> str:
        path = path or str(PROPHET_MODEL_PATH)
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"models": self.models, "meta": self.meta}, handle)
        logger.info(f"Prophet models saved → {path} ({len(self.models)} models)")
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "ProphetOutageModel":
        path = path or str(PROPHET_MODEL_PATH)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        instance = cls()
        instance.models = payload["models"]
        instance.meta = payload.get("meta", {})
        return instance

    # ── forecasting ─────────────────────────────────────────────────
    def _future_frame(self, days_ahead: int) -> pd.DataFrame:
        last_ds = max(m.history["ds"].max() for m in self.models.values())
        future = pd.DataFrame({
            "ds": [last_ds + timedelta(days=i + 1) for i in range(days_ahead)]})
        future["is_salary_day"] = future["ds"].map(lambda d: int(is_salary_day(d)))
        future["is_month_end"] = future["ds"].map(lambda d: int(is_month_end(d)))
        future["is_fy_end"] = future["ds"].map(lambda d: int(is_fy_end(d)))
        future["is_festival_day"] = future["ds"].map(lambda d: int(is_festival_day(d)))
        return future

    def forecast(self, bank: str = "SYSTEM",
                 days_ahead: int = 7) -> pd.DataFrame:
        """Forecast daily mean failure rate for the next n days."""
        model = self.models.get(bank)
        if model is None:
            raise KeyError(f"No Prophet model for '{bank}' — trained banks: "
                           f"{list(self.models)}")
        future = self._future_frame(days_ahead)
        forecast = model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def predict_hours(self, bank: str = "SYSTEM",
                      future_periods_hours: int = 48) -> pd.DataFrame:
        """Forecast covering the next ``future_periods_hours`` hours
        (maps to ceil(hours/24) forecast days, flagged hourly)."""
        days = max(1, -(-future_periods_hours // 24))
        daily = self.forecast(bank, days_ahead=days)
        rows = []
        for _, row in daily.iterrows():
            for hour in range(24):
                rows.append({"ds": row["ds"] + timedelta(hours=hour),
                             "yhat": row["yhat"]})
        return pd.DataFrame(rows).head(future_periods_hours)

    # ── probability mapping ─────────────────────────────────────────
    @staticmethod
    def failure_rate_to_probability(failure_rate: float,
                                    normal_level: float = 0.01) -> float:
        """
        Map a forecast daily mean failure rate to P(outage) ∈ [0, 1].
        Linear ramp from the bank's normal level → outage trigger (5%).
        """
        span = max(OUTAGE_TRIGGER_RATE - normal_level, 1e-6)
        prob = (failure_rate - normal_level) / span
        return round(min(max(prob, 0.0), 1.0), 4)

    def probability_for_next_day(self, bank: str = "SYSTEM",
                                 normal_level: float = 0.01) -> float:
        forecast = self.forecast(bank, days_ahead=1)
        return self.failure_rate_to_probability(
            float(forecast["yhat"].iloc[0]), normal_level)


def model_path() -> str:
    return str(PROPHET_MODEL_PATH)


def model_dir() -> str:
    return str(PROPHET_MODEL_DIR)
