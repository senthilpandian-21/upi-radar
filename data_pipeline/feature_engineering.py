"""Feature engineering — turns raw hourly bank telemetry into ML features.

All transforms are pure pandas/pandas-adjacent so they run identically
offline (training on synthetic CSVs) and online (streaming snapshots).
"""
from __future__ import annotations

import pickle

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


from config import LSTM_SCALER_PATH

try:
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    StandardScaler = None
    SKLEARN_AVAILABLE = False

ROLLING_WINDOWS = (1, 3, 24)

FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "day_of_month", "month",
    "is_salary_day", "is_fy_end", "is_month_end", "is_peak_hour",
    "is_festival_day", "is_monday_morning",
    "failure_rate", "bank_health_score", "total_volume",
    "failure_rate_1hr_avg", "failure_rate_3hr_avg",
    "failure_rate_24hr_avg", "volume_vs_daily_avg",
    "base_td_rate",
]

TIME_FLAG_COLUMNS = [
    "is_salary_day", "is_fy_end", "is_month_end", "is_peak_hour",
    "is_festival_day", "is_monday_morning",
]


class FeatureEngineer:
    # ── time features ────────────────────────────────────────────────
    @staticmethod
    def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add hour/day/month + calendar-flag columns from a datetime column."""
        df = df.copy()
        if "datetime" not in df.columns:
            if "recorded_at" in df.columns:
                df["datetime"] = pd.to_datetime(df["recorded_at"])
            else:
                raise ValueError("Need a 'datetime' or 'recorded_at' column")

        dt = pd.to_datetime(df["datetime"])
        df["hour_of_day"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
        df["day_of_month"] = dt.dt.day
        df["month"] = dt.dt.month
        df["year"] = dt.dt.year

        df["is_salary_day"] = df["day_of_month"].isin([1, 7, 15]).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
        df["is_fy_end"] = (
            ((df["month"] == 3) & (df["day_of_month"] >= 25))
            | ((df["month"] == 4) & (df["day_of_month"] <= 2))
        ).astype(int)
        df["is_peak_hour"] = ((df["hour_of_day"] >= 18) & (df["hour_of_day"] <= 21)).astype(int)
        df["is_monday_morning"] = (
            (df["day_of_week"] == 0) & (df["hour_of_day"] == 9)
        ).astype(int)

        festival_days = {(1, 1), (1, 26), (8, 15), (10, 2), (11, 12), (11, 13), (12, 25)}
        df["is_festival_day"] = (
            df.apply(lambda r: (int(r["month"]), int(r["day_of_month"])) in festival_days,
                     axis=1)
        ).astype(int)
        return df

    # ── rolling averages ─────────────────────────────────────────────
    @staticmethod
    def calculate_rolling_averages(df: pd.DataFrame, group_col: str = "bank",
                                   value_col: str = "failure_rate") -> pd.DataFrame:
        """Per-bank 1hr / 3hr / 24hr rolling failure-rate averages."""
        df = df.copy().sort_values([group_col, "datetime"] if "datetime" in df.columns
                                   else [group_col])
        rolling = df.groupby(group_col)[value_col]
        for window in ROLLING_WINDOWS:
            df[f"{value_col}_{window}hr_avg"] = rolling.transform(
                lambda x: x.rolling(window, min_periods=1).mean())
        return df.fillna(0)

    # ── bank health score ────────────────────────────────────────────
    @staticmethod
    def calculate_bank_health_score(td_15min: float, td_1hr: float,
                                    td_historical: float) -> float:
        """
        Weighted bank health (0-100, higher = healthier):
          0.5 × last-15-min health + 0.3 × last-1hr health + 0.2 × historical health
        Each component: 100 − td × 1000 (a 10% TD → score 0).
        """
        health_15min = max(0.0, 100.0 - td_15min * 1000)
        health_1hr = max(0.0, 100.0 - td_1hr * 1000)
        health_hist = max(0.0, 100.0 - td_historical * 1000)
        score = 0.50 * health_15min + 0.30 * health_1hr + 0.20 * health_hist
        return round(min(max(score, 0.0), 100.0), 2)

    # ── scaling ──────────────────────────────────────────────────────
    def scale_features(self, df: pd.DataFrame, fit: bool = True,
                       scaler_path: str | None = None) -> pd.DataFrame:
        """StandardScaler on FEATURE_COLUMNS; persists scaler for inference."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn unavailable — skipping feature scaling")
            return df
        scaler_path = scaler_path or str(LSTM_SCALER_PATH)
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        df = df.copy()
        scaler = StandardScaler()
        if fit:
            df[available] = scaler.fit_transform(df[available])
            try:
                with open(scaler_path, "wb") as handle:
                    pickle.dump(scaler, handle)
            except OSError:
                logger.warning(f"Could not persist scaler → {scaler_path}")
        else:
            try:
                with open(scaler_path, "rb") as handle:
                    scaler = pickle.load(handle)
                df[available] = scaler.transform(df[available])
            except (OSError, pickle.PickleError):
                df[available] = (df[available] - df[available].mean()) / \
                                (df[available].std() + 1e-8)
        return df

    # ── full pipeline ────────────────────────────────────────────────
    def full_pipeline(self, df: pd.DataFrame, fit: bool = True,
                      add_rolling: bool = True) -> pd.DataFrame:
        out = df.copy()
        if "hour_of_day" not in out.columns or out["hour_of_day"].isna().all():
            out = self.extract_time_features(out)
        if add_rolling and "failure_rate_1hr_avg" not in out.columns:
            out = self.calculate_rolling_averages(out)
        return self.scale_features(out, fit=fit)

    def save_split(self, df: pd.DataFrame,
                   output_dir: str = "data/processed") -> dict:
        """Split chronologically 80/10/10 → train/val/test CSVs."""
        from pathlib import Path

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = df.sort_values(["bank", "datetime"]).reset_index(drop=True)
        n = len(df)
        cut_train, cut_val = int(n * 0.8), int(n * 0.9)
        splits = {
            "train": df.iloc[:cut_train],
            "val": df.iloc[cut_train:cut_val],
            "test": df.iloc[cut_val:],
        }
        paths = {}
        for name, frame in splits.items():
            path = out_dir / f"{name}_data.csv"
            frame.to_csv(path, index=False)
            paths[name] = str(path)
            logger.info(f"{name}: {len(frame)} rows → {path}")
        return paths


if __name__ == "__main__":  # pragma: no cover
    from data.synthetic.synthetic_generator import UPIOutageSimulator

    frame = UPIOutageSimulator(seed=1).generate_hourly_data(days=7)
    engineer = FeatureEngineer()
    processed = engineer.full_pipeline(frame, fit=True)
    print(processed[["bank", "datetime", "failure_rate", "failure_rate_3hr_avg"]].head())
