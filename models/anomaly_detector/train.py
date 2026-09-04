"""Train the Isolation Forest anomaly detector.

    python -m models.anomaly_detector.train --data data/synthetic/generated_data.csv
"""
from __future__ import annotations

import argparse

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


from models.anomaly_detector.model import OutageAnomalyDetector, sklearn_required

DEFAULT_DATA = "data/synthetic/generated_data.csv"


def train(data_path: str = DEFAULT_DATA) -> OutageAnomalyDetector:
    sklearn_required()
    logger.info(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    detector = OutageAnomalyDetector().fit(df)
    path = detector.save()
    logger.success(f"Anomaly detector trained → {path}")

    # sanity check: predict on 5 random rows incl. at least one outage row
    sample = df.sample(5, random_state=7)
    for _, row in sample.iterrows():
        features = {col: row[col] for col in
                    ["failure_rate", "volume_vs_daily_avg", "bank_health_score",
                     "failure_rate_1hr_avg", "hour_of_day"]}
        prob = detector.predict_probability(features)
        logger.info(f"{row['bank']} fr={row['failure_rate']:.4f} "
                    f"is_outage={row['is_outage']} → anomaly prob {prob:.3f}")
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IsolationForest detector")
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()
    train(args.data)


if __name__ == "__main__":
    main()
