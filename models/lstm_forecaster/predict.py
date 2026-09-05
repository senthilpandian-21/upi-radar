"""LSTM inference helper — single-bank 24h window → outage probability.

    python -m models.lstm_forecaster.predict --bank SBI --hours 24
"""
from __future__ import annotations

import argparse
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


from config import LSTM_MODEL_PATH, LSTM_SCALER_PATH
from models.lstm_forecaster.model import (FEATURE_COLUMNS, SEQUENCE_LENGTH,
                                          tf_required)

# Lazy-loaded singletons
_MODEL = None
_SCALER = None


def load_model(model_path: str | None = None):
    """Load (once) the trained Keras model + fitted scaler."""
    global _MODEL, _SCALER
    if _MODEL is None:
        tf_required()
        model_path = model_path or str(LSTM_MODEL_PATH)
        import tensorflow as tf

        _MODEL = tf.keras.models.load_model(model_path)
        with open(LSTM_SCALER_PATH, "rb") as handle:
            _SCALER = pickle.load(handle)
        logger.info(f"LSTM loaded from {model_path}")
    return _MODEL, _SCALER


def predict_outage_probability(last_24h: pd.DataFrame, bank: str | None = None) -> float:
    """
    last_24h: exactly the last 24 hourly rows for one bank, containing
    FEATURE_COLUMNS. Returns P(outage) in (0, 1).
    """
    model, scaler = load_model()
    if len(last_24h) < SEQUENCE_LENGTH:
        raise ValueError(f"Need {SEQUENCE_LENGTH} rows of history, got {len(last_24h)}")
    window = last_24h.tail(SEQUENCE_LENGTH)[FEATURE_COLUMNS].astype("float64")
    scaled = scaler.transform(window)
    x = scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    prob = float(model.predict(x, verbose=0)[0][0])
    return round(prob, 4)


def predict_from_csv(csv_path: str, bank: str | None = None) -> dict:
    """Convenience: predict for the last 24h of each (or one) bank in a CSV."""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["bank", "datetime"])
    banks = [bank] if bank else df["bank"].unique()
    results = {}
    for name in banks:
        bank_df = df[df["bank"] == name]
        prob = predict_outage_probability(bank_df, bank=name)
        results[name] = prob
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM outage probability inference")
    parser.add_argument("--data", default="data/synthetic/generated_data.csv")
    parser.add_argument("--bank", default=None,
                        help="Bank to predict (default: all banks)")
    args = parser.parse_args()
    results = predict_from_csv(args.data, args.bank)
    for bank, prob in results.items():
        print(f"{bank}: P(outage) = {prob:.2%}")


if __name__ == "__main__":
    main()
