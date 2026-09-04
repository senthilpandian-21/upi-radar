"""Train the LSTM outage forecaster on synthetic (or real) data.

    python -m models.lstm_forecaster.train --data data/synthetic/generated_data.csv

Pipeline: load → scale (persist scaler) → per-bank sequences → 80/10/10 split
→ train with class weights + EarlyStopping(val_auc) → evaluate test set.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
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


from config import LSTM_MODEL_DIR, LSTM_SCALER_PATH
from models.lstm_forecaster.model import (FEATURE_COLUMNS, SEQUENCE_LENGTH,
                                          build_lstm_model, create_sequences,
                                          make_callbacks)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

DEFAULT_DATA = "data/synthetic/generated_data.csv"


def load_and_prepare(data_path: str) -> tuple[pd.DataFrame, StandardScaler | None]:
    """Load CSV and standardise features (in-place on a copy)."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for LSTM training")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["bank", "datetime"]).reset_index(drop=True)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data missing feature columns: {missing}")

    scaler = StandardScaler()
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS].astype("float64"))

    LSTM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(LSTM_SCALER_PATH, "wb") as handle:
        pickle.dump(scaler, handle)
    logger.info(f"Scaler saved → {LSTM_SCALER_PATH}")
    return df, scaler


def build_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-bank sliding windows, stacked into one big (X, y)."""
    all_x, all_y = [], []
    for bank in df["bank"].unique():
        bank_df = df[df["bank"] == bank].reset_index(drop=True)
        X, y = create_sequences(bank_df, FEATURE_COLUMNS, "is_outage", SEQUENCE_LENGTH)
        all_x.append(X)
        all_y.append(y)
    return np.vstack(all_x), np.concatenate(all_y)


def train(data_path: str = DEFAULT_DATA, epochs: int = 100, batch_size: int = 64):
    """Full training run; returns (model, history)."""
    logger.info(f"Loading data: {data_path}")
    df, scaler = load_and_prepare(data_path)

    logger.info("Building per-bank sequences…")
    X, y = build_sequences(df)
    logger.info(f"Sequences: X={X.shape} y={y.shape} "
                f"(outage share {y.mean() * 100:.2f}%)")

    # ── 80/10/10 split ──────────────────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # ── class imbalance ─────────────────────────────────────────────
    positives = max(int(y_train.sum()), 1)
    negatives = max(len(y_train) - int(y_train.sum()), 1)
    class_weight = {0: 1.0, 1: negatives / positives}
    logger.info(f"Class weights: {class_weight}")

    model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=make_callbacks(),
        verbose=1,
    )

    results = model.evaluate(X_test, y_test, verbose=0)
    metric_names = [m.name if hasattr(m, "name") else str(m)
                    for m in model.metrics]
    summary = dict(zip(metric_names, results))

    logger.success("Test metrics:")
    logger.success(f"  Accuracy : {summary.get('accuracy', 0):.4f}")
    logger.success(f"  Precision: {summary.get('precision', 0):.4f}")
    logger.success(f"  Recall   : {summary.get('recall', 0):.4f}")
    logger.success(f"  AUC      : {summary.get('auc', 0):.4f}")
    logger.success(f"  Loss     : {summary.get('loss', 0):.4f}")

    # Save test metrics for the evaluation report
    try:
        import json
        from config import audit_path
        (audit_path("lstm_test_metrics.json")
         .write_text(json.dumps({**summary, "samples": int(len(y_test))}, indent=2)))
    except OSError:
        pass

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM outage forecaster")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(data_path=args.data, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
