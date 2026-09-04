"""LSTM forecaster — architecture + sequence utilities.

TensorFlow is imported defensively so the rest of UPI Radar keeps
running (demo mode) on machines where TF is not installed. All
functions raise a clear error only when actually invoked.

LSTM architecture:
    LSTM(128, seq) → BatchNorm → Dropout(0.2)
    LSTM(64,  seq) → BatchNorm → Dropout(0.2)
    LSTM(32,  flat) → BatchNorm → Dropout(0.2)
    Dense(16, relu) → Dense(1, sigmoid)   ⇒ P(outage in next window)
"""
from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                         LSTM)
    from tensorflow.keras.models import Sequential

    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    tf = None
    TENSORFLOW_AVAILABLE = False

from config import LSTM_MODEL_DIR, LSTM_MODEL_PATH, LSTM_SCALER_PATH

FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "day_of_month", "month",
    "is_salary_day", "is_fy_end", "is_month_end", "is_peak_hour",
    "is_festival_day", "is_monday_morning",
    "failure_rate", "bank_health_score", "total_volume",
    "failure_rate_1hr_avg", "failure_rate_3hr_avg",
    "failure_rate_24hr_avg", "volume_vs_daily_avg",
    "base_td_rate",
]

SEQUENCE_LENGTH = 24     # last 24 hours of features
PREDICTION_HORIZON = 2   # we predict the 2-hour-ahead risk window
TARGET = "is_outage"
MODEL_SAVE_PATH = LSTM_MODEL_PATH


def tf_required() -> None:
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. Install the 'ml' extras:\n"
            "    pip install -r requirements.txt\n"
            "or run with DEMO_MODE=true to use built-in mock predictions."
        )


def build_lstm_model(input_shape: tuple[int, int] | None = None):
    """Build + compile the outage-prediction LSTM."""
    tf_required()
    input_shape = input_shape or (SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model


def create_sequences(data, feature_cols: list[str] | None = None,
                     target_col: str = TARGET,
                     seq_len: int = SEQUENCE_LENGTH):
    """
    Convert flat rows into sliding-window (X, y) pairs.
    X shape: (n_samples, seq_len, n_features) ; y: (n_samples,)
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    values = data[feature_cols].to_numpy(dtype="float32")
    targets = data[target_col].to_numpy(dtype="float32")

    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len:i])
        y.append(targets[i])
    if not X:
        raise ValueError(f"Not enough rows ({len(values)}) for seq_len={seq_len}")
    return np.array(X), np.array(y)


def make_callbacks(monitor: str = "val_auc", mode: str = "max",
                   save_path=None) -> list:
    """Early stopping + best-model checkpointing callbacks."""
    tf_required()
    save_path = save_path or str(MODEL_SAVE_PATH)
    return [
        EarlyStopping(monitor=monitor, patience=10, restore_best_weights=True,
                      mode=mode),
        ModelCheckpoint(save_path, save_best_only=True, monitor=monitor,
                        mode=mode, verbose=1),
    ]


def get_scaler_path() -> str:
    return str(LSTM_SCALER_PATH)


def model_dir() -> str:
    return str(LSTM_MODEL_DIR)
