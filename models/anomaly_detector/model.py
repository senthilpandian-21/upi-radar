"""Isolation Forest anomaly detector for the real-time stream.

Fitted on *normal* operating data so that any deviation in
failure rate / volume / health shows up as an anomaly. The decision
function is converted to a 0-1 probability via a logistic transform.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from loguru import logger

from config import ISO_MODEL_DIR, ISO_MODEL_PATH

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    IsolationForest = None
    SKLEARN_AVAILABLE = False

ANOMALY_FEATURES = [
    "failure_rate",
    "volume_vs_daily_avg",
    "bank_health_score",
    "failure_rate_1hr_avg",
    "hour_of_day",
]

ISO_MODEL_SAVE_PATH = ISO_MODEL_PATH
CONTAINATION = 0.05


def sklearn_required() -> None:
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for anomaly detection")


class OutageAnomalyDetector:
    """Thin wrapper: fit → predict_probability, with persistence."""

    def __init__(self):
        self.model = None

    # ── fit ─────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "OutageAnomalyDetector":
        sklearn_required()
        normal = df[df.get("is_outage", 0) == 0] if "is_outage" in df.columns else df
        features = self._feature_matrix(normal)
        features = pd.DataFrame(features, columns=ANOMALY_FEATURES) if isinstance(features, np.ndarray) else features
        self.model = IsolationForest(
            contamination=CONTAINATION,
            random_state=42,
            n_estimators=200,
            max_samples="auto",
        )
        self.model.fit(features)
        logger.info(f"IsolationForest fitted on {len(features)} normal rows")
        return self

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in ANOMALY_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing anomaly features: {missing}")
        return df[ANOMALY_FEATURES].astype("float64").to_numpy()

    # ── probability ─────────────────────────────────────────────────
    def predict_probability(self, features: dict | pd.DataFrame) -> float:
        """
        Anomaly probability in (0, 1).
        decision_function < 0 ⇒ anomaly; sigmoid maps −∞..+∞ → 0..1.
        """
        sklearn_required()
        if self.model is None:
            raise RuntimeError("Model not fitted — run train() first")
        if isinstance(features, dict):
            frame = pd.DataFrame([features])
        else:
            frame = features
        matrix = self._feature_matrix(frame)
        if isinstance(frame, pd.DataFrame):
            matrix = frame[ANOMALY_FEATURES].astype("float64").to_numpy()
        score = float(self.model.decision_function(matrix)[0])
        prob = 1.0 / (1.0 + np.exp(score))
        return round(float(prob), 4)

    # ── persistence ─────────────────────────────────────────────────
    def save(self, path: str | None = None) -> str:
        path = path or str(ISO_MODEL_PATH)
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self.model, handle)
        logger.info(f"IsolationForest saved → {path}")
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "OutageAnomalyDetector":
        path = path or str(ISO_MODEL_PATH)
        instance = cls()
        with open(path, "rb") as handle:
            instance.model = pickle.load(handle)
        return instance


def anomaly_model_dir() -> str:
    return str(ISO_MODEL_DIR)
