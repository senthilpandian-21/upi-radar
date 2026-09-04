"""Anomaly detector inference helper (used by the ensemble at runtime)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
for _ in range(4):
    if (ROOT / "config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from models.anomaly_detector.model import (ANOMALY_FEATURES,
                                           OutageAnomalyDetector)

# lazy singleton
_DETECTOR: OutageAnomalyDetector | None = None


def load_detector() -> OutageAnomalyDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = OutageAnomalyDetector.load()
    return _DETECTOR


def predict_anomaly_probability(features: dict) -> float:
    """features must contain ANOMALY_FEATURES keys."""
    return load_detector().predict_probability(features)


def feature_keys() -> list[str]:
    return list(ANOMALY_FEATURES)
