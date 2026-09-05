"""Evaluate the trained LSTM on the held-out test window.

Computes accuracy / precision / recall / AUC + a confusion matrix and
writes a small JSON report into audit_logs/.
"""
from __future__ import annotations

import argparse
import json

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


from config import LSTM_MODEL_PATH, audit_path
from models.lstm_forecaster.train import build_sequences, load_and_prepare

try:
    from sklearn.metrics import (accuracy_score, auc as roc_auc,
                                 confusion_matrix, precision_recall_fscore_support,
                                 roc_curve)
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False


def evaluate(data_path: str = "data/synthetic/generated_data.csv",
             model_path: str | None = None) -> dict:
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for evaluation")
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"TensorFlow unavailable: {exc}")

    model_path = model_path or str(LSTM_MODEL_PATH)
    model = tf.keras.models.load_model(model_path)

    df, scaler = load_and_prepare(data_path)
    X, y = build_sequences(df)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
                                            stratify=y)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_score = float(roc_auc(fpr, tpr))
    matrix = confusion_matrix(y_test, y_pred).tolist()

    report = {
        "model": str(model_path),
        "test_samples": int(len(y_test)),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(roc_auc_score, 4),
        "confusion_matrix": matrix,
        "threshold": 0.5,
    }
    try:
        (audit_path("lstm_evaluation.json")).write_text(json.dumps(report, indent=2))
    except OSError:
        pass

    logger.info(f"Accuracy={accuracy:.4f} Precision={precision:.4f} "
                f"Recall={recall:.4f} AUC={roc_auc_score:.4f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LSTM forecaster")
    parser.add_argument("--data", default="data/synthetic/generated_data.csv")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data), indent=2))


if __name__ == "__main__":
    main()
