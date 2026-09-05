"""Retrainer agent — watches model accuracy and retrains on drift.

Checks labelled predictions weekly (Sunday midnight via APScheduler);
if rolling accuracy falls below 80% it re-runs all three trainers and
logs the event. All heavy work happens in subprocesses so a slow
training run can never block the agent loop.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve()
for _ in range(4):
    if (ROOT / "config.py").exists():
        break
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from config import PROJECT_ROOT, audit_path


ACCURACY_RETRAIN_THRESHOLD = 0.80
MIN_LABELLED_SAMPLES = 20


class RetrainerAgent:
    # ── accuracy ────────────────────────────────────────────────────
    def label_recent_predictions(self, observed_outage: bool) -> int:
        """
        Label the oldest unlabelled prediction that is older than its
        horizon (feedback loop for accuracy tracking). Returns rows updated.
        """
        from db.models import ModelPrediction, session_scope

        updated = 0
        try:
            with session_scope() as session:
                horizon_cutoff = datetime.now() - timedelta(hours=2)
                candidates = (session.query(ModelPrediction)
                              .filter(ModelPrediction.actual_outage.is_(None),
                                      ModelPrediction.predicted_at < horizon_cutoff)
                              .order_by(ModelPrediction.predicted_at.asc())
                              .limit(10).all())
                for row in candidates:
                    prob = float(row.outage_probability or 0)
                    predicted_outage = prob >= 0.5
                    row.actual_outage = observed_outage
                    row.prediction_correct = (predicted_outage == observed_outage)
                    updated += 1
        except Exception as exc:
            logger.warning(f"Could not label predictions: {exc}")
        return updated

    def check_model_accuracy(self, min_samples: int = MIN_LABELLED_SAMPLES) -> dict:
        """Accuracy over labelled predictions; decides if retraining is due."""
        from db.models import ModelPrediction, session_scope

        try:
            with session_scope() as session:
                rows = (session.query(ModelPrediction)
                        .filter(ModelPrediction.prediction_correct.isnot(None))
                        .all())
                total = len(rows)
                if total < min_samples:
                    return {"labelled_samples": total,
                            "accuracy": None,
                            "needs_retrain": False,
                            "reason": f"only {total} labelled samples "
                                      f"(need ≥ {min_samples})"}
                correct = sum(1 for r in rows if r.prediction_correct)
                accuracy = correct / total
                needs = accuracy < ACCURACY_RETRAIN_THRESHOLD
                return {"labelled_samples": total, "accuracy": round(accuracy, 4),
                        "needs_retrain": needs,
                        "reason": ("accuracy below threshold → retraining"
                                   if needs else "accuracy healthy")}
        except Exception as exc:
            logger.warning(f"Accuracy check unavailable: {exc}")
            return {"labelled_samples": 0, "accuracy": None,
                    "needs_retrain": False, "reason": str(exc)}

    # ── retraining ──────────────────────────────────────────────────
    def retrain_lstm(self, data_path: str = "data/synthetic/generated_data.csv") -> dict:
        return self._run_trainer("models.lstm_forecaster.train",
                                 ["--data", data_path])

    def retrain_prophet(self, data_path: str = "data/synthetic/generated_data.csv") -> dict:
        return self._run_trainer("models.prophet_forecaster.train",
                                 ["--data", data_path])

    def retrain_anomaly(self, data_path: str = "data/synthetic/generated_data.csv") -> dict:
        return self._run_trainer("models.anomaly_detector.train",
                                 ["--data", data_path])

    def retrain_all(self) -> dict:
        results = {}
        results["lstm"] = self.retrain_lstm()
        results["prophet"] = self.retrain_prophet()
        results["anomaly"] = self.retrain_anomaly()
        self._log_retrain_event(results)
        return results

    def _run_trainer(self, module: str, extra: list[str]) -> dict:
        command = [sys.executable, "-m", module] + extra
        try:
            result = subprocess.run(command, cwd=str(PROJECT_ROOT),
                                    capture_output=True, text=True, timeout=3600)
            return {"module": module, "exit_code": result.returncode,
                    "tail": (result.stdout or "")[-600:]}
        except subprocess.TimeoutExpired:
            return {"module": module, "exit_code": -1, "tail": "timeout"}
        except Exception as exc:
            return {"module": module, "exit_code": -2, "tail": str(exc)}

    def _log_retrain_event(self, results: dict) -> None:
        entry = {"timestamp": datetime.now().isoformat(), "event": "RETRAIN",
                 "results": {k: {"exit_code": v["exit_code"]}
                             for k, v in results.items()}}
        try:
            with audit_path("retrainer.log").open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        logger.success(f"Retrain event logged: {entry['results']}")

    # ── scheduler (weekly, Sunday 00:15) ────────────────────────────
    def run_scheduler(self) -> None:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BlockingScheduler()
        scheduler.add_job(self._weekly_check, CronTrigger(day_of_week="sun",
                                                          hour=0, minute=15),
                          id="weekly_retrain_check")
        logger.info("Retrainer agent scheduled (weekly, Sun 00:15)")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Retrainer stopped")

    def _weekly_check(self) -> None:
        status = self.check_model_accuracy()
        logger.info(f"Weekly accuracy check: {status}")
        if status.get("needs_retrain"):
            logger.warning("Accuracy below 80% — triggering retrain pipeline")
            self.retrain_all()
        else:
            logger.info("Models healthy — no retraining needed")


if __name__ == "__main__":  # pragma: no cover
    agent = RetrainerAgent()
    agent._weekly_check()
