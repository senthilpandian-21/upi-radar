"""Tests for the data generator + model layer (pure, no infra)."""
from __future__ import annotations

import pytest

EXPECTED_ROWS = 365 * 24 * 8  # days × hours × banks = 70,080
EXPECTED_BANKS = ["SBI", "HDFC", "ICICI", "AXIS", "PNB", "BOB", "KOTAK", "YES"]


# ── synthetic generator ──────────────────────────────────────────────
@pytest.fixture(scope="module")
def generated():
    from data.synthetic.synthetic_generator import UPIOutageSimulator

    return UPIOutageSimulator(seed=42).generate_hourly_data(days=365)


class TestSyntheticGenerator:
    def test_row_count(self, generated):
        assert len(generated) == EXPECTED_ROWS

    def test_has_target_column(self, generated):
        assert "is_outage" in generated.columns
        assert set(generated["is_outage"].unique()).issubset({0, 1})

    def test_banks_present(self, generated):
        assert set(generated["bank"].unique()) == set(EXPECTED_BANKS)

    def test_all_feature_columns(self, generated):
        expected = {"datetime", "hour_of_day", "day_of_week", "day_of_month",
                    "month", "year", "is_salary_day", "is_fy_end",
                    "is_month_end", "is_peak_hour", "is_festival_day",
                    "is_monday_morning", "bank", "base_td_rate",
                    "total_volume", "success_count", "failure_count",
                    "failure_rate", "bank_health_score", "is_outage",
                    "failure_rate_1hr_avg", "failure_rate_3hr_avg",
                    "failure_rate_24hr_avg", "volume_vs_daily_avg"}
        assert expected.issubset(set(generated.columns))

    def test_volume_and_failure_sane(self, generated):
        assert (generated["failure_rate"] >= 0).all()
        assert (generated["failure_rate"] <= 1.0).all()
        assert (generated["bank_health_score"] >= 0).all()
        assert (generated["bank_health_score"] <= 100).all()
        assert (generated["total_volume"] > 0).all()

    def test_outages_present(self, generated):
        assert generated["is_outage"].sum() > 0
        assert generated["is_outage"].mean() < 0.10

    def test_known_outage_dates_injected(self, generated):
        known = generated[(generated["datetime"] >= "2026-03-26")
                          & (generated["datetime"] < "2026-03-27")]
        assert len(known) == 24 * 8
        assert known["failure_rate"].max() > 0.05

    def test_rolling_features(self, generated):
        sbi = generated[generated["bank"] == "SBI"].sort_values("datetime")
        assert sbi["failure_rate_1hr_avg"].iloc[-1] == pytest.approx(
            sbi["failure_rate"].iloc[-1])
        assert sbi["failure_rate_3hr_avg"].iloc[-1] == pytest.approx(
            sbi["failure_rate"].iloc[-3:].mean())


# ── bank health scorer ────────────────────────────────────────────────
class TestBankHealthScorer:
    def test_score_range(self):
        from models.bank_health_scorer.scorer import BankHealthScorer

        for _ in range(50):
            score = BankHealthScorer.calculate_score(
                "SBI", td_15min=0.0, td_1hr=0.0, td_historical=0.0)
            assert 0 <= score <= 100
        healthy = BankHealthScorer.calculate_score(
            "SBI", td_15min=0.001, td_1hr=0.002, td_historical=0.009)
        unhealthy = BankHealthScorer.calculate_score(
            "SBI", td_15min=0.09, td_1hr=0.07, td_historical=0.05)
        assert healthy > 80
        assert unhealthy < 30

    def test_weighted_formula(self):
        from models.bank_health_scorer.scorer import BankHealthScorer

        score = BankHealthScorer.calculate_score(
            "SBI", td_15min=0.02, td_1hr=0.04, td_historical=0.01)
        expected = 0.5 * (100 - 20) + 0.3 * (100 - 40) + 0.2 * (100 - 10)
        assert score == pytest.approx(expected)

    def test_status_labels(self):
        from models.bank_health_scorer.scorer import BankHealthScorer

        scorer = BankHealthScorer(use_redis=False)
        assert scorer._score_to_status(95) == "EXCELLENT"
        assert scorer._score_to_status(75) == "GOOD"
        assert scorer._score_to_status(55) == "DEGRADED"
        assert scorer._score_to_status(20) == "CRITICAL"

    def test_simulated_scores_are_realistic(self):
        from models.bank_health_scorer.scorer import BankHealthScorer

        scores = BankHealthScorer(use_redis=False).simulate_bank_scores(seed=1)
        assert len(scores) == 8
        for data in scores.values():
            assert 1 <= data["score"] <= 100

    def test_best_available_banks(self, sample_bank_scores):
        from models.bank_health_scorer.scorer import BankHealthScorer

        scorer = BankHealthScorer(use_redis=False)
        for bank, data in sample_bank_scores.items():
            scorer.update_bank_score(bank, data["score"])
        best = scorer.get_best_available_banks(min_score=50.0)
        assert best[0][0] == "HDFC"
        assert all(score >= 50 for _, score in best)
        best_without = scorer.get_best_available_banks(min_score=50.0,
                                                       exclude=["HDFC"])
        assert best_without[0][0] == "KOTAK"


# ── sequences ─────────────────────────────────────────────────────────
class TestSequences:
    def test_create_sequences_shape(self):
        from models.lstm_forecaster.model import (FEATURE_COLUMNS,
                                                  create_sequences)

        import pandas as pd

        frame = pd.DataFrame({
            **{col: [0.0] * 60 for col in FEATURE_COLUMNS},
            "is_outage": [0, 1] * 30})
        X, y = create_sequences(frame, FEATURE_COLUMNS, "is_outage", seq_len=24)
        assert X.shape == (36, 24, len(FEATURE_COLUMNS))
        assert y.shape == (36,)

    def test_sequences_need_enough_rows(self):
        from models.lstm_forecaster.model import FEATURE_COLUMNS, \
            create_sequences

        import pandas as pd

        frame = pd.DataFrame(
            {**{col: [0.0] * 5 for col in FEATURE_COLUMNS},
             "is_outage": [0] * 5})
        with pytest.raises(ValueError):
            create_sequences(frame, FEATURE_COLUMNS, "is_outage", seq_len=24)


# ── ensemble mock ─────────────────────────────────────────────────────
class TestEnsemble:
    def test_mock_prediction_keys(self):
        from models.ensemble import EnsemblePredictor

        prediction = EnsemblePredictor().get_mock_prediction()
        for key in ["final_probability", "lstm_probability",
                    "prophet_probability", "anomaly_probability",
                    "risk_level", "horizon_minutes"]:
            assert key in prediction
        assert 0 <= prediction["final_probability"] <= 1

    def test_mock_deterministic_within_bucket(self):
        from models.ensemble import EnsemblePredictor

        first = EnsemblePredictor().get_mock_prediction()
        second = EnsemblePredictor().get_mock_prediction()
        assert first["final_probability"] == second["final_probability"]

    def test_risk_levels(self):
        from models.ensemble import EnsemblePredictor

        assert EnsemblePredictor._get_risk_level(0.10) == "GREEN"
        assert EnsemblePredictor._get_risk_level(0.50) == "YELLOW"
        assert EnsemblePredictor._get_risk_level(0.90) == "RED"

    def test_recommendations(self):
        from models.ensemble import EnsemblePredictor

        assert "no action" in EnsemblePredictor.recommendation(0.1).lower()
        assert "monitor" in EnsemblePredictor.recommendation(0.5).lower()
        assert "sre" in EnsemblePredictor.recommendation(0.9).lower()
