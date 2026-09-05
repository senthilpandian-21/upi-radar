"""API endpoint tests via FastAPI TestClient."""
from __future__ import annotations


class TestHealth:
    def test_health_ok(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("ok", "degraded")
        assert "components" in body
        assert "demo_mode" in body

    def test_root(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_healthz(self, test_client):
        assert test_client.get("/healthz").status_code == 200


class TestPrediction:
    def test_predict_outage(self, test_client):
        response = test_client.get("/predict/outage")
        assert response.status_code == 200
        body = response.json()
        assert "final_probability" in body
        assert 0 <= body["final_probability"] <= 1
        assert body["risk_level"] in ("GREEN", "YELLOW", "RED")
        assert "recommendation" in body
        assert "timestamp" in body

    def test_predict_history(self, test_client):
        response = test_client.get("/predict/history", params={"limit": 5})
        assert response.status_code == 200
        assert "predictions" in response.json()


class TestBanks:
    def test_bank_health(self, test_client):
        response = test_client.get("/banks/health")
        assert response.status_code == 200
        body = response.json()
        assert "bank_health_scores" in body
        scores = body["bank_health_scores"]
        assert len(scores) >= 8
        assert "best_available_banks" in body
        assert "critical_banks" in body

    def test_single_bank(self, test_client):
        response = test_client.get("/banks/health/SBI")
        assert response.status_code == 200
        body = response.json()
        assert "score" in body

    def test_unknown_bank_404(self, test_client):
        response = test_client.get("/banks/health/NOTABANK")
        assert response.status_code == 404


class TestRouting:
    def test_route_transaction(self, test_client):
        payload = {"transaction": {"amount": 500000, "method": "upi",
                                   "bank": "SBI"}}
        response = test_client.post("/route/transaction", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["original_bank"] == "SBI"
        assert body["action"] in ("PROCEED", "ROUTE", "QUEUE")

    def test_route_high_value(self, test_client):
        payload = {"transaction": {"transaction_id": "txn_api_1",
                                   "amount": 5_000_000, "method": "upi",
                                   "bank": "HDFC"}}
        response = test_client.post("/route/transaction", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["transaction_id"] == "txn_api_1"

    def test_route_simulate_does_not_execute(self, test_client):
        payload = {"transaction": {"transaction_id": "txn_api_sim",
                                   "amount": 50000, "method": "upi",
                                   "bank": "SBI"},
                   "simulate": True}
        response = test_client.post("/route/transaction", json=payload)
        assert response.status_code == 200
        body = response.json()
        # simulate → decision is reported but nothing is queued/executed
        assert body["queued"] is False
        assert body["success"] in (None, True)

    def test_route_invalid_payload(self, test_client):
        payload = {"transaction": {"amount": -5, "method": "upi",
                                   "bank": "SBI"}}
        response = test_client.post("/route/transaction", json=payload)
        assert response.status_code == 422

    def test_route_flat_body_rejected(self, test_client):
        """Contract is the nested RouteRequest — flat bodies must 422."""
        payload = {"amount": 5000, "method": "upi", "bank": "SBI"}
        response = test_client.post("/route/transaction", json=payload)
        assert response.status_code == 422

    def test_route_history(self, test_client):
        response = test_client.get("/route/history", params={"limit": 10})
        assert response.status_code == 200
        assert "decisions" in response.json()


class TestAlertsAndReports:
    def test_active_alerts(self, test_client):
        response = test_client.get("/alerts/active")
        assert response.status_code == 200
        assert "alerts" in response.json()

    def test_alert_history(self, test_client):
        response = test_client.get("/alerts/history", params={"limit": 10})
        assert response.status_code == 200

    def test_reports_list(self, test_client):
        response = test_client.get("/reports/list")
        assert response.status_code == 200
        assert "reports" in response.json()

    def test_generate_report(self, test_client):
        response = test_client.post("/reports/generate")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["filename"].endswith(".pdf")

    def test_download_missing_report_404(self, test_client):
        response = test_client.get("/reports/download/nope.pdf")
        assert response.status_code == 404
