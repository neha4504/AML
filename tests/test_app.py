import os
import pytest
import polars as pl
import warnings
from unittest.mock import patch
from fastapi.testclient import TestClient
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "virajdeshmukh080818")
if not os.environ.get("DAGSHUB_TOKEN"):
    raise EnvironmentError("DAGSHUB_TOKEN not set")
os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.environ["DAGSHUB_TOKEN"])

from FastAPI.app.main import app

@pytest.fixture(scope="module")
def client():
    """single TestClient for all tests. app starts once, model loads once."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def feature_names(client):
    """fetch the feature list the API expects"""
    response = client.get("/features")
    assert response.status_code == 200, f"/features failed: {response.text}"
    return response.json()["features"]


@pytest.fixture(scope="module")
def blank_payload(feature_names):
    """All-zero payload — represents a completely neutral transaction."""
    return {feature: 0.0 for feature in feature_names}


@pytest.fixture(scope="module")
def real_fraud_payload(feature_names):
    df = pl.read_csv(
        "data/test/test_data_sample.csv",
        schema_overrides={"Account_HASHED": pl.Utf8}
    )
    fraud_rows = df.filter(pl.col("Is Laundering") == 1)
    assert len(fraud_rows) > 0, "No fraud rows found in test_data_sample.csv"

    row = fraud_rows.head(1).to_dicts()[0]
    return {
        f: float(row[f]) if row.get(f) is not None else 0.0
        for f in feature_names
        if f in row
    }


@pytest.fixture(scope="module")
def real_safe_payload(feature_names):
    
    df = pl.read_csv(
        "data/test/test_data_sample.csv",
        schema_overrides={"Account_HASHED": pl.Utf8}
    )
    safe_rows = df.filter(pl.col("Is Laundering") == 0)
    assert len(safe_rows) > 0, "No safe rows found in test_data_sample.csv"

    row = safe_rows.head(1).to_dicts()[0]
    return {
        f: float(row[f]) if row.get(f) is not None else 0.0
        for f in feature_names
        if f in row
    }

class TestInfrastructure:

    def test_health_check_ok(self, client):
      
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_features_endpoint_returns_list(self, client, feature_names):
        """feature list is non empty and contains expected AML features"""
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert any("amount" in f.lower() for f in feature_names), f"Expected at least one amount-related feature"

    def test_unknown_route_returns_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404



class TestPrediction:

    def test_predict_response_schema(self, client, blank_payload):
        """response always contains required fields regardless of outcom"""
        response = client.post("/predict", json={"features": blank_payload})
        assert response.status_code == 200
        data = response.json()

        assert "probability" in data
        assert "is_alert" in data
        assert "threshold_used" in data
        assert "emergency_threshold" in data
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_fraud_scores_higher_than_safe(
        self, client, real_fraud_payload, real_safe_payload
    ):
        """
        core model direction test  real fraud row must score higher than
        real safe row. tests model quality, not a hardcoded threshold
        """
        fraud_resp = client.post(
            "/predict", json={"features": real_fraud_payload}
        ).json()
        safe_resp = client.post(
            "/predict", json={"features": real_safe_payload}
        ).json()

        assert fraud_resp["probability"] > safe_resp["probability"], (
            f"Fraud row scored {fraud_resp['probability']:.4f} but "
            f"safe row scored {safe_resp['probability']:.4f} — model direction wrong"
        )

    def test_predict_real_fraud_triggers_alert(self, client, real_fraud_payload):
        """real fraud row from holdout data must cross the alert threshold."""
        response = client.post("/predict", json={"features": real_fraud_payload})
        assert response.status_code == 200
        data = response.json()

        assert data["is_alert"] is True, (
            f"Real fraud row did not trigger alert. "
            f"probability={data['probability']:.4f}, "
            f"threshold={data['threshold_used']:.4f}"
        )
        assert data["alert_tier"] in ("TIER_1_EMERGENCY", "TIER_2_INVESTIGATE"), f"Unexpected alert tier: {data['alert_tier']}"

    def test_predict_invalid_payload_returns_422(self, client):
        """Malformed request must be rejected before reaching the model."""
        response = client.post("/predict", json={"garbage": "data"})
        assert response.status_code == 422

    def test_predict_missing_features_does_not_crash(self, client):
        """Partial payload should not cause a 500 — API must be resilient."""
        response = client.post("/predict", json={"features": {"anomaly_score": 0.5}})
        assert response.status_code != 500, \
            "Partial payload caused a server crash"



class TestInvestigation:

    def test_investigate_safe_tx_rejected(self, client, real_safe_payload):
        """
        transactions below investigation threshold must be rejected with 400.
        LLM should never be called for low-risk transactions.
        """
        response = client.post("/investigate", json={"features": real_safe_payload})
        assert response.status_code == 400
        assert "below the investigation" in response.json()["detail"].lower()


    def test_investigate_fraud_tx_returns_report(self, client, real_fraud_payload):
        """
        High-risk transaction must return an investigation report.
        LLM is mocked — test verifies API contract, not LLM output quality.
        """
        mock_report = (
            "INVESTIGATION REPORT: Transaction flagged for structuring behavior "
            "and toxic corridor routing. Recommend immediate SAR filing."
        )

        with patch(
            "FastAPI.app.main.generate_investigation_summary",
            return_value=mock_report
        ) as mock_llm:
            response = client.post(
                "/investigate", json={"features": real_fraud_payload}
            )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        assert mock_llm.called, "LLM was never called for a high-risk transaction"

    def test_investigate_llm_failure_graceful_degradation(
        self, client, real_fraud_payload
    ):
        """
        If LLM service is down (503), API must not crash with 500.
        Production readiness requirement.
        """
        with patch(
            "FastAPI.app.main.generate_investigation_summary",
            side_effect=Exception("LLM service unavailable")
        ):
            response = client.post(
                "/investigate", json={"features": real_fraud_payload}
            )

        assert response.status_code == 503, f"Expected 503 graceful degradation, got {response.status_code}"
        assert response.status_code != 500


class TestModelConsistency:

    def test_deterministic_predictions(self, client, real_fraud_payload):
        """Same input must always return same probability — model is stateless."""
        resp1 = client.post("/predict", json={"features": real_fraud_payload}).json()
        resp2 = client.post("/predict", json={"features": real_fraud_payload}).json()

        assert resp1["probability"] == resp2["probability"], "Model returned different scores for identical input — not deterministic"

    def test_probability_is_calibrated(self, client, real_fraud_payload):
        """Platt-scaled probability must always be within [0, 1]."""
        data = client.post(
            "/predict", json={"features": real_fraud_payload}
        ).json()
        assert 0.0 <= data["probability"] <= 1.0