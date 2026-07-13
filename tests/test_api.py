"""Smoke + behaviour tests for the diabetes prediction API.

Run:  pytest -q      (from the project root, after `python train.py`)
"""
from fastapi.testclient import TestClient

from app.main import app

LOW_RISK = {
    "Pregnancies": 1, "Glucose": 89, "BloodPressure": 66, "SkinThickness": 23,
    "Insulin": 94, "BMI": 22.0, "DiabetesPedigreeFunction": 0.167, "Age": 21,
}
HIGH_RISK = {
    "Pregnancies": 8, "Glucose": 197, "BloodPressure": 74, "SkinThickness": 45,
    "Insulin": 200, "BMI": 45.0, "DiabetesPedigreeFunction": 1.2, "Age": 58,
}


def client() -> TestClient:
    # `with` triggers the lifespan handler so the model loads.
    return TestClient(app)


def test_health():
    with client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_metadata_exposes_metrics_and_threshold():
    with client() as c:
        body = c.get("/metadata").json()
        assert 0 < body["decision_threshold"] < 1
        assert "test_roc_auc" in body["metrics"]


def test_predict_shape():
    with client() as c:
        body = c.post("/predict", json=LOW_RISK).json()
        assert set(body) >= {"probability", "risk_percent", "risk_level",
                             "prediction", "threshold", "model_name"}
        assert 0.0 <= body["probability"] <= 1.0
        assert body["risk_level"] in {"Low", "Moderate", "High"}


def test_high_risk_scores_above_low_risk():
    with client() as c:
        low = c.post("/predict", json=LOW_RISK).json()["probability"]
        high = c.post("/predict", json=HIGH_RISK).json()["probability"]
        assert high > low


def test_validation_rejects_bad_input():
    with client() as c:
        bad = dict(LOW_RISK, Age=-5)
        assert c.post("/predict", json=bad).status_code == 422
