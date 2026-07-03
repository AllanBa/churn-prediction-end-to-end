import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check_smoke():
    """Smoke test to ensure the API is up and responding."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_success():
    """Test the inference endpoint with a perfectly valid payload."""
    valid_payload = {
        "Tenure Months": 12,
        "Monthly Charges": 75.5,
        "Total Charges": 906.0,
        "CLTV": 4500.0,
        "Churn Score": 85.0,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "Internet Service": "Fiber optic",
        "contract": "Month-to-month"
    }
    
    response = client.post("/predict", json=valid_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "latency_ms" in data
    assert data["churn_prediction"] in [0, 1]

def test_predict_endpoint_validation_error():
    """Test the endpoint's ability to reject invalid data (Pydantic validation)."""
    invalid_payload = {
        "Tenure Months": -5, 
        "Monthly Charges": "Free", 
    }
    
    response = client.post("/predict", json=invalid_payload)
    
    assert response.status_code == 422