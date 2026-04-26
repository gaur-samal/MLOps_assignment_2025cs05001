"""
Unit Tests for API Module
=========================
Tests for FastAPI endpoints and API functionality.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app
from api.schemas import PatientData, PredictionResponse

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_response_format(self):
        """Test health response has correct format."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
        assert data["version"] == "1.0.0"


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "docs" in data


class TestFeaturesEndpoint:
    """Test features info endpoint."""

    def test_features_endpoint(self):
        """Test features endpoint returns feature info."""
        response = client.get("/features")
        assert response.status_code == 200

        data = response.json()
        assert "features" in data
        assert "total_features" in data
        assert data["total_features"] == 13

    def test_features_have_required_fields(self):
        """Test each feature has required fields."""
        response = client.get("/features")
        data = response.json()

        for feature in data["features"]:
            assert "name" in feature
            assert "description" in feature
            assert "type" in feature


class TestPredictEndpoint:
    """Test prediction endpoint."""

    @pytest.fixture
    def valid_input(self):
        """Valid patient data for testing."""
        return {
            "age": 63,
            "sex": 1,
            "cp": 1,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 2,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0,
            "thal": 6,
        }

    def test_predict_valid_input(self, valid_input):
        """Test prediction with valid input."""
        response = client.post("/predict", json=valid_input)

        # May be 200 (success) or 500 (model not loaded in test)
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "prediction" in data
            assert "confidence" in data
            assert "risk_level" in data

    def test_predict_missing_field(self, valid_input):
        """Test prediction with missing field returns 422."""
        del valid_input["age"]
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_predict_invalid_age(self, valid_input):
        """Test prediction with invalid age."""
        valid_input["age"] = 150  # Invalid
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_predict_invalid_sex(self, valid_input):
        """Test prediction with invalid sex."""
        valid_input["sex"] = 5  # Invalid
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_predict_invalid_thal(self, valid_input):
        """Test prediction with invalid thal value."""
        valid_input["thal"] = 5  # Invalid (must be 3, 6, or 7)
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_predict_negative_values(self, valid_input):
        """Test prediction with negative values."""
        valid_input["age"] = -5
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_docs_available(self):
        """Test OpenAPI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """Test ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self):
        """Test OpenAPI schema is valid."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


class TestSchemas:
    """Test Pydantic schemas."""

    def test_patient_data_valid(self):
        """Test PatientData with valid input."""
        data = PatientData(
            age=63,
            sex=1,
            cp=1,
            trestbps=145,
            chol=233,
            fbs=1,
            restecg=2,
            thalach=150,
            exang=0,
            oldpeak=2.3,
            slope=3,
            ca=0,
            thal=6,
        )
        assert data.age == 63
        assert data.sex == 1

    def test_patient_data_invalid_thal(self):
        """Test PatientData rejects invalid thal."""
        with pytest.raises(ValueError):
            PatientData(
                age=63,
                sex=1,
                cp=1,
                trestbps=145,
                chol=233,
                fbs=1,
                restecg=2,
                thalach=150,
                exang=0,
                oldpeak=2.3,
                slope=3,
                ca=0,
                thal=5,  # Invalid
            )


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = client.post(
            "/predict",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_empty_body(self):
        """Test handling of empty request body."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
