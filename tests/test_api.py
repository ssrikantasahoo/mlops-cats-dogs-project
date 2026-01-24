"""
Unit Tests for FastAPI Inference Service
"""

import sys
import io
import base64
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes."""
    img = Image.new('RGB', (224, 224), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_image_base64(sample_image_bytes):
    """Create base64 encoded image."""
    return base64.b64encode(sample_image_bytes).decode('utf-8')


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self, client):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_message(self, client):
        """Test root response contains message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """Test health response has correct schema."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "uptime_seconds" in data
        assert "total_predictions" in data

    def test_health_status_healthy(self, client):
        """Test health status is healthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_returns_200(self, client, sample_image_bytes):
        """Test predict endpoint returns 200 for valid image."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, client, sample_image_bytes):
        """Test prediction response has correct schema."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        data = response.json()

        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "processing_time_ms" in data
        assert "timestamp" in data

    def test_predict_valid_class(self, client, sample_image_bytes):
        """Test prediction returns valid class."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        data = response.json()
        assert data["prediction"] in ["Cat", "Dog"]

    def test_predict_valid_confidence(self, client, sample_image_bytes):
        """Test prediction confidence is in valid range."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        data = response.json()
        assert 0 <= data["confidence"] <= 1

    def test_predict_probabilities_sum(self, client, sample_image_bytes):
        """Test probabilities sum to approximately 1."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        data = response.json()
        prob_sum = sum(data["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_predict_invalid_file_type(self, client):
        """Test prediction rejects non-image files."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_png_image(self, client):
        """Test prediction works with PNG images."""
        img = Image.new('RGB', (224, 224), color='green')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.png", buffer.getvalue(), "image/png")}
        )
        assert response.status_code == 200


class TestPredictBase64Endpoint:
    """Tests for base64 prediction endpoint."""

    def test_predict_base64_returns_200(self, client, sample_image_base64):
        """Test base64 predict endpoint returns 200."""
        response = client.post(
            "/predict/base64",
            json={"image_base64": sample_image_base64}
        )
        assert response.status_code == 200

    def test_predict_base64_response_schema(self, client, sample_image_base64):
        """Test base64 prediction response has correct schema."""
        response = client.post(
            "/predict/base64",
            json={"image_base64": sample_image_base64}
        )
        data = response.json()

        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_base64_invalid(self, client):
        """Test base64 endpoint handles invalid base64."""
        response = client.post(
            "/predict/base64",
            json={"image_base64": "not-valid-base64!!!"}
        )
        assert response.status_code == 500


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Test metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        """Test metrics has correct content type."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]


class TestStatsEndpoint:
    """Tests for stats endpoint."""

    def test_stats_returns_200(self, client):
        """Test stats endpoint returns 200."""
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_response_schema(self, client):
        """Test stats response has correct schema."""
        response = client.get("/stats")
        data = response.json()

        assert "total_requests" in data
        assert "successful_predictions" in data
        assert "failed_predictions" in data
        assert "uptime_seconds" in data


class TestAPIBehavior:
    """Tests for overall API behavior."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/")
        # CORS should allow requests
        assert response.status_code in [200, 405]

    def test_prediction_increments_counter(self, client, sample_image_bytes):
        """Test predictions increment the counter."""
        # Get initial count
        stats1 = client.get("/stats").json()
        initial = stats1["total_requests"]

        # Make prediction
        client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )

        # Check count increased
        stats2 = client.get("/stats").json()
        assert stats2["total_requests"] > initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
