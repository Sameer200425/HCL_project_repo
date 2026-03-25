"""
Integration Tests for Backend (backend.main app)
=================================================
Tests the full authenticated API:
  - Auth (register, login, me, token refresh)
  - Predictions (single, batch, history, stats)
  - Analytics endpoints

Run:
    pytest tests/test_backend.py -v
"""

import sys
import io
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set a stable secret key for testing
os.environ["SECRET_KEY"] = "test-secret-key-for-unit-tests-only"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_model_manager():
    """Create a mock ModelManager that returns fake predictions."""
    manager = MagicMock()
    manager.device = "cpu"
    manager.models = {"cnn": True}
    manager.model_info = {
        "cnn": {"name": "CNN (ResNet50)", "version": "1.0.0", "parameters": 25557032},
        "vit": {"name": "ViT-Base/16", "version": "1.0.0", "parameters": 664},
        "hybrid": {"name": "Hybrid CNN+ViT", "version": "1.0.0", "parameters": 24000000},
    }

    def fake_predict(image, model_type):
        return {
            "class_name": "genuine",
            "confidence": 0.95,
            "probabilities": {"genuine": 0.95, "fraud": 0.02, "tampered": 0.02, "forged": 0.01},
            "risk_level": "LOW",
            "inference_time_ms": 15.3,
        }

    def fake_predict_batch(images, model_type):
        return [fake_predict(img, model_type) for img in images]

    manager.predict = fake_predict
    manager.predict_batch = fake_predict_batch
    manager.get_model = MagicMock(return_value=MagicMock())
    manager.classify_risk = lambda cls, conf: "LOW" if cls == "genuine" else "CRITICAL"
    return manager


@pytest.fixture(scope="module")
def test_client(mock_model_manager):
    """Create a FastAPI TestClient with mocked model manager and in-memory DB."""
    # Patch ModelManager before importing the app
    with patch("backend.main.ModelManager", return_value=mock_model_manager):
        with patch("backend.main.ModelType") as mock_mt:
            mock_mt.CNN = MagicMock(value="cnn")
            from backend.main import app
            from backend.routes_predict import set_model_manager
            from backend.database import get_db, Base
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy.pool import StaticPool

            # Set up in-memory database for testing
            SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
            engine = create_engine(
                SQLALCHEMY_DATABASE_URL, 
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
            TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            Base.metadata.create_all(bind=engine)

            def override_get_db():
                try:
                    db = TestingSessionLocal()
                    yield db
                finally:
                    db.close()

            app.dependency_overrides[get_db] = override_get_db
            
            set_model_manager(mock_model_manager)

            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                yield client


@pytest.fixture
def sample_image_bytes():
    """Create a sample PNG image in-memory."""
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Auth Tests
# ---------------------------------------------------------------------------

class TestAuth:
    """Tests for authentication routes."""

    _registered_user = {
        "email": "testuser@example.com",
        "username": "testuser",
        "password": "SecurePass123!",
        "full_name": "Test User",
    }

    def test_register(self, test_client):
        """Register a new user."""
        resp = test_client.post("/api/auth/register", json=self._registered_user)
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == self._registered_user["email"]
        assert data["username"] == self._registered_user["username"]
        assert "id" in data

    def test_register_duplicate_email(self, test_client):
        """Registering the same email again should fail."""
        resp = test_client.post("/api/auth/register", json=self._registered_user)
        assert resp.status_code in (400, 409, 422)

    def test_login_json(self, test_client):
        """Login via JSON body."""
        resp = test_client.post(
            "/api/auth/login/json",
            json={"email": self._registered_user["email"], "password": self._registered_user["password"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, test_client):
        """Login with wrong password should fail."""
        resp = test_client.post(
            "/api/auth/login/json",
            json={"email": self._registered_user["email"], "password": "WrongPassword!"},
        )
        assert resp.status_code == 401

    def test_get_me(self, test_client):
        """Get current user profile."""
        # Login first
        login = test_client.post(
            "/api/auth/login/json",
            json={"email": self._registered_user["email"], "password": self._registered_user["password"]},
        )
        token = login.json()["access_token"]

        resp = test_client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == self._registered_user["email"]
        assert data["username"] == self._registered_user["username"]

    def test_get_me_no_auth(self, test_client):
        """Accessing /me without auth should fail."""
        test_client.cookies.clear()
        resp = test_client.get("/api/auth/me")
        assert resp.status_code in (401, 403)

    def test_update_profile(self, test_client):
        """Update user profile."""
        login = test_client.post(
            "/api/auth/login/json",
            json={"email": self._registered_user["email"], "password": self._registered_user["password"]},
        )
        token = login.json()["access_token"]

        resp = test_client.patch(
            "/api/auth/me",
            json={"full_name": "Updated Name"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["full_name"] == "Updated Name"


# ---------------------------------------------------------------------------
# Prediction Tests (Authenticated)
# ---------------------------------------------------------------------------

class TestPredictions:
    """Tests for prediction endpoints with authentication."""

    @pytest.fixture(autouse=True)
    def _auth(self, test_client):
        """Login and store token for all tests in this class."""
        # Register user if not already registered
        test_client.post(
            "/api/auth/register",
            json={
                "email": "predictor@example.com",
                "username": "predictor",
                "password": "PredictPass123!",
                "full_name": "Predict User",
            },
        )
        login = test_client.post(
            "/api/auth/login/json",
            json={"email": "predictor@example.com", "password": "PredictPass123!"},
        )
        self.token = login.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_predict_single(self, test_client, sample_image_bytes):
        """Predict on a single image."""
        resp = test_client.post(
            "/api/predict/single?model=cnn",
            files={"file": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_class"] == "genuine"
        assert data["confidence"] == pytest.approx(0.95)
        assert data["risk_level"] == "LOW"
        assert "id" in data

    def test_predict_batch(self, test_client, sample_image_bytes):
        """Predict on a batch of images."""
        files = [
            ("files", (f"test_{i}.png", io.BytesIO(sample_image_bytes), "image/png"))
            for i in range(3)
        ]
        resp = test_client.post(
            "/api/predict/batch?model=cnn",
            files=files,
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        for pred in data:
            assert pred["predicted_class"] == "genuine"

    def test_predict_no_auth(self, test_client, sample_image_bytes):
        """Prediction without auth should still work (optional auth)."""
        resp = test_client.post(
            "/api/predict/single?model=cnn",
            files={"file": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )
        # Should work — predict allows unauthenticated (get_optional_user)
        assert resp.status_code == 200

    def test_predict_invalid_file(self, test_client):
        """Uploading a non-image should fail."""
        resp = test_client.post(
            "/api/predict/single?model=cnn",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
            headers=self.headers,
        )
        assert resp.status_code == 400

    def test_prediction_history(self, test_client, sample_image_bytes):
        """Get prediction history after making predictions."""
        # Make a prediction first
        test_client.post(
            "/api/predict/single?model=cnn",
            files={"file": ("history_test.png", io.BytesIO(sample_image_bytes), "image/png")},
            headers=self.headers,
        )

        resp = test_client.get("/api/predict/history", headers=self.headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert "total" in data
        assert data["total"] >= 1

    def test_prediction_history_pagination(self, test_client):
        """Test history pagination parameters."""
        resp = test_client.get(
            "/api/predict/history?page=1&page_size=5", headers=self.headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 1
        assert data["page_size"] == 5

    def test_prediction_stats(self, test_client):
        """Get prediction statistics."""
        resp = test_client.get("/api/predict/stats", headers=self.headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_predictions" in data
        assert "fraud_detected" in data
        assert "genuine_documents" in data
        assert "by_class" in data
        assert "by_risk_level" in data
        assert "by_model" in data
        assert "avg_confidence" in data

    def test_delete_prediction(self, test_client, sample_image_bytes):
        """Create and delete a prediction."""
        # Create
        create_resp = test_client.post(
            "/api/predict/single?model=cnn",
            files={"file": ("delete_me.png", io.BytesIO(sample_image_bytes), "image/png")},
            headers=self.headers,
        )
        pred_id = create_resp.json()["id"]

        # Delete
        del_resp = test_client.delete(f"/api/predict/{pred_id}", headers=self.headers)
        assert del_resp.status_code == 200

        # Verify gone
        get_resp = test_client.get(f"/api/predict/{pred_id}", headers=self.headers)
        assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# General Endpoint Tests
# ---------------------------------------------------------------------------

class TestGeneralEndpoints:
    """Tests for root, health, and model listing endpoints."""

    def test_root(self, test_client):
        """Root endpoint returns welcome message."""
        resp = test_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data

    def test_health(self, test_client):
        """Health check returns healthy status."""
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_models_list(self, test_client):
        """Model listing returns available models."""
        resp = test_client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_docs_available(self, test_client):
        """Swagger docs should be accessible."""
        resp = test_client.get("/docs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# API Key Tests
# ---------------------------------------------------------------------------

class TestAPIKeys:
    """Tests for API key management."""

    @pytest.fixture(autouse=True)
    def _auth(self, test_client):
        test_client.post(
            "/api/auth/register",
            json={
                "email": "apikey@example.com",
                "username": "apikeyuser",
                "password": "ApiKeyPass123!",
            },
        )
        login = test_client.post(
            "/api/auth/login/json",
            json={"email": "apikey@example.com", "password": "ApiKeyPass123!"},
        )
        self.token = login.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_create_api_key(self, test_client):
        """Create a new API key."""
        resp = test_client.post(
            "/api/auth/api-keys",
            json={"name": "test-key"},
            headers=self.headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-key"
        assert data["is_active"] is True

    def test_list_api_keys(self, test_client):
        """List existing API keys."""
        resp = test_client.get("/api/auth/api-keys", headers=self.headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_delete_api_key(self, test_client):
        """Create and delete an API key."""
        # Create
        create_resp = test_client.post(
            "/api/auth/api-keys",
            json={"name": "del-key"},
            headers=self.headers,
        )
        key_id = create_resp.json()["id"]

        # Delete
        del_resp = test_client.delete(f"/api/auth/api-keys/{key_id}", headers=self.headers)
        assert del_resp.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
