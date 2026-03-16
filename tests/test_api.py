"""
Unit Tests for API
==================
Tests for FastAPI endpoints.

Run tests:
    pytest tests/test_api.py -v
"""

import sys
import io
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes."""
        img = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.read()
    
    @pytest.fixture
    def sample_image_file(self, sample_image_bytes):
        """Create a file-like object for upload testing."""
        return io.BytesIO(sample_image_bytes)
    
    def test_image_creation(self, sample_image_bytes):
        """Test sample image can be created and read."""
        img = Image.open(io.BytesIO(sample_image_bytes))
        assert img.size == (224, 224)
        assert img.mode == 'RGB'
    
    @pytest.mark.skipif(
        not Path("checkpoints/cnn_best.pth").exists(),
        reason="Model checkpoint not available"
    )
    def test_model_manager_creation(self):
        """Test ModelManager can be instantiated."""
        from deployment.fastapi_server import ModelManager, ModelType
        
        manager = ModelManager()
        assert manager is not None
        assert manager.device is not None
    
    @pytest.mark.skipif(
        not Path("checkpoints/cnn_best.pth").exists(),
        reason="Model checkpoint not available"
    )
    def test_model_loading(self):
        """Test models can be loaded."""
        from deployment.fastapi_server import ModelManager, ModelType
        
        manager = ModelManager()
        # Should not raise
        model = manager.get_model(ModelType.CNN)
        assert model is not None
    
    @pytest.mark.skipif(
        not Path("checkpoints/cnn_best.pth").exists(),
        reason="Model checkpoint not available"
    )
    def test_single_prediction(self, sample_image_bytes):
        """Test single image prediction."""
        from deployment.fastapi_server import ModelManager, ModelType
        
        manager = ModelManager()
        image = Image.open(io.BytesIO(sample_image_bytes))
        
        result = manager.predict(image, ModelType.CNN)
        
        assert 'class_name' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'risk_level' in result
        assert result['class_name'] in ['genuine', 'fraud', 'tampered', 'forged']
        assert 0 <= result['confidence'] <= 1
    
    @pytest.mark.skipif(
        not Path("checkpoints/cnn_best.pth").exists(),
        reason="Model checkpoint not available"
    )
    def test_batch_prediction(self, sample_image_bytes):
        """Test batch image prediction."""
        from deployment.fastapi_server import ModelManager, ModelType
        
        manager = ModelManager()
        images = [
            Image.open(io.BytesIO(sample_image_bytes))
            for _ in range(3)
        ]
        
        results = manager.predict_batch(images, ModelType.CNN)
        
        assert len(results) == 3
        for result in results:
            assert 'class_name' in result
            assert 'confidence' in result
    
    def test_risk_classification(self):
        """Test risk level classification."""
        from deployment.fastapi_server import ModelManager
        
        manager = ModelManager()
        
        # High confidence genuine should be LOW risk
        assert manager.classify_risk('genuine', 0.95) == 'LOW'
        
        # High confidence fraud should be CRITICAL
        assert manager.classify_risk('fraud', 0.9) == 'CRITICAL'
        
        # Low confidence fraud should be lower risk
        assert manager.classify_risk('fraud', 0.2) == 'LOW'


class TestFastAPIApp:
    """Integration tests for FastAPI app."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client."""
        try:
            from fastapi.testclient import TestClient
            from deployment.fastapi_server import app
            
            with TestClient(app) as client:
                yield client
        except ImportError:
            pytest.skip("FastAPI test dependencies not available")
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns welcome message."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    def test_health_endpoint(self, test_client):
        """Test health endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_models_endpoint(self, test_client):
        """Test models listing endpoint."""
        response = test_client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
