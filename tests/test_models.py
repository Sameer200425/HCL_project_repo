"""
Unit Tests for Models
=====================
Tests for ViT, CNN, and Hybrid models.

Run tests:
    pytest tests/test_models.py -v
    pytest tests/test_models.py -v -k "test_vit"
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vit_model import VisionTransformer
from models.hybrid_model import CNNBaseline, HybridCNNViT


class TestVisionTransformer:
    """Tests for Vision Transformer model."""
    
    @pytest.fixture
    def vit_model(self):
        """Create a small ViT model for testing."""
        return VisionTransformer(
            image_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=4,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            mlp_dim=128,
            dropout=0.0
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)
    
    def test_model_creation(self, vit_model):
        """Test model can be created."""
        assert vit_model is not None
        assert isinstance(vit_model, nn.Module)
    
    def test_forward_pass(self, vit_model, sample_input):
        """Test forward pass produces correct output shape."""
        vit_model.eval()
        with torch.no_grad():
            output = vit_model(sample_input)
        
        assert output.shape == (2, 4)  # (batch_size, num_classes)
    
    def test_output_range(self, vit_model, sample_input):
        """Test output values are reasonable (for logits)."""
        vit_model.eval()
        with torch.no_grad():
            output = vit_model(sample_input)
        
        # Logits should not be extreme
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_batch_sizes(self, vit_model):
        """Test model works with different batch sizes."""
        vit_model.eval()
        
        for batch_size in [1, 4, 8, 16]:
            input_tensor = torch.randn(batch_size, 3, 224, 224)
            with torch.no_grad():
                output = vit_model(input_tensor)
            assert output.shape == (batch_size, 4)
    
    def test_gradient_flow(self, vit_model, sample_input):
        """Test gradients flow properly during training."""
        vit_model.train()
        output = vit_model(sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = False
        for param in vit_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in model"
    
    def test_parameter_count(self, vit_model):
        """Test model has expected number of parameters."""
        total_params = sum(p.numel() for p in vit_model.parameters())
        assert total_params > 0
        # Small model should have reasonable param count
        assert total_params < 10_000_000  # Less than 10M


class TestCNNBaseline:
    """Tests for CNN Baseline model."""
    
    @pytest.fixture
    def cnn_model(self):
        """Create CNN model for testing."""
        return CNNBaseline(num_classes=4)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)
    
    def test_model_creation(self, cnn_model):
        """Test model can be created."""
        assert cnn_model is not None
        assert isinstance(cnn_model, nn.Module)
    
    def test_forward_pass(self, cnn_model, sample_input):
        """Test forward pass produces correct output shape."""
        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(sample_input)
        
        assert output.shape == (2, 4)
    
    def test_output_validity(self, cnn_model, sample_input):
        """Test output values are valid."""
        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(sample_input)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestHybridModel:
    """Tests for Hybrid CNN+ViT model."""
    
    @pytest.fixture
    def hybrid_model(self):
        """Create Hybrid model for testing."""
        return HybridCNNViT(num_classes=4)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)
    
    def test_model_creation(self, hybrid_model):
        """Test model can be created."""
        assert hybrid_model is not None
        assert isinstance(hybrid_model, nn.Module)
    
    def test_forward_pass(self, hybrid_model, sample_input):
        """Test forward pass produces correct output shape."""
        hybrid_model.eval()
        with torch.no_grad():
            output = hybrid_model(sample_input)
        
        assert output.shape == (2, 4)
    
    def test_hybrid_larger_than_vit(self, hybrid_model):
        """Test hybrid model has more parameters than standalone ViT."""
        vit = VisionTransformer(
            image_size=224, patch_size=16, in_channels=3,
            num_classes=4, embed_dim=64, num_heads=4,
            num_layers=2, mlp_dim=128, dropout=0.0
        )
        
        hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
        vit_params = sum(p.numel() for p in vit.parameters())
        
        # Hybrid should be larger due to CNN backbone
        assert hybrid_params > vit_params


class TestModelTraining:
    """Tests for model training functionality."""
    
    def test_model_can_overfit_small_batch(self):
        """Test model can overfit on a small batch (sanity check)."""
        model = VisionTransformer(
            image_size=224, patch_size=16, in_channels=3,
            num_classes=4, embed_dim=64, num_heads=4,
            num_layers=2, mlp_dim=128, dropout=0.0
        )
        
        # Small batch
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 2, 3])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        initial_loss: float = 0.0
        loss: torch.Tensor = torch.tensor(0.0)
        
        for i in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if i == 0:
                initial_loss = loss.item()
        
        final_loss = loss.item()
        
        # Loss should decrease significantly
        assert initial_loss > 0, "Initial loss not recorded"
        assert final_loss < initial_loss * 0.5, "Model failed to overfit small batch"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
