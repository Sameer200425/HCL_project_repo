"""
Unit Tests for Deployment Utilities
====================================
Tests for ONNX export and model registry.

Run tests:
    pytest tests/test_deployment.py -v
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import cast

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestONNXExporter:
    """Tests for ONNX export functionality."""
    
    def test_export_config_creation(self):
        """Test ExportConfig can be created."""
        from deployment.onnx_export import ExportConfig
        
        config = ExportConfig(
            model_name='vit',
            input_shape=(1, 3, 224, 224),
            opset_version=14
        )
        
        assert config.model_name == 'vit'
        assert config.input_shape == (1, 3, 224, 224)
    
    def test_exporter_creation(self):
        """Test ONNXExporter can be created."""
        from deployment.onnx_export import ONNXExporter, ExportConfig
        
        config = ExportConfig(model_name='vit')
        exporter = ONNXExporter(config)
        
        assert exporter is not None
        assert exporter.config.model_name == 'vit'
    
    @pytest.mark.skipif(
        not Path("checkpoints/cnn_best.pth").exists(),
        reason="Model checkpoint not available"
    )
    def test_model_loading(self):
        """Test model can be loaded for export."""
        from deployment.onnx_export import ONNXExporter, ExportConfig
        
        config = ExportConfig(model_name='cnn', validate=False)
        exporter = ONNXExporter(config)
        
        model = exporter.load_model()
        assert model is not None
    
    @pytest.mark.skipif(
        not Path("checkpoints/cnn_best.pth").exists(),
        reason="Model checkpoint not available"
    )
    def test_onnx_export(self):
        """Test ONNX export functionality."""
        from deployment.onnx_export import ONNXExporter, ExportConfig
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            output_path = f.name
        
        try:
            config = ExportConfig(
                model_name='cnn',
                optimize=False,
                simplify=False,
                validate=False
            )
            exporter = ONNXExporter(config)
            
            path = exporter.export_to_onnx(output_path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestModelRegistry:
    """Tests for model registry functionality."""
    
    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Create a temporary registry."""
        from deployment.model_registry import ModelRegistry
        return ModelRegistry(registry_dir=str(tmp_path / "test_registry"))
    
    @pytest.fixture
    def sample_checkpoint(self, tmp_path):
        """Create a sample checkpoint file."""
        ckpt_path = tmp_path / "sample_model.pth"
        torch.save({
            'model_state_dict': {},
            'epoch': 10,
            'loss': 0.1
        }, ckpt_path)
        return str(ckpt_path)
    
    def test_registry_creation(self, temp_registry):
        """Test registry can be created."""
        assert temp_registry is not None
        assert temp_registry.registry_dir.exists()
    
    def test_register_model(self, temp_registry, sample_checkpoint):
        """Test model registration."""
        version = temp_registry.register_model(
            name='test_model',
            checkpoint_path=sample_checkpoint,
            architecture='TestNet',
            metrics={'accuracy': 0.95, 'f1': 0.93},
            description='Test model for unit testing'
        )
        
        assert version is not None
        assert version.name == 'test_model'
        assert version.metrics['accuracy'] == 0.95
    
    def test_list_models(self, temp_registry, sample_checkpoint):
        """Test listing registered models."""
        temp_registry.register_model(
            name='model_a',
            checkpoint_path=sample_checkpoint
        )
        temp_registry.register_model(
            name='model_b',
            checkpoint_path=sample_checkpoint
        )
        
        models = temp_registry.list_models()
        assert 'model_a' in models
        assert 'model_b' in models
    
    def test_get_model(self, temp_registry, sample_checkpoint):
        """Test getting a specific model."""
        temp_registry.register_model(
            name='my_model',
            checkpoint_path=sample_checkpoint,
            metrics={'accuracy': 0.9}
        )
        
        model = temp_registry.get_model('my_model')
        assert model is not None
        assert model.name == 'my_model'
    
    def test_version_auto_increment(self, temp_registry, sample_checkpoint):
        """Test version auto-increments."""
        v1 = temp_registry.register_model('seq_model', sample_checkpoint)
        v2 = temp_registry.register_model('seq_model', sample_checkpoint)
        v3 = temp_registry.register_model('seq_model', sample_checkpoint)
        
        assert v1.version == '1.0.0'
        assert v2.version == '1.0.1'
        assert v3.version == '1.0.2'
    
    def test_get_best_model(self, temp_registry, sample_checkpoint):
        """Test getting best model by metric."""
        temp_registry.register_model(
            'compare_model', sample_checkpoint,
            metrics={'accuracy': 0.85}
        )
        temp_registry.register_model(
            'compare_model', sample_checkpoint,
            metrics={'accuracy': 0.92}
        )
        temp_registry.register_model(
            'compare_model', sample_checkpoint,
            metrics={'accuracy': 0.88}
        )
        
        best = temp_registry.get_best_model('compare_model', 'accuracy')
        assert best.metrics['accuracy'] == 0.92
    
    def test_promote_to_production(self, temp_registry, sample_checkpoint):
        """Test promoting model to production."""
        v = temp_registry.register_model('prod_model', sample_checkpoint)
        
        success = temp_registry.promote_to_production('prod_model', v.version)
        assert success
        
        prod = temp_registry.get_production_model('prod_model')
        assert prod is not None
        assert prod.status == 'production'
    
    def test_compare_models(self, temp_registry, sample_checkpoint):
        """Test model comparison."""
        temp_registry.register_model(
            'cmp_model', sample_checkpoint,
            metrics={'accuracy': 0.9, 'loss': 0.1}
        )
        temp_registry.register_model(
            'cmp_model', sample_checkpoint,
            metrics={'accuracy': 0.95, 'loss': 0.05}
        )
        
        comparison = temp_registry.compare_models('cmp_model')
        assert 'versions' in comparison
        assert 'metrics' in comparison
        assert len(comparison['versions']) == 2
    
    def test_delete_model(self, temp_registry, sample_checkpoint):
        """Test deleting a model version."""
        temp_registry.register_model('del_model', sample_checkpoint)
        v2 = temp_registry.register_model('del_model', sample_checkpoint)
        
        success = temp_registry.delete_model('del_model', '1.0.0')
        assert success
        
        models = temp_registry.list_models()
        assert len(models['del_model']) == 1
        assert models['del_model'][0] == '1.0.1'
    
    def test_export_registry(self, temp_registry, sample_checkpoint, tmp_path):
        """Test exporting registry to JSON."""
        temp_registry.register_model('export_model', sample_checkpoint)
        
        export_path = tmp_path / 'registry_export.json'
        temp_registry.export_registry(str(export_path))
        
        assert export_path.exists()
        
        with open(export_path) as f:
            data = json.load(f)
        
        assert 'models' in data
        assert 'export_model' in data['models']


class TestDataIntegrity:
    """Tests for data handling and preprocessing."""
    
    def test_image_normalization_values(self):
        """Test that normalization uses correct ImageNet values."""
        from torchvision import transforms
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Create a tensor of ones
        tensor = torch.ones(1, 3, 224, 224)
        normalized = normalize(tensor)
        
        # Check normalization was applied
        assert not torch.allclose(tensor, normalized)
    
    def test_image_resize(self):
        """Test image resizing maintains aspect for center crop."""
        from torchvision import transforms
        from PIL import Image
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        # Create a non-square image
        img = Image.new('RGB', (300, 200))
        tensor = cast(torch.Tensor, transform(img))
        
        assert tensor.shape == (3, 224, 224)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
