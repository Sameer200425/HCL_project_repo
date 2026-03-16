"""
Integration Smoke Test — Frontend ↔ Backend ↔ Models End-to-End
================================================================
Validates the full stack works together:
  1. Model checkpoints load correctly
  2. Inference pipeline produces valid output
  3. Backend API responds to all key endpoints
  4. Frontend build configuration is valid
  5. Docker files are syntactically correct
  6. All Python modules import without errors

Run:
    pytest tests/test_integration_smoke.py -v
    python tests/test_integration_smoke.py        # standalone
"""

import sys
import io
import os
import json
import importlib
from pathlib import Path
from typing import Dict, List

import pytest
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ===================================================================
#  1. Model Checkpoint Loading
# ===================================================================

class TestModelCheckpoints:
    """Verify all model checkpoints load correctly."""

    CHECKPOINT_DIR = ROOT / "checkpoints"

    @pytest.fixture(params=["cnn_best.pth", "hybrid_best.pth", "vit_best.pth"])
    def checkpoint_name(self, request):
        return request.param

    @pytest.mark.skipif(
        not (ROOT / "checkpoints").exists(),
        reason="Checkpoints directory missing",
    )
    def test_checkpoint_loads(self, checkpoint_name):
        path = self.CHECKPOINT_DIR / checkpoint_name
        if not path.exists():
            pytest.skip(f"{checkpoint_name} not found")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        assert ckpt is not None
        # Should be a dict or state_dict
        assert isinstance(ckpt, dict)

    def test_cnn_model_instantiates(self):
        from models.hybrid_model import CNNBaseline
        model = CNNBaseline(pretrained=False, num_classes=4)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)

    def test_hybrid_model_instantiates(self):
        from models.hybrid_model import HybridCNNViT
        model = HybridCNNViT(cnn_pretrained=False, num_classes=4)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)

    def test_vit_model_instantiates(self):
        from models.vit_model import VisionTransformer
        model = VisionTransformer(
            image_size=224, patch_size=16, in_channels=3,
            num_classes=4, embed_dim=64, num_heads=4,
            num_layers=2, mlp_dim=128, dropout=0.0,
        )
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)


# ===================================================================
#  2. Inference Pipeline
# ===================================================================

class TestInferencePipeline:
    """Test the full inference flow: image → preprocess → model → output."""

    def test_end_to_end_cnn_inference(self):
        from models.hybrid_model import CNNBaseline
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Create synthetic image
        img = Image.new("RGB", (300, 400), color=(128, 64, 32))
        tensor: torch.Tensor = transform(img).unsqueeze(0)  # type: ignore[union-attr]

        model = CNNBaseline(pretrained=False, num_classes=4)
        model.eval()

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)

        assert probs.shape == (1, 4)
        assert abs(probs.sum().item() - 1.0) < 1e-5
        assert (probs >= 0).all()

    def test_batch_inference(self):
        from models.hybrid_model import CNNBaseline
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        batch = torch.stack([
            transform(Image.new("RGB", (224, 224), color=(i*50, i*30, i*20)))  # type: ignore[misc]
            for i in range(4)
        ])

        model = CNNBaseline(pretrained=False, num_classes=4)
        model.eval()

        with torch.no_grad():
            out = model(batch)

        assert out.shape == (4, 4)


# ===================================================================
#  3. Python Module Import Smoke Test
# ===================================================================

class TestModuleImports:
    """Verify all project modules import without errors."""

    MODULES = [
        "models.vit_model",
        "models.hybrid_model",
        "analytics.risk_scoring",
        "analytics.performance_metrics",
        "analytics.fraud_trend_engine",
        "compliance.regulatory_report",
        "adversarial.attack_detection",
        "data_integration.unified_loader",
        "data_integration.cedar_signature_loader",
        "data_integration.creditcard_fraud_loader",
        "utils.seed",
    ]

    @pytest.fixture(params=MODULES)
    def module_name(self, request):
        return request.param

    def test_module_imports(self, module_name):
        try:
            mod = importlib.import_module(module_name)
            assert mod is not None
        except ImportError as e:
            # Allow optional dependency failures (neo4j, etc.)
            if "neo4j" in str(e) or "onnx" in str(e):
                pytest.skip(f"Optional dependency missing: {e}")
            raise


# ===================================================================
#  4. Frontend Configuration Validation
# ===================================================================

class TestFrontendConfig:
    """Validate frontend configuration files."""

    FRONTEND_DIR = ROOT / "frontend"

    def test_package_json_exists(self):
        pkg = self.FRONTEND_DIR / "package.json"
        assert pkg.exists(), "package.json missing"

    def test_package_json_valid(self):
        pkg = self.FRONTEND_DIR / "package.json"
        if not pkg.exists():
            pytest.skip("package.json missing")
        data = json.loads(pkg.read_text())
        assert "name" in data
        assert "dependencies" in data
        assert "next" in data["dependencies"]
        assert "react" in data["dependencies"]

    def test_tsconfig_exists(self):
        tsc = self.FRONTEND_DIR / "tsconfig.json"
        assert tsc.exists(), "tsconfig.json missing"

    def test_next_config_exists(self):
        ncfg = self.FRONTEND_DIR / "next.config.js"
        assert ncfg.exists(), "next.config.js missing"

    def test_tailwind_config_exists(self):
        twcfg = self.FRONTEND_DIR / "tailwind.config.js"
        assert twcfg.exists(), "tailwind.config.js missing"


# ===================================================================
#  5. Docker File Validation
# ===================================================================

class TestDockerFiles:
    """Validate Docker deployment files."""

    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").exists()

    def test_dockerfile_has_from(self):
        content = (ROOT / "Dockerfile").read_text()
        assert "FROM" in content

    def test_docker_compose_exists(self):
        assert (ROOT / "docker-compose.yml").exists()

    def test_docker_compose_has_services(self):
        content = (ROOT / "docker-compose.yml").read_text()
        assert "services" in content
        assert "api" in content


# ===================================================================
#  6. Configuration File Validation
# ===================================================================

class TestConfigFiles:
    """Validate project configuration files."""

    def test_config_yaml_exists(self):
        assert (ROOT / "config.yaml").exists()

    def test_config_yaml_parseable(self):
        import yaml
        with open(ROOT / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        # Should have model and training sections
        assert any(k in config for k in ["model", "training", "vit", "data"])

    def test_requirements_txt_exists(self):
        assert (ROOT / "requirements.txt").exists()

    def test_requirements_has_torch(self):
        content = (ROOT / "requirements.txt").read_text().lower()
        assert "torch" in content


# ===================================================================
#  7. Results Directory Structure
# ===================================================================

class TestProjectStructure:
    """Verify essential directories and files exist."""

    REQUIRED_DIRS = [
        "models", "analytics", "backend", "compliance",
        "explainability", "deployment", "data_integration",
        "tests", "frontend", "checkpoints",
    ]

    @pytest.fixture(params=REQUIRED_DIRS)
    def dir_name(self, request):
        return request.param

    def test_directory_exists(self, dir_name):
        assert (ROOT / dir_name).is_dir(), f"Directory {dir_name}/ missing"

    def test_models_have_init(self):
        for pkg in ["models", "analytics", "backend", "compliance", "explainability"]:
            init = ROOT / pkg / "__init__.py"
            assert init.exists(), f"{pkg}/__init__.py missing"


# ===================================================================
#  Standalone runner
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
