"""
Model Registry for Version Control and Management
==================================================
Track and manage model versions, metrics, and artifacts.

Features:
- Model versioning
- Metrics tracking
- Model comparison
- Artifact management
- Deployment history

Usage:
    from deployment.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    registry.register_model('cnn', 'checkpoints/cnn_best.pth', metrics={...})
    registry.list_models()
    registry.get_best_model('accuracy')
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

import torch


@dataclass
class ModelVersion:
    """Represents a single model version."""
    name: str
    version: str
    architecture: str
    checkpoint_path: str
    checkpoint_hash: str
    created_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    status: str = "registered"  # registered, staging, production, archived
    deployed_at: Optional[str] = None


class ModelRegistry:
    """
    Centralized model registry for tracking versions and deployments.
    """
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "registry.json"
        self.models: Dict[str, List[ModelVersion]] = {}
        self.production_models: Dict[str, str] = {}  # model_name -> version
        
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                
            # Reconstruct ModelVersion objects
            for name, versions in data.get('models', {}).items():
                self.models[name] = [
                    ModelVersion(**v) for v in versions
                ]
            self.production_models = data.get('production_models', {})
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        data = {
            'models': {
                name: [asdict(v) for v in versions]
                for name, versions in self.models.items()
            },
            'production_models': self.production_models,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    
    def _get_next_version(self, name: str) -> str:
        """Get next version number for a model."""
        if name not in self.models or not self.models[name]:
            return "1.0.0"
        
        latest = self.models[name][-1].version
        parts = latest.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        return '.'.join(parts)
    
    def register_model(
        self,
        name: str,
        checkpoint_path: str,
        architecture: str = "unknown",
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            name: Model name (e.g., 'cnn', 'vit', 'hybrid')
            checkpoint_path: Path to model checkpoint
            architecture: Model architecture description
            metrics: Evaluation metrics (accuracy, f1, etc.)
            hyperparameters: Training hyperparameters
            description: Model description
            tags: Tags for categorization
            version: Specific version (auto-generated if None)
        
        Returns:
            ModelVersion object
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Auto-generate version if not provided
        if version is None:
            version = self._get_next_version(name)
        
        # Create model version
        model_version = ModelVersion(
            name=name,
            version=version,
            architecture=architecture,
            checkpoint_path=str(ckpt_path.absolute()),
            checkpoint_hash=self._compute_hash(str(ckpt_path)),
            created_at=datetime.now().isoformat(),
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            description=description,
            tags=tags or []
        )
        
        # Add to registry
        if name not in self.models:
            self.models[name] = []
        self.models[name].append(model_version)
        
        # Copy checkpoint to registry (for versioning)
        version_dir = self.registry_dir / name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = version_dir / ckpt_path.name
        shutil.copy2(ckpt_path, dest_path)
        
        # Save metadata
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(asdict(model_version), f, indent=2)
        
        self._save_registry()
        print(f"Registered model: {name} v{version}")
        
        return model_version
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models and their versions."""
        return {
            name: [v.version for v in versions]
            for name, versions in self.models.items()
        }
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get a specific model version.
        
        Args:
            name: Model name
            version: Version string (latest if None)
        
        Returns:
            ModelVersion or None
        """
        if name not in self.models or not self.models[name]:
            return None
        
        if version is None:
            # Return latest
            return self.models[name][-1]
        
        # Find specific version
        for v in self.models[name]:
            if v.version == version:
                return v
        
        return None
    
    def get_best_model(self, name: str, metric: str = "accuracy") -> Optional[ModelVersion]:
        """
        Get the best performing model version by a metric.
        
        Args:
            name: Model name
            metric: Metric to compare (higher is better)
        
        Returns:
            Best ModelVersion or None
        """
        if name not in self.models or not self.models[name]:
            return None
        
        best = None
        best_score = -float('inf')
        
        for v in self.models[name]:
            score = v.metrics.get(metric, -float('inf'))
            if score > best_score:
                best_score = score
                best = v
        
        return best
    
    def promote_to_production(self, name: str, version: str) -> bool:
        """
        Promote a model version to production status.
        
        Args:
            name: Model name
            version: Version to promote
        
        Returns:
            True if successful
        """
        model = self.get_model(name, version)
        if model is None:
            print(f"Model {name} v{version} not found")
            return False
        
        # Demote current production model
        if name in self.production_models:
            old_version = self.production_models[name]
            for v in self.models[name]:
                if v.version == old_version:
                    v.status = "archived"
                    break
        
        # Promote new model
        model.status = "production"
        model.deployed_at = datetime.now().isoformat()
        self.production_models[name] = version
        
        self._save_registry()
        print(f"Promoted {name} v{version} to production")
        
        return True
    
    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """Get the current production model for a given name."""
        if name not in self.production_models:
            return None
        
        return self.get_model(name, self.production_models[name])
    
    def compare_models(self, name: str, versions: Optional[List[str]] = None) -> Dict:
        """
        Compare metrics across model versions.
        
        Args:
            name: Model name
            versions: Specific versions to compare (all if None)
        
        Returns:
            Comparison dictionary
        """
        if name not in self.models:
            return {}
        
        models_to_compare = self.models[name]
        if versions:
            models_to_compare = [
                v for v in models_to_compare 
                if v.version in versions
            ]
        
        comparison = {
            'versions': [],
            'metrics': {}
        }
        
        # Collect all metric names
        all_metrics = set()
        for v in models_to_compare:
            all_metrics.update(v.metrics.keys())
        
        # Build comparison
        for v in models_to_compare:
            comparison['versions'].append(v.version)
            for metric in all_metrics:
                if metric not in comparison['metrics']:
                    comparison['metrics'][metric] = []
                comparison['metrics'][metric].append(v.metrics.get(metric, None))
        
        return comparison
    
    def delete_model(self, name: str, version: str) -> bool:
        """Delete a model version (except production)."""
        if name not in self.models:
            return False
        
        # Prevent deleting production model
        if self.production_models.get(name) == version:
            print("Cannot delete production model. Promote another version first.")
            return False
        
        # Find and remove
        self.models[name] = [
            v for v in self.models[name] 
            if v.version != version
        ]
        
        # Remove files
        version_dir = self.registry_dir / name / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        self._save_registry()
        print(f"Deleted {name} v{version}")
        
        return True
    
    def export_registry(self, output_path: str) -> None:
        """Export registry to a JSON file."""
        data = {
            'models': {
                name: [asdict(v) for v in versions]
                for name, versions in self.models.items()
            },
            'production_models': self.production_models,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported registry to {output_path}")
    
    def load_model_weights(self, name: str, version: Optional[str] = None) -> Dict:
        """Load model weights from registry."""
        model = self.get_model(name, version)
        if model is None:
            raise ValueError(f"Model {name} not found")
        
        # Try registry path first, then original path
        registry_path = self.registry_dir / name / model.version / Path(model.checkpoint_path).name
        
        if registry_path.exists():
            return torch.load(registry_path, map_location='cpu')
        elif Path(model.checkpoint_path).exists():
            return torch.load(model.checkpoint_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Checkpoint not found for {name} v{model.version}")
    
    def summary(self) -> str:
        """Get a text summary of the registry."""
        lines = ["=" * 60, "MODEL REGISTRY SUMMARY", "=" * 60]
        
        for name, versions in self.models.items():
            prod_version = self.production_models.get(name, "none")
            lines.append(f"\n{name.upper()}")
            lines.append("-" * 40)
            
            for v in versions:
                status_mark = "🚀" if v.version == prod_version else "  "
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in list(v.metrics.items())[:3]
                ) if v.metrics else "no metrics"
                
                lines.append(f"  {status_mark} v{v.version}: {metrics_str}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def register_existing_models():
    """Register existing trained models from checkpoints."""
    registry = ModelRegistry()
    project_root = Path(__file__).resolve().parent.parent
    checkpoints_dir = project_root / "checkpoints"
    results_dir = project_root / "results"
    
    # Load test metrics if available
    test_metrics = {}
    metrics_files = {
        'cnn': results_dir / "test_metrics.json",
        'vit': results_dir / "test_metrics_vit.json",
        'hybrid': results_dir / "full_comparison.json"
    }
    
    for name, path in metrics_files.items():
        if path.exists():
            with open(path) as f:
                test_metrics[name] = json.load(f)
    
    # Register models
    model_info = {
        'cnn': {
            'checkpoint': 'cnn_best.pth',
            'architecture': 'ResNet50 + Custom Head',
            'description': 'CNN baseline using pretrained ResNet50'
        },
        'vit': {
            'checkpoint': 'vit_best.pth',
            'architecture': 'ViT-Small (4 layers, 4 heads, 128 dim)',
            'description': 'Vision Transformer trained from scratch'
        },
        'hybrid': {
            'checkpoint': 'hybrid_best.pth',
            'architecture': 'ResNet50 + ViT Encoder',
            'description': 'Hybrid CNN-ViT combining both architectures'
        },
        'vit_ssl': {
            'checkpoint': 'vit_ssl_best.pth',
            'architecture': 'ViT with MAE pretraining',
            'description': 'ViT with self-supervised MAE pretraining'
        }
    }
    
    for name, info in model_info.items():
        ckpt_path = checkpoints_dir / info['checkpoint']
        if ckpt_path.exists():
            metrics = test_metrics.get(name, {})
            # Extract relevant metrics
            if isinstance(metrics, dict):
                filtered_metrics: Dict[str, float] = {
                    k: float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float)) and k in 
                    ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
                }
            else:
                filtered_metrics = {}
            
            registry.register_model(
                name=name,
                checkpoint_path=str(ckpt_path),
                architecture=info['architecture'],
                metrics=filtered_metrics,
                description=info['description'],
                tags=['fraud-detection', 'banking']
            )
    
    print(registry.summary())
    return registry


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Registry')
    parser.add_argument('--register', action='store_true', help='Register existing models')
    parser.add_argument('--list', action='store_true', help='List all models')
    parser.add_argument('--summary', action='store_true', help='Show registry summary')
    
    args = parser.parse_args()
    
    if args.register:
        register_existing_models()
    elif args.list:
        registry = ModelRegistry()
        print(json.dumps(registry.list_models(), indent=2))
    elif args.summary:
        registry = ModelRegistry()
        print(registry.summary())
    else:
        parser.print_help()
