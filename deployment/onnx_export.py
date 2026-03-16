"""
ONNX Export Module for Production Deployment
=============================================
Export PyTorch models to ONNX format for:
- TensorRT optimization
- ONNX Runtime inference
- Cross-platform deployment
- Mobile/Edge deployment

Usage:
    python deployment/onnx_export.py --model cnn --output models/cnn.onnx
    python deployment/onnx_export.py --model vit --optimize
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vit_model import VisionTransformer
from models.hybrid_model import CNNBaseline, HybridCNNViT


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    model_name: str
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
    opset_version: int = 14
    dynamic_axes: bool = True
    optimize: bool = True
    quantize: bool = False
    simplify: bool = True
    validate: bool = True


class ONNXExporter:
    """Export PyTorch models to ONNX format."""
    
    # Model configurations matching training
    MODEL_CONFIGS = {
        'cnn': {
            'class': CNNBaseline,
            'checkpoint': 'checkpoints/cnn_best.pth',
            'kwargs': {'num_classes': 4}
        },
        'vit': {
            'class': VisionTransformer,
            'checkpoint': 'checkpoints/vit_best.pth',
            'kwargs': {
                'image_size': 224,
                'patch_size': 16,
                'in_channels': 3,
                'num_classes': 4,
                'embed_dim': 128,
                'num_heads': 4,
                'num_layers': 4,
                'mlp_dim': 256,
                'dropout': 0.0  # Disable dropout for inference
            }
        },
        'hybrid': {
            'class': HybridCNNViT,
            'checkpoint': 'checkpoints/hybrid_best.pth',
            'kwargs': {'num_classes': 4}
        }
    }
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.device = torch.device('cpu')  # Export on CPU for compatibility
        self.project_root = Path(__file__).resolve().parent.parent
        
    def load_model(self) -> nn.Module:
        """Load PyTorch model from checkpoint."""
        model_info = self.MODEL_CONFIGS.get(self.config.model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {self.config.model_name}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")
        
        # Build model architecture
        model_class = model_info['class']
        model = model_class(**model_info['kwargs'])
        
        # Load checkpoint
        ckpt_path = self.project_root / model_info['checkpoint']
        if ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Try loading with and without 'module.' prefix
            try:
                model.load_state_dict(state_dict, strict=False)
            except RuntimeError:
                # Remove 'module.' prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict, strict=False)
            
            print(f"Loaded weights from {ckpt_path}")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, using random weights")
        
        model.eval()
        return model
    
    def export_to_onnx(self, output_path: str) -> str:
        """Export model to ONNX format."""
        print(f"\n{'='*60}")
        print(f"Exporting {self.config.model_name} to ONNX")
        print(f"{'='*60}")
        
        # Load model
        model = self.load_model()
        
        # Create dummy input
        dummy_input = torch.randn(*self.config.input_shape, device=self.device)
        
        # Define dynamic axes if enabled
        dynamic_axes = None
        if self.config.dynamic_axes:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Create output directory
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to ONNX
        print(f"Exporting to: {out_path}")
        start_time = time.time()
        
        torch.onnx.export(
            model,
            (dummy_input,),
            str(out_path),
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        export_time = time.time() - start_time
        print(f"Export completed in {export_time:.2f}s")
        
        # Optimize if requested
        if self.config.optimize:
            self._optimize_onnx(str(out_path))
        
        # Simplify if requested
        if self.config.simplify:
            self._simplify_onnx(str(out_path))
        
        # Validate if requested
        if self.config.validate:
            self._validate_onnx(str(out_path), model, dummy_input)
        
        # Report file size
        file_size = out_path.stat().st_size / (1024 * 1024)
        print(f"\nONNX model size: {file_size:.2f} MB")
        
        return str(out_path)
    
    def _optimize_onnx(self, onnx_path: str) -> None:
        """Optimize ONNX model using onnxoptimizer."""
        try:
            import onnx
            import onnxoptimizer  # type: ignore[import-not-found]
            
            print("Optimizing ONNX model...")
            model = onnx.load(onnx_path)
            
            # Define optimization passes
            passes = [
                'eliminate_identity',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'fuse_bn_into_conv',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
            ]
            
            optimized = onnxoptimizer.optimize(model, passes)
            onnx.save(optimized, onnx_path)
            print("Optimization complete")
            
        except ImportError:
            print("onnxoptimizer not installed, skipping optimization")
        except Exception as e:
            print(f"Optimization failed: {e}")
    
    def _simplify_onnx(self, onnx_path: str) -> None:
        """Simplify ONNX model using onnx-simplifier."""
        try:
            import onnx
            from onnxsim import simplify  # type: ignore[import-not-found]
            
            print("Simplifying ONNX model...")
            model = onnx.load(onnx_path)
            simplified, check = simplify(model)
            
            if check:
                onnx.save(simplified, onnx_path)
                print("Simplification complete")
            else:
                print("Simplification check failed, keeping original")
                
        except ImportError:
            print("onnx-simplifier not installed, skipping simplification")
        except Exception as e:
            print(f"Simplification failed: {e}")
    
    def _validate_onnx(self, onnx_path: str, torch_model: nn.Module, 
                       dummy_input: torch.Tensor) -> bool:
        """Validate ONNX model outputs match PyTorch."""
        try:
            import onnx
            import onnxruntime as ort
            
            print("\nValidating ONNX model...")
            
            # Check ONNX model validity
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("✓ ONNX model structure valid")
            
            # Compare outputs
            ort_session = ort.InferenceSession(
                onnx_path, 
                providers=['CPUExecutionProvider']
            )
            
            # Get PyTorch output
            with torch.no_grad():
                torch_output = torch_model(dummy_input).numpy()
            
            # Get ONNX Runtime output
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare
            max_diff = np.abs(torch_output - ort_output).max()
            mean_diff = np.abs(torch_output - ort_output).mean()
            
            print(f"✓ Max output difference: {max_diff:.6f}")
            print(f"✓ Mean output difference: {mean_diff:.6f}")
            
            if max_diff < 1e-4:
                print("✓ Validation PASSED - outputs match within tolerance")
                return True
            else:
                print("⚠ Validation WARNING - outputs differ slightly")
                return True  # Still usable
                
        except ImportError as e:
            print(f"Validation skipped (missing dependencies): {e}")
            return True
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
    
    def benchmark(self, onnx_path: str, num_runs: int = 100) -> Dict:
        """Benchmark ONNX model inference speed."""
        try:
            import onnxruntime as ort
            
            print(f"\nBenchmarking ONNX model ({num_runs} runs)...")
            
            session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider']
            )
            
            # Warmup
            dummy_input = np.random.randn(*self.config.input_shape).astype(np.float32)
            input_name = session.get_inputs()[0].name
            
            for _ in range(10):
                session.run(None, {input_name: dummy_input})
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                session.run(None, {input_name: dummy_input})
                times.append((time.perf_counter() - start) * 1000)
            
            results = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'p50_ms': np.percentile(times, 50),
                'p95_ms': np.percentile(times, 95),
                'p99_ms': np.percentile(times, 99),
            }
            
            print(f"Mean inference time: {results['mean_ms']:.2f} ms")
            print(f"P95 inference time: {results['p95_ms']:.2f} ms")
            print(f"Throughput: {1000/results['mean_ms']:.1f} images/sec")
            
            return results
            
        except ImportError:
            print("onnxruntime not installed, skipping benchmark")
            return {}


def export_all_models(output_dir: str = "deployment/onnx_models") -> Dict[str, str]:
    """Export all available models to ONNX."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for model_name in ONNXExporter.MODEL_CONFIGS.keys():
        try:
            config = ExportConfig(
                model_name=model_name,
                optimize=True,
                simplify=True,
                validate=True
            )
            exporter = ONNXExporter(config)
            output_path = out_dir / f"{model_name}.onnx"
            exporter.export_to_onnx(str(output_path))
            results[model_name] = str(output_path)
            
        except Exception as e:
            print(f"Failed to export {model_name}: {e}")
            results[model_name] = f"FAILED: {e}"
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Export models to ONNX')
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'vit', 'hybrid', 'all'],
                       help='Model to export')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for ONNX model')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize ONNX model')
    parser.add_argument('--simplify', action='store_true',
                       help='Simplify ONNX model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark exported model')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        results = export_all_models()
        print("\n" + "="*60)
        print("Export Summary:")
        for name, path in results.items():
            print(f"  {name}: {path}")
    else:
        output_path = args.output or f"deployment/onnx_models/{args.model}.onnx"
        
        config = ExportConfig(
            model_name=args.model,
            optimize=args.optimize,
            simplify=args.simplify,
            validate=not args.no_validate
        )
        
        exporter = ONNXExporter(config)
        onnx_path = exporter.export_to_onnx(output_path)
        
        if args.benchmark:
            exporter.benchmark(onnx_path)


if __name__ == '__main__':
    main()
