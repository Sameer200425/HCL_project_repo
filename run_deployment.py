"""
Deployment Demo Script
======================
Demonstrates all new deployment features:
1. Model Registry - Register and track model versions
2. ONNX Export - Export models for production
3. FastAPI Server - Launch production API

Usage:
    python run_deployment.py --all
    python run_deployment.py --registry
    python run_deployment.py --export
    python run_deployment.py --api
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def demo_model_registry():
    """Demonstrate model registry features."""
    print("\n" + "="*60)
    print("MODEL REGISTRY DEMO")
    print("="*60)
    
    from deployment.model_registry import ModelRegistry, register_existing_models
    
    # Register existing models from checkpoints
    print("\nRegistering existing models...")
    registry = register_existing_models()
    
    # Show summary
    print("\nRegistry Summary:")
    print(registry.summary())
    
    # Get best model
    print("\nFinding best CNN model by accuracy...")
    best = registry.get_best_model('cnn', 'accuracy')
    if best:
        print(f"  Best version: {best.version}")
        print(f"  Metrics: {best.metrics}")
    
    # List all models
    print("\nAll registered models:")
    for name, versions in registry.list_models().items():
        print(f"  {name}: {versions}")
    
    return registry


def demo_onnx_export():
    """Demonstrate ONNX export."""
    print("\n" + "="*60)
    print("ONNX EXPORT DEMO")
    print("="*60)
    
    from deployment.onnx_export import ONNXExporter, ExportConfig, export_all_models
    
    ckpt_dir = Path("checkpoints")
    
    # Check if any checkpoints exist
    if not ckpt_dir.exists() or not list(ckpt_dir.glob("*.pth")):
        print("No checkpoints found. Skipping ONNX export.")
        print("Train models first using: python run_all_tasks.py")
        return
    
    # Export all models
    print("\nExporting all models to ONNX format...")
    results = export_all_models()
    
    print("\nExport Results:")
    for model_name, path in results.items():
        print(f"  {model_name}: {path}")
    
    # Benchmark one model
    if 'cnn' in results and not results['cnn'].startswith('FAILED'):
        print("\nBenchmarking CNN ONNX model...")
        config = ExportConfig(model_name='cnn')
        exporter = ONNXExporter(config)
        exporter.benchmark(results['cnn'], num_runs=50)


def demo_fastapi():
    """Start FastAPI server."""
    print("\n" + "="*60)
    print("FASTAPI SERVER DEMO")
    print("="*60)
    
    print("""
FastAPI server provides:
  - /docs        - Interactive API documentation (Swagger UI)
  - /redoc       - Alternative API documentation
  - /health      - Health check endpoint
  - /predict     - Single image prediction
  - /predict/batch - Batch prediction (up to 32 images)
  - /models      - List available models

Starting server on http://localhost:8000 ...
Press Ctrl+C to stop.
""")
    
    import uvicorn
    uvicorn.run(
        "deployment.fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


def run_tests():
    """Run unit tests."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])


def main():
    parser = argparse.ArgumentParser(description='Deployment Demo')
    parser.add_argument('--all', action='store_true', 
                       help='Run all demos (except API server)')
    parser.add_argument('--registry', action='store_true',
                       help='Demo model registry')
    parser.add_argument('--export', action='store_true',
                       help='Demo ONNX export')
    parser.add_argument('--api', action='store_true',
                       help='Start FastAPI server')
    parser.add_argument('--test', action='store_true',
                       help='Run unit tests')
    
    args = parser.parse_args()
    
    if not any([args.all, args.registry, args.export, args.api, args.test]):
        parser.print_help()
        print("\nQuick start:")
        print("  python run_deployment.py --all    # Run all demos")
        print("  python run_deployment.py --api    # Start API server")
        return
    
    if args.all:
        demo_model_registry()
        demo_onnx_export()
        run_tests()
        print("\n" + "="*60)
        print("All demos complete!")
        print("To start the API server: python run_deployment.py --api")
        print("="*60)
    else:
        if args.registry:
            demo_model_registry()
        if args.export:
            demo_onnx_export()
        if args.test:
            run_tests()
        if args.api:
            demo_fastapi()


if __name__ == '__main__':
    main()
