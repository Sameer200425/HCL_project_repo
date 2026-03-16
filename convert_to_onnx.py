"""
Convert PyTorch Models to ONNX for Real-Time Inference
========================================================
Optimizes the trained models for faster execution using ONNX Runtime.

Usage:
    python convert_to_onnx.py
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vit_model import VisionTransformer
from models.hybrid_model import HybridCNNViT

def convert_vit():
    """Convert the best ViT model to ONNX."""
    print("🔄 Converting ViT model to ONNX...")
    
    # 1. Instantiate the model with the SAME config used in training
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        mlp_dim=256,
        dropout=0.0
    )
    
    # 2. Load weights
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "vit_best.pth"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle both full checkpoint dict and raw state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    try:
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ ViT Weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 3. Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 4. Export to ONNX
    output_path = PROJECT_ROOT / "checkpoints" / "vit_best.onnx"
    
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"🚀 ViT Model saved to: {output_path}")
        
        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validity check passed")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

def convert_hybrid():
    """Convert the Hybrid CNN+ViT model to ONNX."""
    print("\n🔄 Converting Hybrid model to ONNX...")
    
    # 1. Instantiate the model
    # Match training config: embed_dim=128, num_heads=4, num_layers=2
    try:
        model = HybridCNNViT(
            num_classes=4,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )
    except Exception as e:
        print(f"⚠️ Could not instantiate Hybrid model: {e}")
        return

    # 2. Load weights
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "hybrid_best.pth"
    if not checkpoint_path.exists():
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    try:
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("✅ Hybrid Weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 3. Export
    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = PROJECT_ROOT / "checkpoints" / "hybrid_best.onnx"
    
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"🚀 Hybrid Model saved to: {output_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    convert_vit()
    convert_hybrid()
