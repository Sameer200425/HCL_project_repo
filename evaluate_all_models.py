"""
Evaluate All Models and Update Comparison CSV
==============================================
Loads all trained model checkpoints and generates fresh comparison metrics.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.seed import set_seed, get_device
from utils.dataset import create_dataloaders
from models.vit_model import VisionTransformer
from models.hybrid_model import HybridCNNViT, CNNBaseline

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]

CLASSES = ["genuine", "fraud", "tampered", "forged"]
IMAGE_SIZE = 224
DATA_DIR = "data/raw_images"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

VIT_CONFIG = {
    "patch_size": 16, "embed_dim": 128, "num_heads": 4,
    "num_layers": 4, "mlp_dim": 256, "dropout": 0.1,
}

HYBRID_CONFIG = {
    "embed_dim": 128, "num_heads": 4, "num_layers": 2, "dropout": 0.1,
}


def get_model_size(model):
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds, all_labels = [], []
    
    inference_times = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            start = time.time()
            outputs = model(images)
            inference_times.append((time.time() - start) * 1000)  # ms
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    avg_inf_time = np.mean(inference_times)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'inf_time_ms': avg_inf_time,
    }


def load_and_evaluate():
    """Load all models and evaluate them."""
    print("=" * 60)
    print("  EVALUATING ALL MODELS")
    print("=" * 60)
    
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # Create dataloaders with the same class ordering as training
    loaders = create_dataloaders(
        DATA_DIR, IMAGE_SIZE, batch_size=16, num_workers=0,
        class_names=CLASSES  # Use consistent class order
    )
    test_loader = loaders['test']
    print(f"Test samples: {len(test_loader.dataset)}")  # type: ignore[arg-type]
    
    results = []
    
    # 1. CNN (ResNet50)
    print("\n[1/4] Evaluating CNN (ResNet50)...")
    try:
        model = CNNBaseline(num_classes=len(CLASSES))
        cp = torch.load(f'{CHECKPOINT_DIR}/cnn_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(cp['model_state_dict'])
        model = model.to(device)
        
        metrics = evaluate_model(model, test_loader, device)
        results.append({
            'Model': 'CNN (ResNet50)',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Inf. Time (ms)': metrics['inf_time_ms'],
            'Size (MB)': get_model_size(model),
            'Params': f"{count_parameters(model):,}",
        })
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. ViT (from scratch)
    print("\n[2/4] Evaluating ViT (from scratch)...")
    try:
        model = VisionTransformer(
            image_size=IMAGE_SIZE, in_channels=3, num_classes=len(CLASSES),
            **VIT_CONFIG
        )
        cp = torch.load(f'{CHECKPOINT_DIR}/vit_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(cp['model_state_dict'])
        model = model.to(device)
        
        metrics = evaluate_model(model, test_loader, device)
        results.append({
            'Model': 'ViT (from scratch)',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Inf. Time (ms)': metrics['inf_time_ms'],
            'Size (MB)': get_model_size(model),
            'Params': f"{count_parameters(model):,}",
        })
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. ViT + SSL (MAE pretrained)
    print("\n[3/4] Evaluating ViT + SSL (MAE)...")
    try:
        model = VisionTransformer(
            image_size=IMAGE_SIZE, in_channels=3, num_classes=len(CLASSES),
            **VIT_CONFIG
        )
        cp = torch.load(f'{CHECKPOINT_DIR}/vit_ssl_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(cp['model_state_dict'])
        model = model.to(device)
        
        metrics = evaluate_model(model, test_loader, device)
        results.append({
            'Model': 'ViT + SSL (MAE)',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Inf. Time (ms)': metrics['inf_time_ms'],
            'Size (MB)': get_model_size(model),
            'Params': f"{count_parameters(model):,}",
        })
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Hybrid CNN+ViT
    print("\n[4/4] Evaluating Hybrid CNN+ViT...")
    try:
        model = HybridCNNViT(num_classes=len(CLASSES), **HYBRID_CONFIG)
        cp = torch.load(f'{CHECKPOINT_DIR}/hybrid_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(cp['model_state_dict'])
        model = model.to(device)
        
        metrics = evaluate_model(model, test_loader, device)
        results.append({
            'Model': 'Hybrid CNN+ViT',
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'Inf. Time (ms)': metrics['inf_time_ms'],
            'Size (MB)': get_model_size(model),
            'Params': f"{count_parameters(model):,}",
        })
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        csv_path = f'{RESULTS_DIR}/model_comparison.csv'
        df.to_csv(csv_path, index=False)
        
        print("\n" + "=" * 60)
        print("  COMPARISON RESULTS")
        print("=" * 60)
        print(f"\n{'Model':<20} {'Accuracy':>10} {'F1-Score':>10} {'Size (MB)':>10}")
        print("-" * 52)
        for _, row in df.iterrows():
            print(f"{row['Model']:<20} {row['Accuracy']:>10.4f} {row['F1-Score']:>10.4f} {row['Size (MB)']:>10.1f}")
        
        # Best model
        best_idx = int(df['Accuracy'].idxmax())
        best = df.iloc[best_idx]
        print(f"\nBEST: {best['Model']} with {best['Accuracy']:.2%} accuracy")
        print(f"\nResults saved to: {csv_path}")
    
    return results


if __name__ == "__main__":
    load_and_evaluate()
