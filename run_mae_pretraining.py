"""
MAE Pretraining on RVL-CDIP Document Images
============================================
Self-supervised pretraining using Masked Autoencoder (MAE) on document images.

This script trains an MAE model on the RVL-CDIP dataset to learn document
structure representations, which can then be used to initialize the ViT
for better fraud detection performance.

Usage:
    python run_mae_pretraining.py
    python run_mae_pretraining.py --epochs 30
    python run_mae_pretraining.py --quick
"""

import sys
import os
import json
import time
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    _reconfigure = 'reconfigure'
    if hasattr(sys.stdout, _reconfigure):
        getattr(sys.stdout, _reconfigure)(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from utils.seed import set_seed, get_device
from ssl_pretraining.mae_model import MaskedAutoencoder


# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 8
DATA_DIR = Path("data/rvl_cdip")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


class RVLCDIPDataset(Dataset):
    """Dataset for RVL-CDIP document images (unlabeled for MAE)."""
    
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Recursively find all images
        if self.root_dir.exists():
            for ext in self.EXTENSIONS:
                self.samples.extend(self.root_dir.rglob(f"*{ext}"))
            
            # Also check for uppercase extensions
            for ext in [e.upper() for e in self.EXTENSIONS]:
                self.samples.extend(self.root_dir.rglob(f"*{ext}"))
        
        self.samples = [str(p) for p in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            # Return a blank image on error
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


def get_transforms():
    """Get data transforms for MAE pretraining."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_mae(epochs: int = 20, lr: float = 1e-4, quick: bool = False):
    """Main MAE pretraining function."""
    
    print("=" * 60)
    print("  MAE PRETRAINING ON RVL-CDIP")
    print("=" * 60)
    
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # Check data
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        return None
    
    # Create dataset
    dataset = RVLCDIPDataset(str(DATA_DIR), transform=get_transforms())
    print(f"Total images: {len(dataset)}")
    
    if len(dataset) < 10:
        print("Error: Not enough images to pretrain")
        return None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create MAE model (smaller for faster training)
    model = MaskedAutoencoder(
        image_size=IMAGE_SIZE,
        patch_size=16,
        in_channels=3,
        encoder_dim=384,      # Smaller than default 768
        encoder_heads=6,      # Smaller than default 12
        encoder_depth=6,      # Smaller than default 12
        decoder_dim=256,      # Smaller than default 512
        decoder_heads=8,      # Smaller than default 16
        decoder_depth=4,      # Smaller than default 8
        mask_ratio=0.75       # Standard 75% masking
    )
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    if quick:
        epochs = min(epochs, 5)
    
    # Warmup + cosine decay schedule
    warmup_epochs = min(3, epochs // 4)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        decay_epochs = epochs - warmup_epochs
        progress = (epoch - warmup_epochs) / decay_epochs
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nTraining for {epochs} epochs (warmup: {warmup_epochs})...")
    print("-" * 60)
    
    history = {"loss": [], "lr": []}
    best_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            
            optimizer.zero_grad()
            loss = model(images)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        
        history["loss"].append(avg_loss)
        history["lr"].append(current_lr)
        
        print(f"Epoch [{epoch:2d}/{epochs}] Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, CHECKPOINT_DIR / "mae_pretrained.pth")
            print(f"    * Saved best model (loss: {avg_loss:.4f})")
        
        # Also save encoder separately for easy fine-tuning
        encoder_state = model.encoder.state_dict()
        torch.save({
            "encoder_state_dict": encoder_state,
            "image_size": IMAGE_SIZE,
            "patch_size": 16,
            "embed_dim": 384,
        }, CHECKPOINT_DIR / "mae_encoder.pth")
    
    # Save final results
    print("\n" + "=" * 60)
    print("  PRETRAINING COMPLETE")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "model": "MAE",
        "epochs": epochs,
        "best_loss": best_loss,
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "num_images": len(dataset),
        "image_size": IMAGE_SIZE,
        "mask_ratio": 0.75,
        "history": history,
    }
    
    with open(RESULTS_DIR / "mae_pretraining_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults:")
    print(f"  Images trained: {len(dataset)}")
    print(f"  Epochs:         {epochs}")
    print(f"  Best loss:      {best_loss:.4f}")
    print(f"\nFiles saved:")
    print(f"  Model: {CHECKPOINT_DIR / 'mae_pretrained.pth'}")
    print(f"  Encoder: {CHECKPOINT_DIR / 'mae_encoder.pth'}")
    print(f"  Results: {RESULTS_DIR / 'mae_pretraining_results.json'}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true", help="Quick training with fewer epochs")
    args = parser.parse_args()
    
    train_mae(epochs=args.epochs, lr=args.lr, quick=args.quick)
