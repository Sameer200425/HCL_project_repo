"""
Signature Verification Model Training
======================================
Trains a ViT model specifically for signature verification (genuine vs forged).

Uses the CEDAR signature dataset (or synthetic samples for testing).

Usage:
    python train_signature_model.py
    python train_signature_model.py --epochs 20
    python train_signature_model.py --quick
"""

import sys
import os
import time
import json
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    _stdout = sys.stdout
    _method = 'reconfigure'
    if hasattr(_stdout, _method):
        getattr(_stdout, _method)(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.seed import set_seed, get_device
from models.vit_model import VisionTransformer


# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 2
CLASSES = ["genuine", "forged"]
DATA_DIR = Path("data/cedar_signatures")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


class SignatureDataset(Dataset):
    """Dataset for signature images (genuine/forged)."""
    
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Load genuine signatures
        genuine_dir = self.root_dir / "genuine"
        if genuine_dir.exists():
            for f in genuine_dir.iterdir():
                if f.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(f), 0))  # 0 = genuine
        
        # Load forged signatures
        forged_dir = self.root_dir / "forged"
        if forged_dir.exists():
            for f in forged_dir.iterdir():
                if f.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(f), 1))  # 1 = forged
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(is_train: bool = True):
    """Get data transforms."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_signature_model(epochs: int = 15, lr: float = 1e-4, quick: bool = False):
    """Train signature verification model."""
    
    print("=" * 60)
    print("  SIGNATURE VERIFICATION MODEL TRAINING")
    print("=" * 60)
    
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # Check data
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        return None
    
    # Create dataset
    dataset = SignatureDataset(str(DATA_DIR), transform=get_transforms(is_train=True))
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) < 10:
        print("Error: Not enough samples to train")
        return None
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update transforms for val/test
    val_set.dataset.transform = get_transforms(is_train=False)  # type: ignore[union-attr]
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # Create model
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=16,
        in_channels=3,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        mlp_dim=256,
        dropout=0.1
    )
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    if quick:
        epochs = min(epochs, 5)
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch:2d}/{epochs}] "
              f"Train: {train_loss/len(train_loader):.4f}/{train_acc:.4f} | "
              f"Val: {val_loss/len(val_loader):.4f}/{val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, CHECKPOINT_DIR / "signature_vit_best.pth")
            print(f"    * Saved best model (val_acc: {val_acc:.4f})")
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  Genuine predicted as Genuine: {cm[0][0]}")
    print(f"  Genuine predicted as Forged:  {cm[0][1]}")
    print(f"  Forged predicted as Genuine:  {cm[1][0]}")
    print(f"  Forged predicted as Forged:   {cm[1][1]}")
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "model": "Signature ViT",
        "epochs": epochs,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "best_val_acc": best_val_acc,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    
    with open(RESULTS_DIR / "signature_model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR / 'signature_model_results.json'}")
    print(f"Model saved to: {CHECKPOINT_DIR / 'signature_vit_best.pth'}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    train_signature_model(epochs=args.epochs, lr=args.lr, quick=args.quick)
