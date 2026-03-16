"""
K-Fold Cross-Validation & Ensemble Weighted Voting
====================================================
Runs 5-fold stratified cross-validation for CNN, Hybrid, and ViT models,
then implements ensemble weighted voting (CNN + ViT + Hybrid).

Usage:
    python kfold_ensemble.py                  # Run all
    python kfold_ensemble.py --kfold-only     # K-fold only
    python kfold_ensemble.py --ensemble-only  # Ensemble only
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.vit_model import VisionTransformer
from models.hybrid_model import CNNBaseline, HybridCNNViT
from utils.seed import set_seed

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

SEED = 42
DATA_DIR = Path("data/raw_images")
RESULTS_DIR = Path("results")
CHECKPOINTS_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 4
CLASS_NAMES = ["genuine", "fraud", "tampered", "forged"]

# ------------------------------------------------------------------ #
#  Dataset
# ------------------------------------------------------------------ #

class ImageFolderDataset(Dataset):
    """Custom dataset for image folder structure."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ------------------------------------------------------------------ #
#  Model Factories
# ------------------------------------------------------------------ #

def create_cnn_model() -> nn.Module:
    """Create CNN (ResNet50) model."""
    return CNNBaseline(pretrained=True, num_classes=NUM_CLASSES, dropout=0.3)


def create_hybrid_model() -> nn.Module:
    """Create Hybrid CNN+ViT model (must match checkpoint dims)."""
    return HybridCNNViT(
        cnn_pretrained=True,
        cnn_features=2048,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=NUM_CLASSES,
    )


def create_vit_model() -> nn.Module:
    """Create ViT model from scratch (must match checkpoint dims)."""
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=NUM_CLASSES,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        mlp_dim=256,
        dropout=0.1,
    )


MODEL_REGISTRY = {
    "cnn": {"factory": create_cnn_model, "lr": 1e-4, "epochs": 3},
    "hybrid": {"factory": create_hybrid_model, "lr": 1e-4, "epochs": 3},
    "vit": {"factory": create_vit_model, "lr": 3e-4, "epochs": 5},
}

# ------------------------------------------------------------------ #
#  Training Utilities
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ------------------------------------------------------------------ #
#  K-Fold Cross-Validation
# ------------------------------------------------------------------ #

def run_kfold_cv(
    model_name: str,
    n_folds: int = 5,
) -> Dict:
    """Run stratified K-Fold CV for a given model architecture."""
    config = MODEL_REGISTRY[model_name]
    factory = config["factory"]
    lr = config["lr"]
    epochs = config["epochs"]

    print(f"\n{'='*70}")
    print(f"  K-FOLD CROSS-VALIDATION: {model_name.upper()} ({n_folds} folds, {epochs} epochs)")
    print(f"{'='*70}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(str(DATA_DIR), transform=transform)
    labels = np.array([s[1] for s in dataset.samples])
    print(f"  Dataset: {len(dataset)} samples, {len(set(labels))} classes")
    for cls_name, cls_idx in dataset.class_to_idx.items():
        count = (labels == cls_idx).sum()
        print(f"    {cls_name}: {count} samples")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), labels)):
        print(f"\n  --- Fold {fold + 1}/{n_folds} ---")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False, num_workers=0)

        model = factory().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0.0
        best_f1 = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            scheduler.step()

            preds, true_labels, probs = evaluate(model, val_loader, DEVICE)
            val_acc = accuracy_score(true_labels, preds)
            val_f1 = f1_score(true_labels, preds, average="weighted", zero_division=0)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_f1 = val_f1

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1:2d}/{epochs} - Train: {train_acc:.4f} | Val: {val_acc:.4f} | F1: {val_f1:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "best_val_acc": float(best_val_acc),
            "best_f1": float(best_f1),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
        })
        print(f"  Fold {fold + 1}: Acc={best_val_acc:.4f}, F1={best_f1:.4f}")

    # Summary
    accs = [r["best_val_acc"] for r in fold_results]
    f1s = [r["best_f1"] for r in fold_results]
    summary = {
        "model": model_name,
        "n_folds": n_folds,
        "epochs_per_fold": epochs,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "fold_results": fold_results,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n  {'='*50}")
    print(f"  {model_name.upper()} K-FOLD: Acc={np.mean(accs):.4f} ± {np.std(accs):.4f} | F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  {'='*50}")

    return summary


# ------------------------------------------------------------------ #
#  Ensemble Weighted Voting
# ------------------------------------------------------------------ #

def run_ensemble_evaluation() -> Dict:
    """
    Evaluate ensemble of CNN + Hybrid + ViT using weighted voting.
    Loads saved checkpoints and combines predictions.
    """
    print(f"\n{'='*70}")
    print("  ENSEMBLE WEIGHTED VOTING (CNN + Hybrid + ViT)")
    print(f"{'='*70}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(str(DATA_DIR), transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    labels = np.array([s[1] for s in dataset.samples])

    # Define models and their weights (based on individual accuracy)
    model_configs = [
        {
            "name": "cnn",
            "checkpoint": CHECKPOINTS_DIR / "cnn_best.pth",
            "factory": create_cnn_model,
            "weight": 0.45,  # Highest weight - best individual accuracy
        },
        {
            "name": "hybrid",
            "checkpoint": CHECKPOINTS_DIR / "hybrid_best.pth",
            "factory": create_hybrid_model,
            "weight": 0.40,  # Second highest
        },
        {
            "name": "vit",
            "checkpoint": CHECKPOINTS_DIR / "vit_best.pth",
            "factory": create_vit_model,
            "weight": 0.15,  # Lowest - limited data performance
        },
    ]

    # Collect predictions from each model
    all_model_probs = []
    individual_results = {}

    for cfg in model_configs:
        print(f"\n  Loading {cfg['name'].upper()}...")
        model = cfg["factory"]().to(DEVICE)

        ckpt_path = cfg["checkpoint"]
        if ckpt_path.exists():
            checkpoint = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"    Loaded checkpoint: {ckpt_path}")
        else:
            print(f"    ⚠ No checkpoint found at {ckpt_path}, using random weights")

        preds, true_labels, probs = evaluate(model, loader, DEVICE)
        all_model_probs.append((probs, cfg["weight"]))

        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average="weighted", zero_division=0)
        individual_results[cfg["name"]] = {
            "accuracy": float(acc),
            "f1": float(f1),
            "weight": cfg["weight"],
        }
        print(f"    {cfg['name'].upper()}: Acc={acc:.4f}, F1={f1:.4f}")

    # Weighted ensemble
    print(f"\n  Computing weighted ensemble...")
    total_weight = sum(w for _, w in all_model_probs)
    ensemble_probs = np.zeros_like(all_model_probs[0][0])
    for probs, weight in all_model_probs:
        ensemble_probs += probs * (weight / total_weight)

    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_acc = accuracy_score(labels, ensemble_preds)
    ensemble_f1 = f1_score(labels, ensemble_preds, average="weighted", zero_division=0)
    ensemble_prec = precision_score(labels, ensemble_preds, average="weighted", zero_division=0)
    ensemble_rec = recall_score(labels, ensemble_preds, average="weighted", zero_division=0)

    # Per-class metrics
    report: dict = classification_report(labels, ensemble_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)  # type: ignore[assignment]
    cm = confusion_matrix(labels, ensemble_preds)

    print(f"\n  {'='*50}")
    print(f"  ENSEMBLE RESULTS")
    print(f"  {'='*50}")
    print(f"  Accuracy:  {ensemble_acc:.4f}")
    print(f"  F1-Score:  {ensemble_f1:.4f}")
    print(f"  Precision: {ensemble_prec:.4f}")
    print(f"  Recall:    {ensemble_rec:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(labels, ensemble_preds, target_names=CLASS_NAMES, zero_division=0))
    print(f"\n  Confusion Matrix:")
    print(cm)

    # Majority voting comparison
    print(f"\n  --- Majority Voting (unweighted) ---")
    all_preds_list = []
    for probs, _ in all_model_probs:
        all_preds_list.append(probs.argmax(axis=1))
    majority_preds = np.array(all_preds_list)
    # Use mode (most common prediction per sample)
    from scipy import stats as scipy_stats
    majority_result = scipy_stats.mode(majority_preds, axis=0, keepdims=False)
    majority_final = majority_result.mode
    majority_acc = accuracy_score(labels, majority_final)
    majority_f1 = f1_score(labels, majority_final, average="weighted", zero_division=0)
    print(f"  Majority Voting: Acc={majority_acc:.4f}, F1={majority_f1:.4f}")

    results = {
        "ensemble_method": "weighted_voting",
        "weights": {cfg["name"]: cfg["weight"] for cfg in model_configs},
        "ensemble_accuracy": float(ensemble_acc),
        "ensemble_f1": float(ensemble_f1),
        "ensemble_precision": float(ensemble_prec),
        "ensemble_recall": float(ensemble_rec),
        "majority_voting_accuracy": float(majority_acc),
        "majority_voting_f1": float(majority_f1),
        "individual_models": individual_results,
        "per_class_report": {k: v for k, v in report.items() if k in CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
        "timestamp": datetime.now().isoformat(),
    }

    return results


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="K-Fold CV & Ensemble Evaluation")
    parser.add_argument("--kfold-only", action="store_true", help="Run only K-fold CV")
    parser.add_argument("--ensemble-only", action="store_true", help="Run only ensemble evaluation")
    parser.add_argument("--models", nargs="+", default=["cnn", "hybrid", "vit"], help="Models for K-fold")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  K-Fold Cross-Validation & Ensemble Evaluation")
    print(f"  Device: {DEVICE}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # K-fold CV
    if not args.ensemble_only:
        kfold_results = {}
        for model_name in args.models:
            if model_name in MODEL_REGISTRY:
                result = run_kfold_cv(model_name, n_folds=args.folds)
                kfold_results[model_name] = result
            else:
                print(f"  ⚠ Unknown model: {model_name}")

        # Save K-fold results
        output_path = RESULTS_DIR / "kfold_all_models.json"
        with open(output_path, "w") as f:
            json.dump(kfold_results, f, indent=2)
        print(f"\n✅ K-fold results saved to: {output_path}")

        # Print comparison table
        print(f"\n{'='*70}")
        print("  K-FOLD COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<15} {'Mean Acc':>10} {'Std Acc':>10} {'Mean F1':>10} {'Std F1':>10}")
        print(f"  {'-'*55}")
        for name, res in kfold_results.items():
            print(f"  {name:<15} {res['mean_accuracy']:>10.4f} {res['std_accuracy']:>10.4f} {res['mean_f1']:>10.4f} {res['std_f1']:>10.4f}")

    # Ensemble evaluation
    if not args.kfold_only:
        ensemble_results = run_ensemble_evaluation()

        output_path = RESULTS_DIR / "ensemble_results.json"
        with open(output_path, "w") as f:
            json.dump(ensemble_results, f, indent=2)
        print(f"\n✅ Ensemble results saved to: {output_path}")

    print(f"\n{'='*70}")
    print("  ALL TASKS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
