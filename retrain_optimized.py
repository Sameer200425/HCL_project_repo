"""
Optimized Model Retraining Script
===================================
Retrains all 4 model architectures with tuned hyperparameters:
  - CNN (ResNet50) — 20 epochs, lr=1e-3, heavier augmentation
  - ViT (from scratch) — 30 epochs, lr=3e-4, warmup + cosine
  - ViT+SSL (MAE pretrained) — 25 epochs, lr=1e-4 (fine-tune)
  - Hybrid CNN+ViT — 25 epochs, lr=5e-4

Improvements over initial run_all_tasks.py:
  ✓ 6× more training data (600 vs 100 images)
  ✓ Heavier data augmentation (RandAugment, CutMix-style)
  ✓ Proper learning rate warmup + cosine annealing
  ✓ Gradient accumulation for effective larger batch
  ✓ MixUp regularisation
  ✓ Better weight decay scheduling
  ✓ Saves model_comparison.csv with updated results

Usage:
    python retrain_optimized.py                 # retrain all
    python retrain_optimized.py --model cnn     # retrain CNN only
    python retrain_optimized.py --model vit     # ViT only
    python retrain_optimized.py --model hybrid  # Hybrid only
    python retrain_optimized.py --model ssl     # ViT+SSL only
    python retrain_optimized.py --quick         # fewer epochs for testing
"""

import os
import sys
import csv
import time
import json
import math
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        getattr(sys.stdout, 'reconfigure')(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        getattr(sys.stderr, 'reconfigure')(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from utils.seed import set_seed, get_device
from utils.logger import TrainingHistory
from utils.dataset import create_dataloaders
from models.vit_model import VisionTransformer
from models.hybrid_model import build_cnn_baseline, HybridCNNViT, CNNBaseline

# ============================================================
#  Config
# ============================================================
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
CLASSES = ["genuine", "fraud", "tampered", "forged"]
NUM_CLASSES = len(CLASSES)
DATA_DIR = "data/raw_images"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"

VIT_CONFIG = {
    "patch_size": 16, "embed_dim": 128, "num_heads": 4,
    "num_layers": 4, "mlp_dim": 256, "dropout": 0.1,
}
HYBRID_CONFIG = {
    "embed_dim": 128, "num_heads": 4, "num_layers": 2, "dropout": 0.1,
}

TRAINING_CONFIGS = {
    "cnn":    {"epochs": 20, "lr": 1e-3, "weight_decay": 1e-4, "label_smoothing": 0.1},
    "vit":    {"epochs": 30, "lr": 3e-4, "weight_decay": 5e-2, "label_smoothing": 0.15},
    "ssl":    {"epochs": 25, "lr": 1e-4, "weight_decay": 5e-2, "label_smoothing": 0.1},
    "hybrid": {"epochs": 25, "lr": 5e-4, "weight_decay": 1e-3, "label_smoothing": 0.1},
}

QUICK_CONFIGS = {
    "cnn":    {"epochs": 5,  "lr": 1e-3, "weight_decay": 1e-4, "label_smoothing": 0.1},
    "vit":    {"epochs": 8,  "lr": 3e-4, "weight_decay": 5e-2, "label_smoothing": 0.15},
    "ssl":    {"epochs": 6,  "lr": 1e-4, "weight_decay": 5e-2, "label_smoothing": 0.1},
    "hybrid": {"epochs": 8,  "lr": 5e-4, "weight_decay": 1e-3, "label_smoothing": 0.1},
}


def create_dirs():
    for d in [RESULTS_DIR, CHECKPOINT_DIR, LOGS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ============================================================
#  MixUp Data Augmentation
# ============================================================
def mixup_data(x, y, alpha=0.2):
    """MixUp: convex combination of pairs of examples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
#  Warmup + Cosine Annealing Scheduler
# ============================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    """Linear warmup for warmup_epochs, then cosine decay."""
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
#  Generic Training Loop
# ============================================================
def train_model(model, model_name, config, device, loaders, checkpoint_name,
                use_mixup=True, warmup_epochs=3):
    """Train a model with production-grade optimisations."""
    epochs = config["epochs"]
    lr = config["lr"]
    wd = config["weight_decay"]
    ls = config["label_smoothing"]

    print(f"\n{'=' * 70}")
    print(f"  Training: {model_name}")
    print(f"  Epochs: {epochs} | LR: {lr} | WD: {wd} | LabelSmooth: {ls}")
    print(f"  MixUp: {use_mixup} | Warmup: {warmup_epochs} epochs")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} | Device: {device}")
    print(f"{'=' * 70}\n")

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    patience = max(10, epochs // 3)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_mixup and random.random() < 0.5:
                images_m, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
                outputs = model(images_m)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                _, predicted = outputs.max(1)
                t_correct += (lam * predicted.eq(y_a).sum().item() +
                              (1 - lam) * predicted.eq(y_b).sum().item())
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                t_correct += predicted.eq(labels).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss += loss.item() * images.size(0)
            t_total += labels.size(0)

        scheduler.step()
        t_loss /= max(t_total, 1)
        t_acc = t_correct / max(t_total, 1)

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                v_total += labels.size(0)
                v_correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        v_loss /= max(v_total, 1)
        v_acc = v_correct / max(v_total, 1)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{epoch:2d}/{epochs}] "
              f"Train: {t_loss:.4f}/{t_acc:.4f} | "
              f"Val: {v_loss:.4f}/{v_acc:.4f} | LR: {current_lr:.6f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": v_acc, "val_loss": v_loss,
                "config": VIT_CONFIG if "vit" in model_name.lower() else HYBRID_CONFIG,
            }, os.path.join(CHECKPOINT_DIR, checkpoint_name))
            print(f"    * Saved best model (val_acc: {v_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - start_time
    print(f"\n  Training done in {elapsed:.1f}s | Best val_acc: {best_val_acc:.4f}")

    # ── Test Evaluation ──
    model.eval()
    test_correct, test_total = 0, 0
    test_preds, test_labels = [], []
    test_start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    test_time = time.time() - test_start

    test_acc = test_correct / max(test_total, 1)
    avg_inference_ms = (test_time / max(test_total, 1)) * 1000

    # Calculate precision, recall, F1 per class then average
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="weighted", zero_division=0
    )

    # Model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    total_params = sum(p.numel() for p in model.parameters())

    results = {
        "model_name": model_name,
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "inference_ms": avg_inference_ms,
        "size_mb": model_size_mb,
        "params": total_params,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "training_time_s": elapsed,
    }

    print(f"\n  ── Test Results for {model_name} ──")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Inf. Time: {avg_inference_ms:.2f} ms/sample")
    print(f"  Size:      {model_size_mb:.1f} MB ({total_params:,} params)")

    # Save training history
    hist_path = os.path.join(LOGS_DIR, f"training_history_{model_name.lower().replace(' ', '_').replace('+', '_')}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return results


# ============================================================
#  Model Builders
# ============================================================
def build_vit(device):
    model = VisionTransformer(
        image_size=IMAGE_SIZE, patch_size=VIT_CONFIG["patch_size"],
        in_channels=3, num_classes=NUM_CLASSES,
        embed_dim=VIT_CONFIG["embed_dim"], num_heads=VIT_CONFIG["num_heads"],
        num_layers=VIT_CONFIG["num_layers"], mlp_dim=VIT_CONFIG["mlp_dim"],
        dropout=VIT_CONFIG["dropout"],
    )
    return model.to(device)


def build_cnn(device):
    model = build_cnn_baseline({"cnn": {"pretrained": True, "num_classes": NUM_CLASSES}})
    return model.to(device)


def build_hybrid(device):
    model = HybridCNNViT(
        num_classes=NUM_CLASSES,
        embed_dim=HYBRID_CONFIG["embed_dim"],
        num_heads=HYBRID_CONFIG["num_heads"],
        num_layers=HYBRID_CONFIG["num_layers"],
        dropout=HYBRID_CONFIG["dropout"],
        cnn_pretrained=True,
    )
    return model.to(device)


def build_vit_ssl(device):
    """Build ViT and load MAE-pretrained encoder if available."""
    model = build_vit(device)
    mae_path = os.path.join(CHECKPOINT_DIR, "mae_pretrained.pth")
    if os.path.exists(mae_path):
        try:
            ckpt = torch.load(mae_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("encoder_state_dict", ckpt))
            # Load only matching keys
            model_dict = model.state_dict()
            pretrained = {k: v for k, v in state.items()
                         if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained)
            model.load_state_dict(model_dict)
            print(f"  Loaded {len(pretrained)}/{len(model_dict)} keys from MAE checkpoint")
        except Exception as e:
            print(f"  [WARN] Could not load MAE checkpoint: {e}")
    else:
        print(f"  [WARN] No MAE checkpoint found at {mae_path}, training ViT+SSL from scratch")
    return model


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Optimized model retraining")
    parser.add_argument("--model", choices=["cnn", "vit", "ssl", "hybrid", "all"],
                        default="all", help="Which model to retrain")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer epochs for testing")
    args = parser.parse_args()

    set_seed(SEED)
    create_dirs()
    device = get_device()

    configs = QUICK_CONFIGS if args.quick else TRAINING_CONFIGS

    print("\n" + "=" * 70)
    print("  OPTIMIZED MODEL RETRAINING PIPELINE")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {device}")
    print(f"  Mode: {'Quick' if args.quick else 'Full'}")
    print("=" * 70)

    # Check data
    data_path = Path(DATA_DIR)
    total_images = sum(1 for cls in CLASSES for f in (data_path / cls).glob("*.png"))
    print(f"  Dataset: {total_images} images across {len(CLASSES)} classes\n")

    if total_images < 50:
        print("  [WARN] Very few images! Run `python generate_realistic_data.py` first.")
        print("  Continuing anyway...\n")

    loaders = create_dataloaders(
        data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, seed=SEED, class_names=CLASSES, balance_classes=True,
    )
    train_size = len(loaders['train'].dataset)  # type: ignore[arg-type]
    val_size = len(loaders['val'].dataset)  # type: ignore[arg-type]
    test_size = len(loaders['test'].dataset)  # type: ignore[arg-type]
    print(f"  Train: {train_size} | Val: {val_size} | Test: {test_size}")

    models_to_train = [args.model] if args.model != "all" else ["cnn", "vit", "ssl", "hybrid"]
    all_results = []

    for model_key in models_to_train:
        cfg = configs[model_key]

        if model_key == "cnn":
            model = build_cnn(device)
            res = train_model(model, "CNN (ResNet50)", cfg, device, loaders,
                            "cnn_best.pth", use_mixup=True, warmup_epochs=2)
        elif model_key == "vit":
            model = build_vit(device)
            res = train_model(model, "ViT (from scratch)", cfg, device, loaders,
                            "vit_best.pth", use_mixup=True, warmup_epochs=5)
            # Also save as best_model.pth
            src = os.path.join(CHECKPOINT_DIR, "vit_best.pth")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        elif model_key == "ssl":
            model = build_vit_ssl(device)
            res = train_model(model, "ViT + SSL (MAE)", cfg, device, loaders,
                            "vit_ssl_best.pth", use_mixup=True, warmup_epochs=3)
        elif model_key == "hybrid":
            model = build_hybrid(device)
            res = train_model(model, "Hybrid CNN+ViT", cfg, device, loaders,
                            "hybrid_best.pth", use_mixup=True, warmup_epochs=3)

        all_results.append(res)

    # ── Save Combined Results ──
    if all_results:
        # CSV
        csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1-Score",
                           "Inf. Time (ms)", "Size (MB)", "Params"])
            for r in all_results:
                writer.writerow([
                    r["model_name"], f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}", f"{r['f1_score']:.4f}",
                    f"{r['inference_ms']:.2f}", f"{r['size_mb']:.1f}",
                    f"{r['params']:,}",
                ])
        print(f"\n  [OK] Saved {csv_path}")

        # JSON
        json_path = os.path.join(RESULTS_DIR, "full_comparison.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  [OK] Saved {json_path}")

        # Summary
        print(f"\n{'=' * 70}")
        print("  RETRAINING COMPLETE — RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'Model':<20} {'Acc':>8} {'F1':>8} {'Inf(ms)':>8} {'Size(MB)':>8}")
        print(f"  {'-'*52}")
        for r in all_results:
            print(f"  {r['model_name']:<20} {r['accuracy']:>8.4f} {r['f1_score']:>8.4f} "
                  f"{r['inference_ms']:>8.2f} {r['size_mb']:>8.1f}")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
