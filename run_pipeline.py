"""
Training Pipeline — Real Data
===============================
Trains the ViT fraud detector on REAL bank document images.

Pre-requisites:
  1. Place images in data/raw_images/{genuine,fraud,tampered,forged}/
  2. Run: python setup_datasets.py --prepare

Usage:
    py run_pipeline.py
"""

import os
import sys
import time
import json
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from utils.seed import set_seed, get_device
from utils.logger import setup_logger, TrainingHistory, HyperparameterLogger
from utils.dataset import create_dataloaders
from models.vit_model import VisionTransformer
from models.hybrid_model import build_cnn_baseline
from analytics.performance_metrics import MetricsCalculator, plot_confusion_matrix, plot_training_history


# ------------------------------------------------------------------ #
#  Configuration (uses processed real images)
# ------------------------------------------------------------------ #
QUICK_CONFIG = {
    "seed": 42,
    "data_dir": "data/processed",       # prepared from data/raw_images
    "image_size": 224,
    "batch_size": 8,
    "num_workers": 0,
    "epochs": 10,            # Increase for production (50-100)
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "classes": ["genuine", "fraud", "tampered", "forged"],
    # Compact ViT (increase dimensions if GPU available)
    "vit": {
        "patch_size": 16,    # Standard 16x16 patches
        "embed_dim": 128,    # Moderate embedding
        "num_heads": 4,
        "num_layers": 4,
        "mlp_dim": 256,
        "dropout": 0.1,
    },
    "output_dir": "results",
    "checkpoint_dir": "checkpoints",
}


def run_training():
    """Train ViT on real document images."""
    cfg = QUICK_CONFIG
    set_seed(cfg["seed"])
    device = get_device()

    # --- Validate data exists ---
    data_path = Path(cfg["data_dir"])
    if not data_path.exists() or not any(data_path.iterdir()):
        print("❌  No data found in", cfg["data_dir"])
        print("   Please place your bank document images in data/raw_images/{genuine,fraud,tampered,forged}/")
        print("   Then run:  python setup_datasets.py --prepare")
        sys.exit(1)

    # Count images per class
    total_images = 0
    for cls in cfg["classes"]:
        cls_dir = data_path / cls
        if cls_dir.exists():
            n = len([f for f in cls_dir.iterdir() if f.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.bmp'}])
            total_images += n
            print(f"  {cls}: {n} images")
    
    if total_images < 10:
        print(f"\n❌  Only {total_images} images found — need at least 10 to train.")
        print("   Add more images to data/raw_images/ and run: python setup_datasets.py --prepare")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ViT Training Pipeline — Real Data")
    print(f"  Device: {device}")
    print(f"  Total images: {total_images}")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"{'='*60}\n")

    # Setup dirs
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    logger = setup_logger("pipeline", "logs/pipeline.log")

    # ---- Data ----
    print("[1/5] Loading dataset...")
    loaders = create_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        class_names=cfg["classes"],
        balance_classes=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # ---- Model ----
    print("\n[2/5] Building ViT model...")
    model = VisionTransformer(
        image_size=cfg["image_size"],
        patch_size=cfg["vit"]["patch_size"],
        in_channels=3,
        num_classes=len(cfg["classes"]),
        embed_dim=cfg["vit"]["embed_dim"],
        num_heads=cfg["vit"]["num_heads"],
        num_layers=cfg["vit"]["num_layers"],
        mlp_dim=cfg["vit"]["mlp_dim"],
        dropout=cfg["vit"]["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # ---- Training ----
    print(f"\n[3/5] Training for {cfg['epochs']} epochs...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    history = TrainingHistory()
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history.update(epoch, {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        print(
            f"  Epoch [{epoch}/{cfg['epochs']}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, os.path.join(cfg["checkpoint_dir"], "best_model.pth"))
            print(f"    → Saved best model (acc: {val_acc:.4f})")

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s | Best val acc: {best_val_acc:.4f}")

    # Save training history
    history.save("training_history.json")

    # ---- Evaluation ----
    print("\n[4/5] Evaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    calculator = MetricsCalculator(class_names=cfg["classes"])
    metrics = calculator.compute_all_metrics(all_labels, all_preds, all_probs)

    print(f"\n  {'='*40}")
    print(f"  TEST RESULTS")
    print(f"  {'='*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1 Score:  {metrics['f1_macro']:.4f}")
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  {'='*40}")

    # Save metrics
    metrics_path = os.path.join(cfg["output_dir"], "test_metrics.json")
    serializable = {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()
                    if not isinstance(v, np.ndarray)}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # Plot confusion matrix
    try:
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            cfg["classes"],
            save_path=os.path.join(cfg["output_dir"], "confusion_matrix.png"),
        )
        print(f"  Confusion matrix saved: {cfg['output_dir']}/confusion_matrix.png")
    except Exception as e:
        print(f"  Warning: Could not plot confusion matrix: {e}")

    # Plot training history
    try:
        plot_training_history(
            history.history,
            save_path=os.path.join(cfg["output_dir"], "training_curves.png"),
        )
        print(f"  Training curves saved: {cfg['output_dir']}/training_curves.png")
    except Exception as e:
        print(f"  Warning: Could not plot training curves: {e}")

    # ---- Summary ----
    print(f"\n[5/5] Pipeline complete!")
    print(f"\n  Output files:")
    print(f"    {cfg['checkpoint_dir']}/best_model.pth")
    print(f"    {cfg['output_dir']}/training_history.json")
    print(f"    {cfg['output_dir']}/test_metrics.json")
    print(f"    {cfg['output_dir']}/confusion_matrix.png")
    print(f"    {cfg['output_dir']}/training_curves.png")
    print(f"\n  To launch frontend:")
    print(f"    cd frontend && npm install && npm run dev")
    print(f"\n{'='*60}")

    # Log hyperparams
    hp_logger = HyperparameterLogger("logs/hyperparameters.json")
    hp_logger.log({
        "experiment": "quick_run",
        "config": cfg,
        "results": {
            "best_val_acc": best_val_acc,
            "test_accuracy": float(metrics["accuracy"]),
            "test_f1": float(metrics["f1_macro"]),
            "total_training_time_seconds": total_time,
            "total_params": total_params,
        },
    })


if __name__ == "__main__":
    run_training()
