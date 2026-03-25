"""
Full Pipeline Execution — All 5 Tasks
========================================
Task 1: Train longer / bigger model (15 epochs, medium ViT)
Task 2: MAE self-supervised pretraining + fine-tuning
Task 3: Model comparison study (CNN vs ViT vs ViT+SSL vs Hybrid)
Task 4: Test PDF report generation
Task 5: Enhanced data pipeline (generate more diverse data)
========================================
Usage:
    py run_all_tasks.py
"""

import os
import sys
import time
import json
import math
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.seed import set_seed, get_device
from utils.logger import setup_logger, TrainingHistory, HyperparameterLogger
from utils.dataset import create_dataloaders
from utils.checkpoint_loader import load_vit_from_checkpoint
from models.vit_model import VisionTransformer
from models.hybrid_model import build_cnn_baseline, HybridCNNViT, CNNBaseline
from analytics.performance_metrics import MetricsCalculator, plot_confusion_matrix, plot_training_history


# ===================================================================
#  Shared Config (Medium-sized models for CPU, enough to learn)
# ===================================================================
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
CLASSES = ["genuine", "fraud", "tampered", "forged"]
DATA_DIR = "data/raw_images"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

# Medium ViT — larger than tiny but still CPU-feasible
MEDIUM_VIT = {
    "patch_size": 16,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 4,
    "mlp_dim": 256,
    "dropout": 0.1,
}

# Tiny ViT — for MAE pretraining (lighter decoder)
TINY_VIT_SSL = {
    "patch_size": 16,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 4,
    "mlp_dim": 256,
    "dropout": 0.1,
}


def warm_up_module_usage() -> None:
    """Optionally touch all modules to keep project-wide module usage active."""
    try:
        from module_usage_manifest import touch_all_modules

        summary: dict[str, Any] = touch_all_modules()
        loaded = summary.get("loaded", 0)
        total = summary.get("total", 0)

        failed_raw = summary.get("failed", [])
        failed: list[dict[str, Any]] = []
        if isinstance(failed_raw, list):
            failed = [item for item in failed_raw if isinstance(item, dict)]

        print(
            f"[module-usage] loaded={loaded}/{total} "
            f"failed={len(failed)}"
        )
        if failed:
            preview = failed[:3]
            for item in preview:
                print(f"  - {item.get('module')}: {item.get('error')}")
            if len(failed) > len(preview):
                print(f"  ... {len(failed) - len(preview)} more failures")
    except Exception as exc:
        print(f"[module-usage] skipped: {exc}")


def create_dirs():
    for d in [RESULTS_DIR, CHECKPOINT_DIR, "logs", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def build_medium_vit():
    return VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=MEDIUM_VIT["patch_size"],
        in_channels=3,
        num_classes=len(CLASSES),
        embed_dim=MEDIUM_VIT["embed_dim"],
        num_heads=MEDIUM_VIT["num_heads"],
        num_layers=MEDIUM_VIT["num_layers"],
        mlp_dim=MEDIUM_VIT["mlp_dim"],
        dropout=MEDIUM_VIT["dropout"],
    )


def get_dataloaders():
    return create_dataloaders(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        class_names=CLASSES,
        balance_classes=True,
    )


# ===================================================================
#  TASK 1: Train Longer / Bigger Model
# ===================================================================
def task1_train_bigger_model():
    """Train a medium ViT for 15 epochs."""
    EPOCHS = 15
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1

    print("\n" + "=" * 70)
    print("  TASK 1: Train Medium ViT (15 epochs)")
    print("=" * 70)

    device = get_device()
    loaders = get_dataloaders()
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    model = build_medium_vit().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} params | {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  Config: patch={MEDIUM_VIT['patch_size']}, embed={MEDIUM_VIT['embed_dim']}, "
          f"heads={MEDIUM_VIT['num_heads']}, layers={MEDIUM_VIT['num_layers']}")
    print(f"  Epochs: {EPOCHS} | LR: {LR} | Device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    history = TrainingHistory()
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
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
        val_loss, val_correct, val_total = 0.0, 0, 0
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
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        })
        print(f"  Epoch [{epoch:2d}/{EPOCHS}] "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc, "val_loss": val_loss,
                "config": MEDIUM_VIT,
            }, os.path.join(CHECKPOINT_DIR, "vit_best.pth"))
            print(f"    -> Saved best model (val_acc: {val_acc:.4f})")

    total_time = time.time() - start_time
    print(f"\n  Training done in {total_time:.1f}s | Best val_acc: {best_val_acc:.4f}")

    # Also save as best_model.pth for dashboard compatibility
    shutil.copy(
        os.path.join(CHECKPOINT_DIR, "vit_best.pth"),
        os.path.join(CHECKPOINT_DIR, "best_model.pth"),
    )

    # Evaluate on test
    print("\n  Evaluating on test set...")
    calc = MetricsCalculator(class_names=CLASSES)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = calc.compute_all_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    print(f"\n  TEST: Acc={metrics['accuracy']:.4f} | "
          f"F1={metrics['f1_macro']:.4f} | "
          f"Prec={metrics['precision_macro']:.4f} | "
          f"Rec={metrics['recall_macro']:.4f}")

    # Save artifacts
    with open(os.path.join(RESULTS_DIR, "test_metrics_vit.json"), "w") as f:
        serializable = {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in metrics.items() if not isinstance(v, np.ndarray)}
        json.dump(serializable, f, indent=2)

    try:
        plot_confusion_matrix(
            metrics["confusion_matrix"], CLASSES,
            save_path=os.path.join(RESULTS_DIR, "confusion_matrix_vit.png"),
        )
        plot_training_history(
            history.history,
            save_path=os.path.join(RESULTS_DIR, "training_curves_vit.png"),
        )
    except Exception as e:
        print(f"  Warning: Plot error: {e}")

    history.save("training_history_vit.json")
    print("  [TASK 1 COMPLETE]\n")
    return model, metrics


# ===================================================================
#  TASK 2: MAE Self-Supervised Pretraining + Fine-Tuning
# ===================================================================
def task2_mae_pretraining():
    """Pretrain MAE, then fine-tune with labels."""
    MAE_EPOCHS = 10
    FINETUNE_EPOCHS = 10
    MAE_LR = 1.5e-4
    FINETUNE_LR = 3e-4

    print("\n" + "=" * 70)
    print("  TASK 2: MAE Self-Supervised Pretraining + Fine-Tuning")
    print("=" * 70)

    device = get_device()
    loaders = get_dataloaders()
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    # ---- Phase A: MAE Pretraining ----
    print(f"\n  Phase A: MAE Pretraining ({MAE_EPOCHS} epochs)")
    print("  " + "-" * 50)

    from ssl_pretraining.mae_model import MaskedAutoencoder, pretrain_mae

    mae = MaskedAutoencoder(
        image_size=IMAGE_SIZE,
        patch_size=TINY_VIT_SSL["patch_size"],
        in_channels=3,
        encoder_dim=TINY_VIT_SSL["embed_dim"],
        encoder_heads=TINY_VIT_SSL["num_heads"],
        encoder_depth=TINY_VIT_SSL["num_layers"],
        decoder_dim=64,
        decoder_heads=4,
        decoder_depth=2,
        mask_ratio=0.75,
    )
    total_params = sum(p.numel() for p in mae.parameters())
    print(f"  MAE params: {total_params:,}")

    mae_history = pretrain_mae(
        model=mae,
        dataloader=train_loader,
        epochs=MAE_EPOCHS,
        lr=MAE_LR,
        weight_decay=0.05,
        warmup_epochs=2,
        device=device,
        save_path=os.path.join(CHECKPOINT_DIR, "mae_pretrained.pth"),
    )

    # Save MAE training history
    with open(os.path.join(RESULTS_DIR, "mae_pretrain_history.json"), "w") as f:
        json.dump(mae_history, f, indent=2)

    # ---- Phase B: Transfer encoder → ViT classifier ----
    print(f"\n  Phase B: Fine-Tuning with Labels ({FINETUNE_EPOCHS} epochs)")
    print("  " + "-" * 50)

    # Build classifier with same encoder architecture
    vit_ssl = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=TINY_VIT_SSL["patch_size"],
        in_channels=3,
        num_classes=len(CLASSES),
        embed_dim=TINY_VIT_SSL["embed_dim"],
        num_heads=TINY_VIT_SSL["num_heads"],
        num_layers=TINY_VIT_SSL["num_layers"],
        mlp_dim=TINY_VIT_SSL["mlp_dim"],
        dropout=TINY_VIT_SSL["dropout"],
    ).to(device)

    # Load MAE encoder weights into ViT
    mae_ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "mae_pretrained.pth"),
        map_location=device,
    )
    encoder_state = mae_ckpt["encoder_state_dict"]

    # Map MAE encoder keys → ViT keys
    vit_state = vit_ssl.state_dict()
    loaded_keys = 0
    for key in encoder_state:
        # MAE encoder has same structure: patch_embed, pos_embed, cls_token, blocks, norm
        mapped_key = key
        # Map blocks → encoder.layers
        if key.startswith("blocks."):
            # MAE: blocks.0.xxx → ViT: encoder.layers.0.xxx
            mapped_key = "encoder.layers." + key[len("blocks."):]
        elif key == "norm.weight":
            mapped_key = "encoder.norm.weight"
        elif key == "norm.bias":
            mapped_key = "encoder.norm.bias"
        elif key.startswith("patch_embed."):
            mapped_key = "patch_embedding." + key[len("patch_embed."):]
            # MAE uses .proj, ViT uses .projection
            mapped_key = mapped_key.replace(".proj.", ".projection.")
        elif key == "cls_token":
            mapped_key = "cls_token"
        elif key == "pos_embed":
            mapped_key = "pos_embedding"

        if mapped_key in vit_state and vit_state[mapped_key].shape == encoder_state[key].shape:
            vit_state[mapped_key] = encoder_state[key]
            loaded_keys += 1

    vit_ssl.load_state_dict(vit_state)
    print(f"  Loaded {loaded_keys} / {len(encoder_state)} encoder weights into ViT")

    # Fine-tune
    optimizer = torch.optim.AdamW(vit_ssl.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = TrainingHistory()
    best_val_acc = 0.0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        vit_ssl.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = vit_ssl(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vit_ssl.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        vit_ssl.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = vit_ssl(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history.update(epoch, {
            "train_loss": train_loss, "val_loss": val_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        })
        print(f"  Epoch [{epoch:2d}/{FINETUNE_EPOCHS}] "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": vit_ssl.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc, "val_loss": val_loss,
                "config": TINY_VIT_SSL,
                "pretrained": "mae",
            }, os.path.join(CHECKPOINT_DIR, "vit_ssl_best.pth"))
            print(f"    -> Saved best SSL model (val_acc: {val_acc:.4f})")

    # Evaluate
    print("\n  Evaluating SSL model on test set...")
    calc = MetricsCalculator(class_names=CLASSES)
    vit_ssl.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vit_ssl(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = calc.compute_all_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    print(f"\n  TEST (SSL): Acc={metrics['accuracy']:.4f} | "
          f"F1={metrics['f1_macro']:.4f}")

    with open(os.path.join(RESULTS_DIR, "test_metrics_vit_ssl.json"), "w") as f:
        serializable = {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in metrics.items() if not isinstance(v, np.ndarray)}
        json.dump(serializable, f, indent=2)

    history.save("training_history_vit_ssl.json")
    print("  [TASK 2 COMPLETE]\n")
    return vit_ssl, metrics


# ===================================================================
#  TASK 3: Model Comparison Study
# ===================================================================
def task3_model_comparison():
    """Train CNN baseline + Hybrid, compare all models."""
    CNN_EPOCHS = 10
    HYBRID_EPOCHS = 10

    print("\n" + "=" * 70)
    print("  TASK 3: Model Comparison Study")
    print("=" * 70)

    device = get_device()
    loaders = get_dataloaders()
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
    calc = MetricsCalculator(class_names=CLASSES, output_dir=RESULTS_DIR)

    all_results = []

    # ---- 3a: CNN Baseline (ResNet50) ----
    print(f"\n  [3a] Training CNN Baseline (ResNet50) — {CNN_EPOCHS} epochs")
    cnn = CNNBaseline(pretrained=True, num_classes=len(CLASSES), dropout=0.3).to(device)
    cnn_params = sum(p.numel() for p in cnn.parameters())
    print(f"  CNN params: {cnn_params:,}")

    optimizer = torch.optim.AdamW(cnn.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0

    for epoch in range(1, CNN_EPOCHS + 1):
        cnn.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        cnn.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = val_correct / max(val_total, 1)
        print(f"    Epoch [{epoch:2d}/{CNN_EPOCHS}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": cnn.state_dict(),
                "val_acc": val_acc,
            }, os.path.join(CHECKPOINT_DIR, "cnn_best.pth"))

    # Evaluate CNN
    cnn_metrics, _ = calc.evaluate_model(cnn, test_loader, device)
    cnn_metrics.update(calc.get_model_size(cnn))
    cnn_metrics["model_name"] = "CNN (ResNet50)"
    all_results.append(cnn_metrics)
    print(f"  CNN Test: Acc={cnn_metrics['accuracy']:.4f} | F1={cnn_metrics['f1_macro']:.4f}")

    # ---- 3b: Load ViT (from Task 1) ----
    print(f"\n  [3b] Loading ViT (from Task 1)...")
    vit_ckpt_path = os.path.join(CHECKPOINT_DIR, "vit_best.pth")
    if os.path.exists(vit_ckpt_path):
        vit, _, loaded_path, _ = load_vit_from_checkpoint(
            checkpoint_candidates=[
                os.path.join(CHECKPOINT_DIR, "vit_best.pth"),
                os.path.join(CHECKPOINT_DIR, "best_model.pth"),
            ],
            device=device,
            image_size=IMAGE_SIZE,
            num_classes=len(CLASSES),
            default_config=MEDIUM_VIT,
        )
        print(f"  Loaded checkpoint: {loaded_path}")
        vit_metrics, _ = calc.evaluate_model(vit, test_loader, device)
        vit_metrics.update(calc.get_model_size(vit))
        vit_metrics["model_name"] = "ViT (from scratch)"
        all_results.append(vit_metrics)
        print(f"  ViT Test: Acc={vit_metrics['accuracy']:.4f} | F1={vit_metrics['f1_macro']:.4f}")
    else:
        print("  ViT checkpoint not found — skipping")

    # ---- 3c: Load ViT + SSL (from Task 2) ----
    print(f"\n  [3c] Loading ViT + SSL (from Task 2)...")
    ssl_ckpt_path = os.path.join(CHECKPOINT_DIR, "vit_ssl_best.pth")
    if os.path.exists(ssl_ckpt_path):
        vit_ssl = VisionTransformer(
            image_size=IMAGE_SIZE,
            patch_size=TINY_VIT_SSL["patch_size"],
            in_channels=3,
            num_classes=len(CLASSES),
            embed_dim=TINY_VIT_SSL["embed_dim"],
            num_heads=TINY_VIT_SSL["num_heads"],
            num_layers=TINY_VIT_SSL["num_layers"],
            mlp_dim=TINY_VIT_SSL["mlp_dim"],
            dropout=TINY_VIT_SSL["dropout"],
        ).to(device)
        ckpt = torch.load(ssl_ckpt_path, map_location=device)
        vit_ssl.load_state_dict(ckpt["model_state_dict"])
        ssl_metrics, _ = calc.evaluate_model(vit_ssl, test_loader, device)
        ssl_metrics.update(calc.get_model_size(vit_ssl))
        ssl_metrics["model_name"] = "ViT + SSL (MAE)"
        all_results.append(ssl_metrics)
        print(f"  SSL Test: Acc={ssl_metrics['accuracy']:.4f} | F1={ssl_metrics['f1_macro']:.4f}")
    else:
        print("  SSL checkpoint not found — skipping")

    # ---- 3d: Hybrid CNN+ViT ----
    print(f"\n  [3d] Training Hybrid CNN+ViT — {HYBRID_EPOCHS} epochs")
    hybrid = HybridCNNViT(
        cnn_pretrained=True,
        cnn_features=2048,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=len(CLASSES),
    ).to(device)
    hybrid_params = sum(p.numel() for p in hybrid.parameters())
    print(f"  Hybrid params: {hybrid_params:,}")

    # Freeze CNN initially
    hybrid.freeze_cnn()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, hybrid.parameters()),
        lr=3e-4, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HYBRID_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0.0

    for epoch in range(1, HYBRID_EPOCHS + 1):
        # Unfreeze CNN after 5 epochs
        if epoch == 6:
            hybrid.unfreeze_cnn()
            optimizer = torch.optim.AdamW(hybrid.parameters(), lr=1e-5, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=HYBRID_EPOCHS - 5
            )
            print("    -> Unfroze CNN backbone, LR=1e-5")

        hybrid.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = hybrid(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hybrid.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        hybrid.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = hybrid(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = val_correct / max(val_total, 1)
        print(f"    Epoch [{epoch:2d}/{HYBRID_EPOCHS}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": hybrid.state_dict(),
                "val_acc": val_acc,
            }, os.path.join(CHECKPOINT_DIR, "hybrid_best.pth"))

    # Evaluate Hybrid
    hybrid_metrics, _ = calc.evaluate_model(hybrid, test_loader, device)
    hybrid_metrics.update(calc.get_model_size(hybrid))
    hybrid_metrics["model_name"] = "Hybrid CNN+ViT"
    all_results.append(hybrid_metrics)
    print(f"  Hybrid Test: Acc={hybrid_metrics['accuracy']:.4f} | F1={hybrid_metrics['f1_macro']:.4f}")

    # ---- Generate Comparison ----
    print(f"\n  Generating comparison charts...")
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model_names = [r["model_name"] for r in all_results]

        # Classification metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
        metric_keys = [
            ("accuracy", "Accuracy"),
            ("precision_macro", "Precision (Macro)"),
            ("recall_macro", "Recall (Macro)"),
            ("f1_macro", "F1-Score (Macro)"),
        ]
        for idx, (key, label) in enumerate(metric_keys):
            values = [r.get(key, 0) for r in all_results]
            bars = axes[idx].bar(model_names, values, color=colors[:len(model_names)])
            axes[idx].set_title(label, fontsize=13, fontweight="bold")
            axes[idx].set_ylim(0, 1.05)
            axes[idx].grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                               f"{val:.3f}", ha="center", fontsize=10)
            axes[idx].tick_params(axis='x', rotation=15)

        fig.suptitle("Model Comparison — Classification Metrics", fontsize=15, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "comparison_classification.png"), dpi=150)
        plt.close(fig)

        # Efficiency metrics
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        inf_times = [r.get("inference_time_per_sample_ms", 0) for r in all_results]
        bars = axes[0].bar(model_names, inf_times, color="#FF5722")
        axes[0].set_title("Inference Time (ms/sample)", fontweight="bold")
        axes[0].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, inf_times):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f"{val:.2f}", ha="center", fontsize=10)

        sizes = [r.get("model_size_mb", 0) for r in all_results]
        bars = axes[1].bar(model_names, sizes, color="#9C27B0")
        axes[1].set_title("Model Size (MB)", fontweight="bold")
        axes[1].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, sizes):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}", ha="center", fontsize=10)

        params_list = [r.get("total_parameters", 0) / 1e6 for r in all_results]
        bars = axes[2].bar(model_names, params_list, color="#009688")
        axes[2].set_title("Parameters (Millions)", fontweight="bold")
        axes[2].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, params_list):
            axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                         f"{val:.1f}M", ha="center", fontsize=10)

        fig.suptitle("Model Comparison — Efficiency", fontsize=15, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "comparison_efficiency.png"), dpi=150)
        plt.close(fig)

        # Comparison table
        rows = []
        for r in all_results:
            rows.append({
                "Model": r["model_name"],
                "Accuracy": f"{r.get('accuracy', 0):.4f}",
                "Precision": f"{r.get('precision_macro', 0):.4f}",
                "Recall": f"{r.get('recall_macro', 0):.4f}",
                "F1-Score": f"{r.get('f1_macro', 0):.4f}",
                "Inf. Time (ms)": f"{r.get('inference_time_per_sample_ms', 0):.2f}",
                "Size (MB)": f"{r.get('model_size_mb', 0):.1f}",
                "Params": f"{r.get('total_parameters', 0):,}",
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
        print(f"\n{'='*90}")
        print("  MODEL COMPARISON RESULTS")
        print(f"{'='*90}")
        print(df.to_string(index=False))
        print(f"{'='*90}")

        # Save JSON
        json_results = []
        for r in all_results:
            cleaned = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in r.items() if not isinstance(v, np.ndarray)}
            json_results.append(cleaned)
        with open(os.path.join(RESULTS_DIR, "full_comparison.json"), "w") as f:
            json.dump({"models": json_results}, f, indent=2)

    except Exception as e:
        print(f"  Warning: Comparison chart error: {e}")

    print("  [TASK 3 COMPLETE]\n")
    return all_results


# ===================================================================
#  TASK 4: Test PDF Report Generation
# ===================================================================
def task4_test_pdf_report():
    """Generate a PDF audit report using the best model."""
    print("\n" + "=" * 70)
    print("  TASK 4: PDF Report Generation")
    print("=" * 70)

    device = get_device()

    from explainability.report_generator import ExplainabilityReport
    from explainability.attention_visualization import (
        compute_attention_rollout,
        attention_to_heatmap,
        overlay_attention_on_image,
    )
    from analytics.risk_scoring import FraudRiskScorer
    from utils.augmentation import get_val_transforms
    from PIL import Image
    import cv2

    # Load best model (from Task 1)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "vit_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("  No trained model found — skipping PDF generation")
        return

    model, _, loaded_path, model_cfg = load_vit_from_checkpoint(
        checkpoint_candidates=[
            os.path.join(CHECKPOINT_DIR, "vit_best.pth"),
            os.path.join(CHECKPOINT_DIR, "best_model.pth"),
        ],
        device=device,
        image_size=IMAGE_SIZE,
        num_classes=len(CLASSES),
        default_config=MEDIUM_VIT,
    )
    print(f"  Loaded checkpoint: {loaded_path}")

    transform = get_val_transforms(IMAGE_SIZE)

    # Pick sample images (one per class)
    reporter = ExplainabilityReport(output_dir="reports/")
    scorer = FraudRiskScorer()

    for cls_name in CLASSES:
        cls_dir = Path(DATA_DIR) / cls_name
        if not cls_dir.exists():
            continue
        candidates: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            candidates.extend(cls_dir.glob(ext))
        if not candidates:
            print(f"  No sample images found for class '{cls_name}' — skipping")
            continue
        sample_path = candidates[0]
        image = Image.open(sample_path).convert("RGB")
        original_np = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))

        # Predict
        image_tensor: torch.Tensor = transform(image).unsqueeze(0).to(device)  # type: ignore[union-attr]
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
        pred_idx = outputs.argmax(dim=1).item()
        pred_label = CLASSES[pred_idx]
        confidence = probs[0, pred_idx].item()
        class_probs = {CLASSES[i]: probs[0, i].item() for i in range(len(CLASSES))}

        # Attention overlay
        attention_overlay = None
        try:
            _ = model(image_tensor)  # forward pass for attention maps
            attn_maps = model.get_attention_maps()
            if attn_maps:
                rollout = compute_attention_rollout(attn_maps)
                if rollout is not None:
                    patch_size = int(model_cfg.get("patch_size", MEDIUM_VIT["patch_size"]))
                    heatmap = attention_to_heatmap(rollout, IMAGE_SIZE, patch_size)
                    attention_overlay = overlay_attention_on_image(original_np, heatmap)
        except Exception as e:
            print(f"    Attention error for {cls_name}: {e}")

        # Risk score
        conf_risk = scorer.compute_confidence_score(class_probs, pred_label)
        risk_score, risk_level = scorer.compute_risk_score(0.5, conf_risk, 0.5)

        # Generate report
        doc_id = f"TASK4_{cls_name.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            report_path = reporter.generate(
                document_image=original_np,
                prediction=pred_label,
                confidence=confidence,
                class_probabilities=class_probs,
                attention_overlay=attention_overlay,
                document_id=doc_id,
                metadata={
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "model": "Medium ViT",
                    "source_file": str(sample_path),
                },
            )
            print(f"  Report generated: {report_path}")
        except Exception as e:
            print(f"  Report error for {cls_name}: {e}")

    print("  [TASK 4 COMPLETE]\n")


# ===================================================================
#  TASK 5: Enhanced Data Pipeline
# ===================================================================
def task5_enhanced_data():
    """Validate real data is available before training."""
    print("\n" + "=" * 70)
    print("  TASK 5: Data Validation & Preparation")
    print("=" * 70)

    output_dir = Path(DATA_DIR)

    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    classes = ['genuine', 'fraud', 'tampered', 'forged']
    total_images = 0

    for cls in classes:
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        count = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS) if cls_dir.exists() else 0
        total_images += count
        status = "OK" if count >= 10 else "NEEDS DATA"
        print(f"  {cls:12s}: {count:5d} images  [{status}]")

    if total_images < 40:
        print("\n  WARNING: Not enough real data for training.")
        print("  Place real bank document images into data/raw_images/{class}/")
        print("  Or run: python setup_datasets.py --check")
        print("  See REAL_DATA_GUIDE.md for details.")
        print("\n  Skipping training tasks — add real data first.")
        print("  [TASK 5 COMPLETE — data insufficient]\n")
        return False

    print(f"\n  Total images available: {total_images}")
    print(f"  Data directory: {output_dir}")
    print("  [TASK 5 COMPLETE]\n")
    return True


# ===================================================================
#  Main — Run All Tasks
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Run full or partial 5-task pipeline.")
    parser.add_argument(
        "--only-task3",
        action="store_true",
        help="Run only Task 3 (model comparison study).",
    )
    parser.add_argument(
        "--only-task4",
        action="store_true",
        help="Run only Task 4 (PDF report generation).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training-heavy tasks and run quick validation tasks only.",
    )
    args = parser.parse_args()

    if args.only_task3 and args.only_task4:
        raise SystemExit("Choose only one of --only-task3 or --only-task4.")

    set_seed(SEED)
    create_dirs()

    # Module warmup imports many files and is expensive; skip it in quick-run modes.
    if not (args.only_task3 or args.only_task4 or args.skip_training):
        warm_up_module_usage()

    print("\n" + "#" * 70)
    print("  FULL PIPELINE — 5 TASKS")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {get_device()}")
    print("#" * 70)

    start = time.time()

    if args.only_task3:
        print("\n  Quick mode: running only Task 3")
        task3_model_comparison()
        total = time.time() - start
        print(f"\n  DONE (only Task 3) in {total / 60:.1f} minutes")
        return

    if args.only_task4:
        print("\n  Quick mode: running only Task 4")
        task4_test_pdf_report()
        total = time.time() - start
        print(f"\n  DONE (only Task 4) in {total / 60:.1f} minutes")
        return

    if args.skip_training:
        print("\n  Quick mode: --skip-training enabled")
        print("  Skipping Tasks 1 and 2; running Tasks 3 and 4 for validation.")
        task3_model_comparison()
        task4_test_pdf_report()
        total = time.time() - start
        print(f"\n  DONE (skip-training mode) in {total / 60:.1f} minutes")
        return

    # Task 5 first — validate real data before training
    has_data = task5_enhanced_data()

    if not has_data:
        print("\n  Pipeline aborted — no real training data.")
        print("  See REAL_DATA_GUIDE.md for setup instructions.")
        return

    # Task 1 — Train with real data
    model_vit, metrics_vit = task1_train_bigger_model()

    # Task 2 — MAE pretraining + fine-tuning
    model_ssl, metrics_ssl = task2_mae_pretraining()

    # Task 3 — Compare all models
    comparison = task3_model_comparison()

    # Task 4 — PDF report generation
    task4_test_pdf_report()

    total = time.time() - start
    print("\n" + "#" * 70)
    print(f"  ALL 5 TASKS COMPLETE!")
    print(f"  Total time: {total / 60:.1f} minutes")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    print("\n  Output files:")
    for f in sorted(Path(RESULTS_DIR).glob("*")):
        print(f"    {f}")
    print(f"\n  Checkpoints:")
    for f in sorted(Path(CHECKPOINT_DIR).glob("*")):
        print(f"    {f}")
    print(f"\n  Reports:")
    for f in sorted(Path("reports").glob("*")):
        print(f"    {f}")

    print(f"\n  To launch frontend: cd frontend && npm run dev")


if __name__ == "__main__":
    main()
