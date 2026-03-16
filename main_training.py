"""
Main Training Script for ViT Banking Fraud Detection.
============================================================
Orchestrates the full training pipeline:
  1. Load configuration
  2. Set reproducibility seeds
  3. Prepare data (clean, augment, split)
  4. Optional: Self-supervised pretraining (MAE / Contrastive)
  5. Fine-tune Vision Transformer
  6. Evaluate and save results
  7. Generate explainability reports
============================================================
Usage:
  python main_training.py                        # Full pipeline
  python main_training.py --skip-ssl             # Skip SSL pretraining
  python main_training.py --model hybrid         # Use CNN+ViT hybrid
  python main_training.py --pretrained-encoder checkpoints/mae_pretrained.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from typing import Any

# Project imports
from utils.seed import set_seed, get_device
from utils.logger import setup_logger, TrainingHistory, HyperparameterLogger
from utils.dataset import create_dataloaders, SSLImageDataset
from utils.augmentation import get_ssl_transforms
from models.vit_model import VisionTransformer, build_vit
from models.hybrid_model import HybridCNNViT, CNNBaseline, build_hybrid, build_cnn_baseline
from ssl_pretraining.mae_model import MaskedAutoencoder, pretrain_mae
from ssl_pretraining.contrastive_model import ContrastiveModel, pretrain_contrastive
from analytics.performance_metrics import MetricsCalculator, plot_training_history, export_results_json


# ------------------------------------------------------------------ #
#  Configuration Loading
# ------------------------------------------------------------------ #
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected dict from config, got {type(config)}")
    return config


# ------------------------------------------------------------------ #
#  Training Loop
# ------------------------------------------------------------------ #
def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {
        "train_loss": total_loss / max(total, 1),
        "train_acc": correct / max(total, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import f1_score, roc_auc_score

    val_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    return {
        "val_loss": total_loss / max(total, 1),
        "val_acc": correct / max(total, 1),
        "val_f1": val_f1,
    }


# ------------------------------------------------------------------ #
#  Full Training Pipeline
# ------------------------------------------------------------------ #
def train_model(
    model: nn.Module,
    dataloaders: Dict,
    config: Dict,
    device: torch.device,
    model_name: str = "vit",
    pretrained_path: Optional[str] = None,
) -> Dict:
    """
    Complete training pipeline with early stopping, scheduling, and logging.

    Args:
        model: Model to train.
        dataloaders: Dict with 'train', 'val', 'test' DataLoaders.
        config: Configuration dict.
        device: Device.
        model_name: Name prefix for saving.
        pretrained_path: Optional pretrained encoder path.

    Returns:
        Training history dict.
    """
    logger = setup_logger("training", config.get("logging", {}).get("log_dir", "logs/"))
    history_tracker = TrainingHistory(save_dir="logs/")
    hparam_logger = HyperparameterLogger(save_dir="logs/")

    train_cfg = config["training"]

    # Load pretrained encoder if available
    if pretrained_path and hasattr(model, "load_pretrained_encoder"):
        load_fn = getattr(model, "load_pretrained_encoder")
        load_fn(pretrained_path)
        logger.info(f"Loaded pretrained encoder from {pretrained_path}")

    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"]
    )

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(
        label_smoothing=train_cfg.get("label_smoothing", 0.1)
    )

    # Mixed precision
    scaler = None
    if train_cfg.get("mixed_precision", False) and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training enabled")

    # Early stopping
    es_config = train_cfg.get("early_stopping", {})
    patience = es_config.get("patience", 10)
    min_delta = es_config.get("min_delta", 0.001)
    best_metric = 0.0
    patience_counter = 0

    # Checkpoints
    ckpt_dir = Path(config.get("checkpoints", {}).get("save_dir", "checkpoints/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Log hyperparameters
    hparam_logger.log(config, f"{model_name}_training")

    epochs = train_cfg["epochs"]
    history: Dict[str, Any] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_f1": [], "learning_rate": [],
    }

    logger.info(f"Starting {model_name} training for {epochs} epochs")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device, scaler
        )

        # Validate
        val_metrics = validate(model, dataloaders["val"], criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["train_loss"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["train_acc"].append(train_metrics["train_acc"])
        history["val_acc"].append(val_metrics["val_acc"])
        history["val_f1"].append(val_metrics["val_f1"])
        history["learning_rate"].append(current_lr)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch:3d}/{epochs}] | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Train Acc: {train_metrics['train_acc']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_acc']:.4f} | "
            f"Val F1: {val_metrics['val_f1']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        monitor_metric = val_metrics["val_f1"]
        if monitor_metric > best_metric + min_delta:
            best_metric = monitor_metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "config": config,
            }, ckpt_dir / f"{model_name}_best.pth")
            logger.info(f"  → Saved best model (F1: {best_metric:.4f})")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        save_every = config.get("checkpoints", {}).get("save_every", 5)
        if epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, ckpt_dir / f"{model_name}_epoch{epoch}.pth")

        # Early stopping
        if es_config.get("enabled", True) and patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch} (patience: {patience})")
            break

    total_time = time.time() - start_time
    history["total_training_time_seconds"] = total_time
    logger.info(f"Training complete in {total_time:.1f}s. Best F1: {best_metric:.4f}")

    # Save training history
    history_tracker.history = history
    history_tracker.save(f"{model_name}_history.json")

    # Plot training curves
    plot_training_history(history, save_path=f"results/{model_name}_training_curves.png")

    return history


# ------------------------------------------------------------------ #
#  Self-Supervised Pretraining Phase
# ------------------------------------------------------------------ #
def run_ssl_pretraining(config: Dict, device: torch.device) -> str:
    """
    Run self-supervised pretraining and return checkpoint path.
    """
    ssl_cfg = config["ssl"]
    data_dir = config["data"]["processed_dir"]

    logger = setup_logger("ssl", "logs/")
    logger.info(f"Starting SSL pretraining with method: {ssl_cfg['method']}")

    if ssl_cfg["method"] == "mae":
        mae_cfg = ssl_cfg["mae"]

        # Create SSL dataset
        ssl_transform = get_ssl_transforms(config["data"]["image_size"])
        ssl_dataset = SSLImageDataset(data_dir, transform=ssl_transform)
        ssl_loader = torch.utils.data.DataLoader(
            ssl_dataset,
            batch_size=mae_cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        logger.info(f"SSL dataset size: {len(ssl_dataset)}")

        # Build MAE model
        vit_cfg = config["vit"]
        model = MaskedAutoencoder(
            image_size=vit_cfg["image_size"],
            patch_size=vit_cfg["patch_size"],
            encoder_dim=vit_cfg["embedding_dim"],
            encoder_heads=vit_cfg["num_heads"],
            encoder_depth=vit_cfg["num_layers"],
            decoder_dim=mae_cfg["decoder_embedding_dim"],
            decoder_heads=mae_cfg["decoder_num_heads"],
            decoder_depth=mae_cfg["decoder_num_layers"],
            mask_ratio=mae_cfg["mask_ratio"],
        )

        save_path = "checkpoints/mae_pretrained.pth"
        pretrain_mae(
            model=model,
            dataloader=ssl_loader,
            epochs=mae_cfg["epochs"],
            lr=mae_cfg["learning_rate"],
            weight_decay=mae_cfg["weight_decay"],
            warmup_epochs=mae_cfg["warmup_epochs"],
            device=device,
            save_path=save_path,
        )
        return save_path

    elif ssl_cfg["method"] == "contrastive":
        cont_cfg = ssl_cfg["contrastive"]
        from utils.dataset import ContrastiveDataset
        from utils.augmentation import get_contrastive_pair_transforms

        t1, t2 = get_contrastive_pair_transforms(config["data"]["image_size"])
        ssl_dataset = ContrastiveDataset(data_dir, transform1=t1, transform2=t2)
        ssl_loader = torch.utils.data.DataLoader(
            ssl_dataset,
            batch_size=cont_cfg["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        vit_cfg = config["vit"]
        model = ContrastiveModel(
            image_size=vit_cfg["image_size"],
            patch_size=vit_cfg["patch_size"],
            embed_dim=vit_cfg["embedding_dim"],
            num_heads=vit_cfg["num_heads"],
            depth=vit_cfg["num_layers"],
            projection_dim=cont_cfg["projection_dim"],
            temperature=cont_cfg["temperature"],
        )

        save_path = "checkpoints/contrastive_pretrained.pth"
        pretrain_contrastive(
            model=model,
            dataloader=ssl_loader,
            epochs=cont_cfg["epochs"],
            lr=cont_cfg["learning_rate"],
            device=device,
            save_path=save_path,
        )
        return save_path

    else:
        raise ValueError(f"Unknown SSL method: {ssl_cfg['method']}")


# ------------------------------------------------------------------ #
#  Main Entry Point
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="ViT Banking Fraud Detection - Training Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model", type=str, default="vit",
        choices=["vit", "hybrid", "cnn"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--skip-ssl", action="store_true",
        help="Skip self-supervised pretraining",
    )
    parser.add_argument(
        "--pretrained-encoder", type=str, default=None,
        help="Path to pretrained encoder checkpoint",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data directory from config",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.data_dir:
        config["data"]["processed_dir"] = args.data_dir

    # Reproducibility
    set_seed(config.get("seed", 42))
    device = get_device()

    # Create output directories
    for d in ["checkpoints", "results", "reports", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Explainable ViT for Financial Document Fraud Detection")
    print("  Domain: Banking & Finance | Area: Data Analytics + AI/ML")
    print("=" * 70)

    # ---- Phase 1: Data Preparation ----
    print("\n[PHASE 1] Data Preparation")
    data_dir = config["data"]["processed_dir"]
    class_names = config["data"]["classes"]

    dataloaders = create_dataloaders(
        data_dir=data_dir,
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        seed=config.get("seed", 42),
        class_names=class_names,
        balance_classes=True,
    )

    # ---- Phase 2: Self-Supervised Pretraining ----
    pretrained_path = args.pretrained_encoder

    if not args.skip_ssl and pretrained_path is None:
        print("\n[PHASE 2] Self-Supervised Pretraining")
        try:
            pretrained_path = run_ssl_pretraining(config, device)
        except Exception as e:
            print(f"  SSL pretraining skipped due to: {e}")
    else:
        print("\n[PHASE 2] SSL Pretraining: SKIPPED")

    # ---- Phase 3: Build Model ----
    print(f"\n[PHASE 3] Building {args.model.upper()} Model")

    if args.model == "vit":
        model = build_vit(config)
    elif args.model == "hybrid":
        model = build_hybrid(config)
    elif args.model == "cnn":
        model = build_cnn_baseline(config)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Phase 4: Training ----
    print(f"\n[PHASE 4] Fine-tuning {args.model.upper()}")
    history = train_model(
        model=model,
        dataloaders=dataloaders,
        config=config,
        device=device,
        model_name=args.model,
        pretrained_path=pretrained_path if args.model == "vit" else None,
    )

    # ---- Phase 5: Evaluation ----
    print("\n[PHASE 5] Final Evaluation")
    metrics_calc = MetricsCalculator(class_names, output_dir="results/")

    # Load best model
    best_ckpt = f"checkpoints/{args.model}_best.pth"
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best model from {best_ckpt}")

    metrics, inf_time = metrics_calc.evaluate_model(model, dataloaders["test"], device)
    model_size = metrics_calc.get_model_size(model)

    metrics.update(model_size)
    metrics["model_type"] = args.model
    metrics["ssl_pretrained"] = pretrained_path is not None

    export_results_json(metrics, f"results/{args.model}_evaluation.json")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  RESULTS SUMMARY - {args.model.upper()}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision_macro']:.4f}")
    print(f"  Recall:      {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    if metrics.get("roc_auc_macro"):
        print(f"  ROC-AUC:     {metrics['roc_auc_macro']:.4f}")
    print(f"  Inference:   {metrics['inference_time_per_sample_ms']:.2f} ms/sample")
    print(f"  Model Size:  {model_size['model_size_mb']:.2f} MB")
    print(f"  Parameters:  {model_size['total_parameters']:,}")
    print(f"{'='*50}\n")

    print("[DONE] Training pipeline complete!")
    print("  Checkpoints: checkpoints/")
    print("  Results:     results/")
    print("  Logs:        logs/")
    print("  Reports:     reports/")


if __name__ == "__main__":
    main()
