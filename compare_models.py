"""
Model Comparison Study: CNN vs ViT vs ViT+SSL vs Hybrid
============================================================
Trains and evaluates all model variants, then generates a
comparative analysis with all metrics side by side.

Metrics compared:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC
  - False Positive Rate
  - Inference Time (ms)
  - Model Size (MB)
  - Total Parameters

============================================================
Usage:
  python compare_models.py
  python compare_models.py --config config.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from utils.seed import set_seed, get_device
from utils.dataset import create_dataloaders
from models.vit_model import build_vit
from models.hybrid_model import build_hybrid, build_cnn_baseline
from analytics.performance_metrics import MetricsCalculator, export_results_json


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected dict from config, got {type(config)}")
    return config


def evaluate_model(
    model,
    dataloaders: Dict,
    class_names: List[str],
    device: torch.device,
    model_name: str,
) -> Dict:
    """Evaluate a single model variant."""
    calc = MetricsCalculator(class_names, output_dir="results/")
    metrics, inf_time = calc.evaluate_model(model, dataloaders["test"], device)
    model_size = calc.get_model_size(model)
    metrics.update(model_size)
    metrics["model_name"] = model_name
    return metrics


def plot_comparison(results: List[Dict], save_dir: str = "results/") -> None:
    """Generate comparison plots."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model_names = [r["model_name"] for r in results]
    
    # Bar chart comparison
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision (Macro)"),
        ("recall_macro", "Recall (Macro)"),
        ("f1_macro", "F1-Score (Macro)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for idx, (key, label) in enumerate(metric_keys):
        values = [r.get(key, 0) for r in results]
        bars = axes[idx].bar(model_names, values, color=colors[:len(model_names)])
        axes[idx].set_title(label, fontsize=13, fontweight="bold")
        axes[idx].set_ylim(0, 1.05)
        axes[idx].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=10,
            )

    fig.suptitle("Model Comparison — Classification Metrics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison_classification_metrics.png"), dpi=150)
    plt.close(fig)

    # Inference time & model size comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Inference time
    inf_times = [r.get("inference_time_per_sample_ms", 0) for r in results]
    bars = axes[0].bar(model_names, inf_times, color="#FF5722")
    axes[0].set_title("Inference Time (ms/sample)", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, inf_times):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", fontsize=10)

    # Model size
    sizes = [r.get("model_size_mb", 0) for r in results]
    bars = axes[1].bar(model_names, sizes, color="#9C27B0")
    axes[1].set_title("Model Size (MB)", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, sizes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", fontsize=10)

    # Parameters
    params = [r.get("total_parameters", 0) / 1e6 for r in results]
    bars = axes[2].bar(model_names, params, color="#009688")
    axes[2].set_title("Parameters (Millions)", fontsize=12, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, params):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{val:.1f}M", ha="center", fontsize=10)

    fig.suptitle("Model Comparison — Efficiency Metrics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison_efficiency_metrics.png"), dpi=150)
    plt.close(fig)

    # ROC-AUC comparison
    auc_values = [r.get("roc_auc_macro", 0) or 0 for r in results]
    if any(v > 0 for v in auc_values):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(model_names, auc_values, color="#3F51B5")
        ax.set_title("ROC-AUC (Macro)", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, auc_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "comparison_roc_auc.png"), dpi=150)
        plt.close(fig)


def generate_comparison_table(results: List[Dict], save_dir: str = "results/") -> None:
    """Generate comparison table as CSV and print to console."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": f"{r.get('accuracy', 0):.4f}",
            "Precision": f"{r.get('precision_macro', 0):.4f}",
            "Recall": f"{r.get('recall_macro', 0):.4f}",
            "F1-Score": f"{r.get('f1_macro', 0):.4f}",
            "ROC-AUC": f"{r.get('roc_auc_macro', 'N/A')}",
            "Inference (ms)": f"{r.get('inference_time_per_sample_ms', 0):.2f}",
            "Size (MB)": f"{r.get('model_size_mb', 0):.2f}",
            "Parameters": f"{r.get('total_parameters', 0):,}",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*90}")
    print("  MODEL COMPARISON RESULTS")
    print(f"{'='*90}")
    print(df.to_string(index=False))
    print(f"{'='*90}")
    print(f"\n  Saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare all model architectures")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()

    class_names = config["data"]["classes"]
    Path("results").mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON STUDY")
    print("  CNN vs ViT vs ViT+SSL vs Hybrid CNN+ViT")
    print("=" * 70)

    # Data
    dataloaders = create_dataloaders(
        data_dir=config["data"]["processed_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        seed=config.get("seed", 42),
        class_names=class_names,
    )

    results = []

    # ---- 1. CNN Baseline ----
    print("\n[1/4] Evaluating CNN (ResNet50)...")
    cnn_ckpt = "checkpoints/cnn_best.pth"
    if os.path.exists(cnn_ckpt):
        cnn_model = build_cnn_baseline(config)
        ckpt = torch.load(cnn_ckpt, map_location=device)
        cnn_model.load_state_dict(ckpt["model_state_dict"])
        cnn_model = cnn_model.to(device)
        metrics = evaluate_model(cnn_model, dataloaders, class_names, device, "CNN (ResNet50)")
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    else:
        print(f"  Skipped — checkpoint not found: {cnn_ckpt}")

    # ---- 2. ViT (from scratch) ----
    print("\n[2/4] Evaluating ViT (from scratch)...")
    vit_ckpt = "checkpoints/vit_best.pth"
    if os.path.exists(vit_ckpt):
        vit_model = build_vit(config)
        ckpt = torch.load(vit_ckpt, map_location=device)
        vit_model.load_state_dict(ckpt["model_state_dict"])
        vit_model = vit_model.to(device)
        metrics = evaluate_model(vit_model, dataloaders, class_names, device, "ViT")
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    else:
        print(f"  Skipped — checkpoint not found: {vit_ckpt}")

    # ---- 3. ViT + SSL ----
    print("\n[3/4] Evaluating ViT + SSL...")
    vit_ssl_ckpt = "checkpoints/vit_ssl_best.pth"
    if os.path.exists(vit_ssl_ckpt):
        vit_ssl_model = build_vit(config)
        ckpt = torch.load(vit_ssl_ckpt, map_location=device)
        vit_ssl_model.load_state_dict(ckpt["model_state_dict"])
        vit_ssl_model = vit_ssl_model.to(device)
        metrics = evaluate_model(vit_ssl_model, dataloaders, class_names, device, "ViT + SSL")
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    else:
        print(f"  Skipped — checkpoint not found: {vit_ssl_ckpt}")

    # ---- 4. Hybrid CNN + ViT ----
    print("\n[4/4] Evaluating Hybrid CNN+ViT...")
    hybrid_ckpt = "checkpoints/hybrid_best.pth"
    if os.path.exists(hybrid_ckpt):
        hybrid_model = build_hybrid(config)
        ckpt = torch.load(hybrid_ckpt, map_location=device)
        hybrid_model.load_state_dict(ckpt["model_state_dict"])
        hybrid_model = hybrid_model.to(device)
        metrics = evaluate_model(hybrid_model, dataloaders, class_names, device, "Hybrid CNN+ViT")
        results.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    else:
        print(f"  Skipped — checkpoint not found: {hybrid_ckpt}")

    # ---- Generate comparison ----
    if results:
        plot_comparison(results)
        generate_comparison_table(results)
        export_results_json(
            {"models": results},
            "results/full_comparison.json",
        )
    else:
        print("\n  No trained models found. Train models first using main_training.py")

    print("\n[DONE] Comparison study complete!")


if __name__ == "__main__":
    main()
