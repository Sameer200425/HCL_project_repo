"""
Performance Metrics & Evaluation Analytics.

Computes comprehensive metrics for model evaluation:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, Precision-Recall curves
  - Confusion Matrix
  - Training/inference time benchmarks
  - Model size comparison
  - CSV export for data analytics
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ------------------------------------------------------------------ #
#  Metrics Calculator
# ------------------------------------------------------------------ #
class MetricsCalculator:
    """Compute and store comprehensive evaluation metrics."""

    def __init__(self, class_names: List[str], output_dir: str = "results/"):
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_probs: (N, C) prediction probabilities.

        Returns:
            Dictionary of all metrics.
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # Per-class metrics
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics["per_class"] = report

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # ROC-AUC (if probabilities available)
        if y_probs is not None and len(self.class_names) > 2:
            try:
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
                )
                metrics["roc_auc_weighted"] = float(
                    roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
                )
            except ValueError:
                metrics["roc_auc_macro"] = None
                metrics["roc_auc_weighted"] = None

        # False Positive Rate (per class)
        fps = {}
        for i, cls in enumerate(self.class_names):
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            tn = ((binary_pred == 0) & (binary_true == 0)).sum()
            fps[cls] = float(fp / (fp + tn + 1e-8))
        metrics["false_positive_rates"] = fps

        return metrics

    def evaluate_model(
        self,
        model,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Dict, float]:
        """
        Evaluate a model on a dataloader.

        Args:
            model: Trained model.
            dataloader: Test DataLoader.
            device: Device.

        Returns:
            (metrics_dict, inference_time_per_sample)
        """
        model = model.to(device).eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_time = 0.0
        num_samples = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                batch_size = images.shape[0]

                start = time.time()
                outputs = model(images)
                torch.cuda.synchronize() if device.type == "cuda" else None
                total_time += time.time() - start

                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                num_samples += batch_size

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        metrics = self.compute_all_metrics(y_true, y_pred, y_probs)
        inference_time = total_time / max(num_samples, 1)
        metrics["inference_time_per_sample_ms"] = inference_time * 1000
        metrics["total_samples"] = num_samples

        return metrics, inference_time

    def get_model_size(self, model) -> Dict[str, float]:
        """Compute model size in parameters and MB."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(size_mb, 2),
        }


# ------------------------------------------------------------------ #
#  Plotting Functions
# ------------------------------------------------------------------ #
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,  # type: ignore[arg-type]
        yticklabels=class_names,  # type: ignore[arg-type]
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        if binary_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_true, y_probs[:, i])
        auc = roc_auc_score(binary_true, y_probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot Precision-Recall curves for all classes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        binary_true = (y_true == i).astype(int)
        if binary_true.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary_true, y_probs[:, i])
        ax.plot(recall, precision, label=cls)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[str] = None,
) -> None:
    """Plot training loss and validation accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    if "train_loss" in history:
        axes[0].plot(history.get("epoch", range(len(history["train_loss"]))),
                     history["train_loss"], label="Train Loss", color="blue")
    if "val_loss" in history:
        axes[0].plot(history.get("epoch", range(len(history["val_loss"]))),
                     history["val_loss"], label="Val Loss", color="red")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    if "train_acc" in history:
        axes[1].plot(history.get("epoch", range(len(history["train_acc"]))),
                     history["train_acc"], label="Train Acc", color="blue")
    if "val_acc" in history:
        axes[1].plot(history.get("epoch", range(len(history["val_acc"]))),
                     history["val_acc"], label="Val Acc", color="red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fraud_trend(
    risk_scores: List[float],
    timestamps: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot fraud probability trend over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = timestamps or list(range(len(risk_scores)))
    scores_arr = np.array(risk_scores)
    ax.plot(x, risk_scores, color="red", marker="o", markersize=3)
    ax.fill_between(x, scores_arr, alpha=0.1, color="red")  # type: ignore[arg-type]
    ax.axhline(y=0.8, color="darkred", linestyle="--", alpha=0.7, label="Critical Threshold")
    ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.7, label="High Threshold")
    ax.set_xlabel("Document Index / Time")
    ax.set_ylabel("Fraud Risk Score")
    ax.set_title("Fraud Probability Trend")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Export Utilities
# ------------------------------------------------------------------ #
def export_metrics_csv(
    metrics: Dict,
    filepath: str = "results/metrics.csv",
) -> str:
    """Export metrics to CSV file."""
    import csv

    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)

    flat_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            flat_metrics[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float, str)):
                    flat_metrics[f"{key}_{sub_key}"] = sub_val

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in flat_metrics.items():
            writer.writerow([key, value])

    print(f"[METRICS] Exported to {filepath}")
    return filepath


def export_results_json(
    results: Dict,
    filepath: str = "results/evaluation_results.json",
) -> str:
    """Export full results to JSON."""
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"[RESULTS] Exported to {filepath}")
    return filepath
