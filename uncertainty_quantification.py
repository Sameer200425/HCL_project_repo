"""
Uncertainty Quantification for ViT-Based Fraud Detection.
============================================================
Implements Bayesian deep learning via Monte Carlo (MC) Dropout.

WHY THIS MATTERS FOR BANKING:
  Banks cannot afford to blindly trust model predictions. A cheque worth
  ₹10 lakhs flagged as "genuine" with 60% confidence MUST be routed to
  human review. This module quantifies TWO types of uncertainty:

  1. Epistemic Uncertainty  — model doesn't have enough knowledge
     (unseen forgery technique, rare document type)
  2. Aleatoric Uncertainty  — inherent noise/ambiguity in the image
     (scan quality, lighting, physically degraded document)

USAGE:
  predictor = MCDropoutPredictor(model, n_passes=50, dropout_rate=0.1)
  result = predictor.predict(image_tensor)
  print(result.uncertainty_level)  # "HIGH" → route to human auditor

REFERENCE:
  Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
  Kendall & Gal, "What Uncertainties Do We Need?" (NeurIPS 2017)
============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Result Data Classes
# ------------------------------------------------------------------ #

@dataclass
class UncertaintyResult:
    """
    Full uncertainty-aware prediction result.

    Attributes:
        predicted_class   : Most likely class index.
        class_name        : Human-readable class label.
        mean_confidence   : Mean predicted probability for top class.
        epistemic_std     : Std of predictions across MC passes (model uncertainty).
        aleatoric_entropy : Entropy of mean prediction (data uncertainty).
        total_uncertainty : Combined uncertainty score in [0, 1].
        uncertainty_level : "LOW" / "MEDIUM" / "HIGH" / "CRITICAL".
        class_probabilities : Mean probability for each class.
        requires_human_review : Whether to flag for manual audit.
        mc_passes_used    : Number of forward passes used.
    """
    predicted_class: int
    class_name: str
    mean_confidence: float
    epistemic_std: float
    aleatoric_entropy: float
    total_uncertainty: float
    uncertainty_level: str
    class_probabilities: Dict[str, float]
    requires_human_review: bool
    mc_passes_used: int
    raw_passes: Optional[np.ndarray] = field(default=None, repr=False)


# ------------------------------------------------------------------ #
#  Enable Dropout at Inference
# ------------------------------------------------------------------ #

def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers for inference (MC Dropout mode).
    By default PyTorch disables dropout in eval() mode — this reverses that
    so dropout acts as a stochastic sampler.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


# ------------------------------------------------------------------ #
#  Monte Carlo Dropout Predictor
# ------------------------------------------------------------------ #

class MCDropoutPredictor:
    """
    Bayesian inference via Monte Carlo Dropout.

    Performs N stochastic forward passes with dropout enabled,
    then derives epistemic and aleatoric uncertainty from the
    distribution of predictions.
    """

    CLASS_NAMES = ["genuine", "fraud", "tampered", "forged"]

    # Thresholds for uncertainty_level classification
    UNCERTAINTY_THRESHOLDS = {
        "LOW": 0.15,
        "MEDIUM": 0.30,
        "HIGH": 0.50,
    }

    def __init__(
        self,
        model: nn.Module,
        n_passes: int = 50,
        dropout_rate: float = 0.1,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
        human_review_threshold: float = 0.35,
    ):
        """
        Args:
            model          : Trained ViT/Hybrid model.
            n_passes       : Number of stochastic forward passes (higher = more accurate).
            dropout_rate   : Injected dropout rate if model has no dropout layers.
            device         : Compute device.
            class_names    : Class label names.
            human_review_threshold : Total uncertainty above this → flag for review.
        """
        self.model = model
        self.n_passes = n_passes
        self.dropout_rate = dropout_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or self.CLASS_NAMES
        self.human_review_threshold = human_review_threshold
        self.model.to(self.device)

    def predict(
        self,
        image_tensor: torch.Tensor,
        return_raw_passes: bool = False,
    ) -> UncertaintyResult:
        """
        Run MC Dropout inference and return an uncertainty-aware result.

        Args:
            image_tensor     : (1, C, H, W) preprocessed image.
            return_raw_passes: Include all N softmax outputs in result.

        Returns:
            UncertaintyResult with full uncertainty analysis.
        """
        image_tensor = image_tensor.to(self.device)

        # Set model to eval first (bn, layernorm behave deterministically)
        self.model.eval()
        # Re-enable dropout only
        enable_mc_dropout(self.model)

        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for _ in range(self.n_passes):
                logits = self.model(image_tensor)          # (1, C)
                probs  = F.softmax(logits, dim=-1)         # (1, C)
                all_probs.append(probs.cpu().numpy())      # → list of (1, C)

        # Shape: (n_passes, 1, n_classes) → (n_passes, n_classes)
        raw = np.concatenate(all_probs, axis=0)            # (n_passes, n_classes)

        mean_probs  = raw.mean(axis=0)                     # (n_classes,)
        std_probs   = raw.std(axis=0)                      # (n_classes,)

        predicted_class = int(mean_probs.argmax())
        mean_confidence = float(mean_probs[predicted_class])

        # Epistemic uncertainty: mean std across classes (model knowledge gap)
        epistemic_std = float(std_probs.mean())

        # Aleatoric uncertainty: entropy of mean prediction (data ambiguity)
        eps = 1e-8
        aleatoric_entropy = float(
            -np.sum(mean_probs * np.log(mean_probs + eps))
        )
        # Normalize entropy to [0,1] using max possible entropy = log(n_classes)
        max_entropy = np.log(len(self.class_names))
        aleatoric_entropy_norm = aleatoric_entropy / max_entropy

        # Total uncertainty: weighted combination
        total_uncertainty = float(
            0.5 * epistemic_std * 4.0   # scale std to ~[0,1]
            + 0.5 * aleatoric_entropy_norm
        )
        total_uncertainty = min(total_uncertainty, 1.0)

        # Classify uncertainty level
        t = self.UNCERTAINTY_THRESHOLDS
        if total_uncertainty < t["LOW"]:
            level = "LOW"
        elif total_uncertainty < t["MEDIUM"]:
            level = "MEDIUM"
        elif total_uncertainty < t["HIGH"]:
            level = "HIGH"
        else:
            level = "CRITICAL"

        return UncertaintyResult(
            predicted_class=predicted_class,
            class_name=self.class_names[predicted_class],
            mean_confidence=mean_confidence,
            epistemic_std=epistemic_std,
            aleatoric_entropy=float(aleatoric_entropy),
            total_uncertainty=total_uncertainty,
            uncertainty_level=level,
            class_probabilities={
                name: float(mean_probs[i])
                for i, name in enumerate(self.class_names)
            },
            requires_human_review=(total_uncertainty >= self.human_review_threshold),
            mc_passes_used=self.n_passes,
            raw_passes=raw if return_raw_passes else None,
        )

    def batch_predict(
        self,
        images: torch.Tensor,
    ) -> List[UncertaintyResult]:
        """
        Run uncertainty-aware predictions on a batch.

        Args:
            images: (B, C, H, W)

        Returns:
            List of UncertaintyResult, one per image.
        """
        return [
            self.predict(images[i].unsqueeze(0))
            for i in range(images.shape[0])
        ]

    def uncertainty_summary(self, results: List[UncertaintyResult]) -> Dict:
        """
        Aggregate uncertainty statistics across a batch.

        Returns a summary for dashboard display.
        """
        total   = len(results)
        high    = sum(1 for r in results if r.uncertainty_level in ("HIGH", "CRITICAL"))
        reviews = sum(1 for r in results if r.requires_human_review)
        avg_unc = np.mean([r.total_uncertainty for r in results])

        level_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for r in results:
            level_counts[r.uncertainty_level] += 1

        return {
            "total_predictions": total,
            "requiring_human_review": reviews,
            "review_rate_pct": round(reviews / max(total, 1) * 100, 2),
            "avg_total_uncertainty": round(float(avg_unc), 4),
            "high_uncertainty_count": high,
            "uncertainty_breakdown": level_counts,
        }


# ------------------------------------------------------------------ #
#  Uncertainty Calibration (Temperature Scaling)
# ------------------------------------------------------------------ #

class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling.
    Adjusts model confidence to better match empirical accuracy.

    Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Find optimal temperature on validation set using NLL minimisation.

        Returns:
            Optimal temperature value.
        """
        self.to(device)
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                logits_list.append(logits.cpu())
                labels_list.append(labels)

        logits_all = torch.cat(logits_list)
        labels_all = torch.cat(labels_list)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval_closure)
        return float(self.temperature.item())


# ------------------------------------------------------------------ #
#  Expected Calibration Error (ECE)
# ------------------------------------------------------------------ #

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error — measures how well
    predicted confidence matches actual accuracy.

    Lower ECE → better calibrated model.
    Target for production: ECE < 0.05

    Args:
        probs  : (N,) max class probability per sample.
        labels : (N,) true class indices.

    Returns:
        ECE score as float.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (probs > low) & (probs <= high)
        if mask.sum() == 0:
            continue
        bin_acc = (labels[mask] == probs[mask].argmax(axis=-1)).mean() if probs.ndim > 1 else labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += np.abs(bin_conf - bin_acc) * mask.mean()

    return float(ece)


# ------------------------------------------------------------------ #
#  CLI Demo
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from models.vit_model import build_vit

    print("=" * 60)
    print("Uncertainty Quantification — MC Dropout Demo")
    print("=" * 60)

    device = torch.device("cpu")
    model  = build_vit(config={
        "vit": {
            "image_size": 224, "patch_size": 16, "embedding_dim": 768,
            "num_heads": 12, "num_layers": 12, "mlp_dim": 3072,
            "dropout": 0.1, "attention_dropout": 0.0, "num_classes": 4,
        }
    })

    predictor = MCDropoutPredictor(model, n_passes=30, device=device)

    dummy_image = torch.randn(1, 3, 224, 224)
    result = predictor.predict(dummy_image)

    print(f"\nPredicted Class  : {result.class_name} ({result.predicted_class})")
    print(f"Mean Confidence  : {result.mean_confidence:.4f}")
    print(f"Epistemic Std    : {result.epistemic_std:.4f}  (model uncertainty)")
    print(f"Aleatoric Entropy: {result.aleatoric_entropy:.4f}  (data uncertainty)")
    print(f"Total Uncertainty: {result.total_uncertainty:.4f}")
    print(f"Uncertainty Level: {result.uncertainty_level}")
    print(f"Requires Review  : {result.requires_human_review}")
    print(f"\nClass Probabilities:")
    for cls, prob in result.class_probabilities.items():
        print(f"  {cls:12s}: {prob:.4f}")
