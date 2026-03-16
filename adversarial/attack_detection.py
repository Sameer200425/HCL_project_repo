"""
Adversarial Attack Detection & Robustness Testing.
============================================================
Simulates common document manipulation attacks:
  - Gaussian noise injection
  - Image blur
  - Pixel tampering (localized attack)
  - FGSM adversarial perturbation

Tests model robustness by measuring accuracy degradation
under each attack type and severity level.
============================================================
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Attack Implementations
# ------------------------------------------------------------------ #
class GaussianNoiseAttack:
    """Add Gaussian noise to document images."""

    def __init__(self, severity: float = 0.05):
        """
        Args:
            severity: Noise standard deviation (0 = none, 0.1 = heavy).
        """
        self.severity = severity

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(images) * self.severity
        return torch.clamp(images + noise, 0, 1)

    def __repr__(self):
        return f"GaussianNoise(severity={self.severity})"


class BlurAttack:
    """Apply Gaussian blur to simulate low-quality scans."""

    def __init__(self, kernel_size: int = 5):
        """
        Args:
            kernel_size: Blur kernel size (odd number).
        """
        self.kernel_size = kernel_size

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply blur to each image in the batch."""
        blurred = []
        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = cv2.GaussianBlur(
                img_np, (self.kernel_size, self.kernel_size), 0
            )
            blurred.append(torch.from_numpy(img_np).permute(2, 0, 1))
        return torch.stack(blurred).to(images.device)

    def __repr__(self):
        return f"Blur(kernel={self.kernel_size})"


class PixelTamperAttack:
    """Simulate localized pixel tampering (document alteration)."""

    def __init__(self, ratio: float = 0.05):
        """
        Args:
            ratio: Fraction of pixels to randomly alter.
        """
        self.ratio = ratio

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        mask = torch.rand(B, 1, H, W, device=images.device) < self.ratio
        random_pixels = torch.rand(B, C, H, W, device=images.device)
        tampered = torch.where(mask.expand_as(images), random_pixels, images)
        return tampered

    def __repr__(self):
        return f"PixelTamper(ratio={self.ratio})"


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    Creates perturbed images that fool the classifier.
    """

    def __init__(self, epsilon: float = 0.03):
        """
        Args:
            epsilon: Perturbation magnitude.
        """
        self.epsilon = epsilon

    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial examples.

        Args:
            model: Target model.
            images: (B, C, H, W) input images.
            labels: (B,) ground truth labels.

        Returns:
            Adversarial images.
        """
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # FGSM perturbation
        if images.grad is None:
            raise RuntimeError("No gradient computed for input images")
        perturbation = self.epsilon * images.grad.sign()
        adv_images = torch.clamp(images + perturbation, 0, 1)
        return adv_images.detach()

    def __repr__(self):
        return f"FGSM(epsilon={self.epsilon})"


# ------------------------------------------------------------------ #
#  Robustness Evaluator
# ------------------------------------------------------------------ #
class RobustnessEvaluator:
    """
    Evaluate model robustness against various attacks.
    
    Tests accuracy under:
      1. Gaussian noise at multiple severities
      2. Blur at multiple kernel sizes
      3. Pixel tampering at multiple ratios
      4. FGSM adversarial perturbation
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device).eval()
        self.device = device

    def evaluate_attack(
        self,
        dataloader: torch.utils.data.DataLoader,
        attack,
        is_fgsm: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model accuracy under a specific attack.

        Returns:
            Dict with accuracy and other stats.
        """
        correct = 0
        total = 0
        confidences = []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            if is_fgsm:
                self.model.train()  # Need gradients
                adv_images = attack(self.model, images, labels)
                self.model.eval()
            else:
                adv_images = attack(images)

            with torch.no_grad():
                outputs = self.model(adv_images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                confidences.extend(probs.max(1).values.cpu().numpy())

        accuracy = correct / max(total, 1)
        avg_confidence = float(np.mean(confidences))

        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "total_samples": total,
        }

    def run_full_robustness_test(
        self,
        dataloader: torch.utils.data.DataLoader,
        noise_severities: List[float] = [0.01, 0.05, 0.1],
        blur_kernels: List[int] = [3, 5, 7],
        tamper_ratios: List[float] = [0.01, 0.05, 0.1],
        fgsm_epsilons: List[float] = [0.01, 0.03, 0.05],
    ) -> Dict[str, list]:
        """
        Run comprehensive robustness test suite.

        Returns:
            Dictionary of results organized by attack type.
        """
        results = {"clean": [], "noise": [], "blur": [], "tamper": [], "fgsm": []}

        # Clean accuracy (baseline)
        print("  Testing clean accuracy...")
        clean_result = self.evaluate_attack(
            dataloader, lambda x: x  # Identity
        )
        clean_result["attack"] = "Clean"
        clean_result["param"] = "N/A"
        results["clean"].append(clean_result)
        print(f"    Clean Accuracy: {clean_result['accuracy']:.4f}")

        # Gaussian noise
        print("  Testing Gaussian noise robustness...")
        for severity in noise_severities:
            attack = GaussianNoiseAttack(severity)
            r = self.evaluate_attack(dataloader, attack)
            r["attack"] = str(attack)
            r["param"] = severity
            results["noise"].append(r)
            print(f"    Noise σ={severity}: Acc={r['accuracy']:.4f}")

        # Blur
        print("  Testing blur robustness...")
        for kernel in blur_kernels:
            attack = BlurAttack(kernel)
            r = self.evaluate_attack(dataloader, attack)
            r["attack"] = str(attack)
            r["param"] = kernel
            results["blur"].append(r)
            print(f"    Blur k={kernel}: Acc={r['accuracy']:.4f}")

        # Pixel tamper
        print("  Testing pixel tamper robustness...")
        for ratio in tamper_ratios:
            attack = PixelTamperAttack(ratio)
            r = self.evaluate_attack(dataloader, attack)
            r["attack"] = str(attack)
            r["param"] = ratio
            results["tamper"].append(r)
            print(f"    Tamper r={ratio}: Acc={r['accuracy']:.4f}")

        # FGSM
        print("  Testing FGSM robustness...")
        for eps in fgsm_epsilons:
            attack = FGSMAttack(eps)
            r = self.evaluate_attack(dataloader, attack, is_fgsm=True)
            r["attack"] = str(attack)
            r["param"] = eps
            results["fgsm"].append(r)
            print(f"    FGSM ε={eps}: Acc={r['accuracy']:.4f}")

        return results


def plot_robustness_results(
    results: Dict[str, list],
    save_path: Optional[str] = None,
) -> None:
    """Plot robustness test results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    clean_acc = results["clean"][0]["accuracy"] if results["clean"] else 0

    attack_types = [
        ("noise", "Gaussian Noise", "Noise Severity (σ)"),
        ("blur", "Gaussian Blur", "Kernel Size"),
        ("tamper", "Pixel Tamper", "Tamper Ratio"),
        ("fgsm", "FGSM Attack", "Epsilon (ε)"),
    ]

    for idx, (key, title, xlabel) in enumerate(attack_types):
        ax = axes[idx // 2][idx % 2]
        if results[key]:
            params = [r["param"] for r in results[key]]
            accs = [r["accuracy"] for r in results[key]]
            ax.plot(range(len(params)), accs, "o-", color="red", linewidth=2, markersize=8)
            ax.axhline(y=clean_acc, color="green", linestyle="--", alpha=0.7, label="Clean Acc")
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels([str(p) for p in params])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Accuracy")
            ax.set_title(title, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle("Model Robustness Against Adversarial Attacks", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
