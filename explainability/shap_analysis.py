"""
SHAP (SHapley Additive exPlanations) Analysis for ViT.

Provides feature importance scores for fraud detection decisions.
Shows which input features (patch regions) contribute most to the
model's confidence in a particular classification.

Essential for banking compliance and audit reporting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ------------------------------------------------------------------ #
#  SHAP Explainer Wrapper
# ------------------------------------------------------------------ #
class ViTSHAPExplainer:
    """
    SHAP-based explanation for Vision Transformer predictions.
    
    Uses DeepExplainer or GradientExplainer to compute Shapley values
    for image patches, providing feature-level importance scores.
    """

    def __init__(
        self,
        model,
        background_data: torch.Tensor,
        class_names: Optional[List[str]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            model: Trained VisionTransformer model.
            background_data: (N, C, H, W) reference images for SHAP.
            class_names: List of class names.
            device: Computation device.
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )

        self.model = model.to(device).eval()
        self.background_data = background_data.to(device)
        self.class_names = class_names or ["genuine", "fraud", "tampered", "forged"]
        self.device = device

        # Create SHAP explainer
        self.explainer = shap.GradientExplainer(
            self.model,
            self.background_data,
        )

    def explain(
        self,
        image_tensor: torch.Tensor,
        num_samples: int = 100,
    ) -> Any:
        """
        Compute SHAP values for an image.

        Args:
            image_tensor: (1, C, H, W) image to explain.
            num_samples: Number of samples for SHAP estimation.

        Returns:
            SHAP values (list of ndarray or ndarray).
        """
        image_tensor = image_tensor.to(self.device)
        shap_values = self.explainer.shap_values(
            image_tensor,
            nsamples=num_samples,
        )
        return shap_values

    def get_feature_importance(
        self,
        shap_values: np.ndarray,
        target_class: int,
        patch_size: int = 16,
        image_size: int = 224,
    ) -> np.ndarray:
        """
        Aggregate SHAP values into per-patch importance scores.

        Args:
            shap_values: SHAP values from explain().
            target_class: Class index to analyze.
            patch_size: ViT patch size.
            image_size: Image size.

        Returns:
            (grid_size, grid_size) importance scores per patch.
        """
        if isinstance(shap_values, list):
            sv = shap_values[target_class][0]  # (C, H, W)
        else:
            sv = shap_values[0, :, :, :, target_class] if shap_values.ndim == 5 else shap_values[0]

        # Aggregate across channels
        importance = np.abs(sv).mean(axis=0)  # (H, W)

        # Average within each patch
        grid_size = image_size // patch_size
        patch_importance = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                patch = importance[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patch_importance[i, j] = patch.mean()

        # Normalize
        patch_importance = patch_importance / (patch_importance.max() + 1e-8)
        return patch_importance


def visualize_shap(
    model,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    background_data: torch.Tensor,
    class_names: List[str],
    target_class: Optional[int] = None,
    num_samples: int = 100,
    image_size: int = 224,
    patch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
    max_display: int = 20,
) -> Dict:
    """
    Complete SHAP visualization pipeline.

    Args:
        model: Trained VisionTransformer.
        image_tensor: (1, C, H, W) image.
        original_image: (H, W, 3) RGB image.
        background_data: (N, C, H, W) reference images.
        class_names: Class names.
        target_class: Class to explain (None = predicted).
        num_samples: SHAP samples.
        image_size: Image size.
        patch_size: Patch size.
        device: Device.
        save_path: Save path.
        max_display: Max patches to display.

    Returns:
        Dict with SHAP analysis results.
    """
    if not SHAP_AVAILABLE:
        print("[SHAP] SHAP not installed. Skipping analysis.")
        return {}

    # Get prediction
    model = model.to(device).eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    if target_class is None:
        target_class = int(pred_class)

    # Compute SHAP values
    explainer = ViTSHAPExplainer(model, background_data, class_names, device)
    shap_values = explainer.explain(image_tensor, num_samples)
    importance = explainer.get_feature_importance(
        shap_values, target_class, patch_size, image_size
    )

    # Resize for overlay
    import cv2
    if original_image.shape[:2] != (image_size, image_size):
        original_image = cv2.resize(original_image, (image_size, image_size))

    importance_resized = cv2.resize(importance, (image_size, image_size))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Document", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(importance_resized, cmap="hot")
    axes[1].set_title(f"SHAP Importance ({class_names[target_class]})", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    # Confidence bar chart
    probs_np = probs[0].cpu().numpy()
    colors = ["green" if i == pred_class else "steelblue" for i in range(len(class_names))]
    axes[2].barh(class_names, probs_np, color=colors)
    axes[2].set_xlabel("Confidence", fontsize=11)
    axes[2].set_title("Decision Confidence", fontsize=12)
    axes[2].set_xlim(0, 1)

    fig.suptitle(
        f"SHAP Analysis | Prediction: {class_names[pred_class]} ({confidence:.1%})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Saved visualization to {save_path}")

    plt.close(fig)

    return {
        "predicted_class": pred_class,
        "predicted_label": class_names[pred_class],
        "confidence": confidence,
        "class_probabilities": {
            class_names[i]: float(probs_np[i]) for i in range(len(class_names))
        },
        "patch_importance": importance,
    }
