"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ViT.

Highlights the specific regions of a financial document that most
influenced the fraud/genuine classification decision.

Key uses:
  - Show tampered area on a cheque
  - Highlight edited text regions
  - Identify signature mismatch areas

Comparison with attention maps:
  - Attention maps show where the model "looked"
  - Grad-CAM shows what was "important" for the prediction
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Grad-CAM for ViT
# ------------------------------------------------------------------ #
class GradCAM:
    """
    Grad-CAM implementation for Vision Transformers.
    
    Computes gradient-weighted activation maps from a target layer,
    highlighting regions important for the predicted class.
    """

    def __init__(self, model, target_layer_name: str = "encoder.layers.11"):
        """
        Args:
            model: VisionTransformer model.
            target_layer_name: Name of the layer to compute Grad-CAM for.
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook_handles = []

        # Register hooks on target layer
        target_layer = self._get_layer(model, target_layer_name)
        self._hook_handles.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self._hook_handles.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    @staticmethod
    def _get_layer(model, layer_name: str):
        """Navigate to a nested layer by dot-separated name."""
        module = model
        for attr in layer_name.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)
        return module

    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass."""
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        image_size: int = 224,
        patch_size: int = 16,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.

        Args:
            image_tensor: (1, C, H, W) input image.
            target_class: Class to explain. None = predicted class.
            image_size: Image size.
            patch_size: Patch size.

        Returns:
            (heatmap, predicted_class, confidence)
            heatmap: (image_size, image_size) normalized heatmap.
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward
        output = self.model(image_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        confidence = float(probs[0, target_class].item())

        # Backward
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward(retain_graph=True)

        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations")

        gradients = self.gradients  # (1, N, D) or similar
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(dim=-1, keepdim=True)  # (1, N, 1)
        cam = (weights * activations).sum(dim=-1)  # (1, N)

        # ReLU
        cam = F.relu(cam)

        # Remove CLS token if present
        grid_size = image_size // patch_size
        num_patches = grid_size * grid_size

        cam = cam.squeeze(0)  # (N,)
        if cam.shape[0] > num_patches:
            cam = cam[1:]  # Remove CLS

        cam = cam[:num_patches]  # Ensure correct size

        # Reshape to grid
        cam = cam.cpu().numpy()
        cam = cam.reshape(grid_size, grid_size)

        # Upsample to image size
        cam = cv2.resize(cam, (image_size, image_size))

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class, confidence

    def cleanup(self):
        """Remove hooks."""
        for handle in self._hook_handles:
            handle.remove()


# ------------------------------------------------------------------ #
#  Visualization Utilities
# ------------------------------------------------------------------ #
def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image: (H, W, 3) RGB image, [0, 255].
        heatmap: (H, W) heatmap in [0, 1].
        alpha: Blending factor.

    Returns:
        Blended image.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)


def visualize_gradcam(
    model,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_names: list,
    target_layer: str = "encoder.layers.11",
    image_size: int = 224,
    patch_size: int = 16,
    target_class: Optional[int] = None,
    save_path: Optional[str] = None,
    title: str = "Grad-CAM – Financial Document Analysis",
) -> Tuple[np.ndarray, int, float]:
    """
    Complete Grad-CAM visualization pipeline.

    Args:
        model: VisionTransformer model.
        image_tensor: (1, C, H, W) preprocessed tensor.
        original_image: (H, W, 3) original image (RGB, 0-255).
        class_names: List of class names.
        target_layer: Layer for Grad-CAM.
        image_size: Image size.
        patch_size: Patch size.
        target_class: Class to explain (None = predicted).
        save_path: Save path for figure.
        title: Figure title.

    Returns:
        (overlay, predicted_class, confidence)
    """
    gradcam = GradCAM(model, target_layer)

    try:
        heatmap, pred_class, confidence = gradcam.generate(
            image_tensor, target_class, image_size, patch_size
        )
    finally:
        gradcam.cleanup()

    # Resize original
    if original_image.shape[:2] != (image_size, image_size):
        original_image = cv2.resize(original_image, (image_size, image_size))

    overlay = overlay_gradcam(original_image, heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Document", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    pred_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    axes[2].set_title(
        f"Prediction: {pred_label} ({confidence:.1%})",
        fontsize=12,
        color="red" if pred_label in ["fraud", "tampered", "forged"] else "green",
    )
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GRAD-CAM] Saved visualization to {save_path}")

    plt.close(fig)
    return overlay, pred_class, confidence


def compare_attention_vs_gradcam(
    model,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_names: list,
    target_layer: str = "encoder.layers.11",
    image_size: int = 224,
    patch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
) -> None:
    """
    Side-by-side comparison of Attention Maps vs Grad-CAM.
    Important for evaluating which XAI method is more suitable for banking audits.
    """
    from explainability.attention_visualization import (
        extract_attention_maps,
        compute_attention_rollout,
        attention_to_heatmap,
        overlay_attention_on_image,
    )

    # Resize original
    if original_image.shape[:2] != (image_size, image_size):
        original_image = cv2.resize(original_image, (image_size, image_size))

    # Attention rollout
    attn_maps = extract_attention_maps(model, image_tensor, device)
    attn_scores = compute_attention_rollout(attn_maps)
    attn_heatmap = attention_to_heatmap(attn_scores, image_size, patch_size)
    attn_overlay = overlay_attention_on_image(original_image, attn_heatmap)

    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    try:
        gc_heatmap, pred_class, confidence = gradcam.generate(
            image_tensor.to(device), None, image_size, patch_size
        )
    finally:
        gradcam.cleanup()
    gc_overlay = overlay_gradcam(original_image, gc_heatmap)

    pred_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Document")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(attn_heatmap, cmap="jet")
    axes[0, 1].set_title("Attention Rollout")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(attn_overlay)
    axes[0, 2].set_title("Attention Overlay")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title(f"Prediction: {pred_label} ({confidence:.1%})")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gc_heatmap, cmap="jet")
    axes[1, 1].set_title("Grad-CAM")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(gc_overlay)
    axes[1, 2].set_title("Grad-CAM Overlay")
    axes[1, 2].axis("off")

    fig.suptitle(
        "Attention Map vs Grad-CAM Comparison",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] Saved comparison to {save_path}")

    plt.close(fig)
