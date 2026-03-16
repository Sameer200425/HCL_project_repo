"""
Attention Map Visualization for Vision Transformer.
Shows which regions of a financial document (cheque, signature, ID card)
influenced the model's fraud/genuine decision.

This is critical for bank audit-readiness and regulatory compliance.
"""

from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def extract_attention_maps(
    model,
    image_tensor: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """
    Run a forward pass and extract attention maps from all transformer layers.

    Args:
        model: VisionTransformer or HybridCNNViT with `get_attention_maps()`.
        image_tensor: (1, C, H, W) preprocessed image tensor.
        device: Computation device.

    Returns:
        List of attention tensors, each (1, Heads, N, N).
    """
    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        _ = model(image_tensor)

    return model.get_attention_maps()


def compute_attention_rollout(
    attention_maps: List[torch.Tensor],
    head: str = "mean",
    discard_ratio: float = 0.1,
) -> np.ndarray:
    """
    Compute attention rollout across all layers.
    Multiplies attention matrices layer-by-layer for a global view.

    Args:
        attention_maps: List of (1, Heads, N, N) attention tensors.
        head: "mean" to average all heads, or int for specific head.
        discard_ratio: Fraction of lowest attention values to discard.

    Returns:
        (N-1,) attention scores for each patch (excluding CLS).
    """
    result = None

    for attn in attention_maps:
        attn = attn.squeeze(0)  # (Heads, N, N)

        if head == "mean":
            attn = attn.mean(dim=0)  # (N, N)
        else:
            attn = attn[int(head)]

        # Add identity (residual connection)
        attn = attn.cpu().numpy()
        attn = attn + np.eye(attn.shape[0])

        # Discard low attention
        if discard_ratio > 0:
            flat = attn.flatten()
            threshold = np.percentile(flat, discard_ratio * 100)
            attn[attn < threshold] = 0

        # Normalize rows
        attn = attn / attn.sum(axis=-1, keepdims=True)

        if result is None:
            result = attn
        else:
            result = result @ attn

    # Attention from CLS token to all patches
    assert result is not None, "No attention maps provided"
    cls_attention = result[0, 1:]  # Exclude CLS-to-CLS
    cls_attention = cls_attention / cls_attention.max()  # Normalize to [0, 1]
    return cls_attention


def attention_to_heatmap(
    attention_scores: np.ndarray,
    image_size: int = 224,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Reshape 1D patch attention scores into a 2D heatmap.

    Args:
        attention_scores: (num_patches,) attention values.
        image_size: Original image size.
        patch_size: Patch size used by ViT.

    Returns:
        (image_size, image_size) heatmap.
    """
    grid_size = image_size // patch_size
    heatmap = attention_scores.reshape(grid_size, grid_size)
    heatmap = cv2.resize(heatmap, (image_size, image_size))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def overlay_attention_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay attention heatmap on the original image.

    Args:
        image: (H, W, 3) original image in RGB, [0, 255].
        heatmap: (H, W) attention heatmap in [0, 1].
        alpha: Blend factor (0 = only image, 1 = only heatmap).
        colormap: OpenCV colormap.

    Returns:
        (H, W, 3) blended image in RGB.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Ensure same size
    if image.shape[:2] != colored.shape[:2]:
        colored = cv2.resize(colored, (image.shape[1], image.shape[0]))

    blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return blended


def visualize_attention(
    model,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    image_size: int = 224,
    patch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
    title: str = "Attention Map – Document Fraud Analysis",
) -> np.ndarray:
    """
    Complete attention visualization pipeline.

    Args:
        model: VisionTransformer with get_attention_maps().
        image_tensor: (1, C, H, W) preprocessed tensor.
        original_image: (H, W, 3) original image (RGB, 0-255).
        image_size: Image size.
        patch_size: Patch size.
        device: Device.
        save_path: If set, save figure to this path.
        title: Plot title.

    Returns:
        Blended overlay image.
    """
    # Extract attention
    attn_maps = extract_attention_maps(model, image_tensor, device)
    attn_scores = compute_attention_rollout(attn_maps)
    heatmap = attention_to_heatmap(attn_scores, image_size, patch_size)

    # Resize original image
    if original_image.shape[:2] != (image_size, image_size):
        original_image = cv2.resize(original_image, (image_size, image_size))

    overlay = overlay_attention_on_image(original_image, heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Document", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Attention Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Attention Overlay", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ATTN] Saved attention visualization to {save_path}")

    plt.close(fig)
    return overlay


def visualize_per_head_attention(
    model,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    layer_idx: int = -1,
    image_size: int = 224,
    patch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize attention from each head in a specific layer.
    Useful for understanding which heads focus on different document regions.
    """
    attn_maps = extract_attention_maps(model, image_tensor, device)
    attn = attn_maps[layer_idx].squeeze(0).cpu().numpy()  # (Heads, N, N)
    num_heads = attn.shape[0]
    grid_size = image_size // patch_size

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for h in range(num_heads):
        head_attn = attn[h, 0, 1:]  # CLS to patches
        head_attn = head_attn / (head_attn.max() + 1e-8)
        heatmap = head_attn.reshape(grid_size, grid_size)
        heatmap = cv2.resize(heatmap, (image_size, image_size))
        axes[h].imshow(heatmap, cmap="viridis")
        axes[h].set_title(f"Head {h}", fontsize=10)
        axes[h].axis("off")

    for i in range(num_heads, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Per-Head Attention (Layer {layer_idx})", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
