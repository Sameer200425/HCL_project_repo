"""
Quick Evaluation + XAI Demo.
Loads the best checkpoint and generates explainability outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from PIL import Image

from utils.seed import set_seed, get_device
from utils.augmentation import get_val_transforms
from utils.checkpoint_loader import load_vit_from_checkpoint
from explainability.attention_visualization import (
    compute_attention_rollout,
    attention_to_heatmap,
    overlay_attention_on_image,
)
from explainability.gradcam import GradCAM, overlay_gradcam
from analytics.risk_scoring import FraudRiskScorer

CLASSES = ["genuine", "fraud", "tampered", "forged"]
IMAGE_SIZE = 224
CHECKPOINT_CANDIDATES = [
    "checkpoints/vit_best.pth",
    "checkpoints/best_model.pth",
]
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

set_seed(42)
device = get_device()


def build_model():
    """Build ViT using the architecture stored in the checkpoint."""
    model, ckpt, checkpoint_path, cfg = load_vit_from_checkpoint(
        checkpoint_candidates=CHECKPOINT_CANDIDATES,
        device=device,
        image_size=IMAGE_SIZE,
        num_classes=len(CLASSES),
        default_config={
            "patch_size": 32,
            "embed_dim": 64,
            "num_heads": 2,
            "num_layers": 2,
            "mlp_dim": 128,
            "dropout": 0.1,
        },
    )

    val_acc = ckpt.get("val_acc")
    val_acc_str = f"{float(val_acc):.4f}" if isinstance(val_acc, (int, float)) else "n/a"
    print(
        f"[OK] Model loaded from {checkpoint_path} "
        f"(epoch {ckpt.get('epoch', 'n/a')}, val_acc {val_acc_str})"
    )
    return model, int(cfg["patch_size"])


def predict(model, image_tensor: torch.Tensor):
    """Run prediction."""
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())
    return CLASSES[pred_idx], probs.cpu().numpy()


def run_xai_demo(model, image_path: str, patch_size: int):
    """Run XAI analysis on a single image."""
    print(f"\n{'='*50}")
    print(f"  XAI Analysis: {image_path}")
    print(f"{'='*50}")

    # Load & transform
    transform = get_val_transforms(IMAGE_SIZE)
    original = Image.open(image_path).convert("RGB")
    image_tensor: torch.Tensor = transform(original)  # type: ignore[assignment]
    original_np = np.array(original.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0

    # Prediction
    pred_class, probs = predict(model, image_tensor)
    print(f"\n  Prediction: {pred_class.upper()}")
    print(f"  Confidence: {probs.max():.4f}")
    for cls, p in zip(CLASSES, probs):
        bar = "|" * int(p * 40)
        print(f"    {cls:>10}: {p:.4f} {bar}")

    scorer = FraudRiskScorer()

    # Attention maps
    print("\n  Generating Attention Rollout...")
    attention_intensity = 0.5  # default if attention extraction fails
    heatmap = None
    try:
        # Forward pass to populate attention maps
        _ = model(image_tensor.unsqueeze(0).to(device))
        attention_maps = model.get_attention_maps()
        if attention_maps and len(attention_maps) > 0:
            rollout = compute_attention_rollout(attention_maps)
            if rollout is not None:
                # Convert 1D patch attention to 2D heatmap
                heatmap = attention_to_heatmap(rollout, IMAGE_SIZE, patch_size)
                overlaid = overlay_attention_on_image(original_np, heatmap)

                # Compute actual attention intensity from heatmap
                attention_intensity = scorer.compute_attention_intensity(heatmap)

                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(original_np)
                axes[0].set_title("Original Document")
                axes[0].axis("off")
                axes[1].imshow(heatmap, cmap="hot")
                axes[1].set_title("Attention Rollout")
                axes[1].axis("off")
                axes[2].imshow(overlaid)
                axes[2].set_title("Attention Overlay")
                axes[2].axis("off")
                save_path = RESULTS_DIR / "xai_attention.png"
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"    Saved: {save_path}")
            else:
                print("    Attention rollout returned None")
        else:
            print("    No attention maps available")
    except Exception as e:
        print(f"    Attention error: {e}")

    # Risk Score
    print("\n  Computing Risk Score...")
    class_probs = {cls: float(p) for cls, p in zip(CLASSES, probs)}
    conf_score = scorer.compute_confidence_score(class_probs, pred_class)

    # Compute anomaly score from prediction entropy (high entropy = more anomalous)
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy = np.log(len(CLASSES))
    anomaly_score = float(np.clip(entropy / (max_entropy + 1e-8), 0, 1))

    risk_score, risk_level = scorer.compute_risk_score(
        attention_intensity=attention_intensity,
        confidence_score=conf_score,
        anomaly_score=anomaly_score,
    )
    print(f"    Risk Score: {risk_score:.4f}")
    print(f"    Risk Level: {risk_level}")
    color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "orange", "CRITICAL": "red"}
    print(f"    Status: {'FLAGGED FOR REVIEW' if risk_level in ['HIGH', 'CRITICAL'] else 'PASSED'}")

    return pred_class, probs, risk_level


def main():
    model, patch_size = build_model()

    # Test on one image per class
    import glob
    image_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    for cls in CLASSES:
        images = []
        for ext in image_exts:
            images.extend(glob.glob(f"data/raw_images/{cls}/{ext}"))
        if images:
            run_xai_demo(model, images[0], patch_size)

    print(f"\n{'='*50}")
    print(f"  XAI Demo Complete!")
    print(f"  Results in: {RESULTS_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
