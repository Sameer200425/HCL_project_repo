"""
Knowledge Distillation: Large ViT (Teacher) → Small ViT (Student).
For deployment on edge devices (ATM systems, embedded hardware).

The student model learns to mimic the teacher's softened probability
distribution, enabling efficient inference with minimal accuracy loss.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit_model import VisionTransformer


# ------------------------------------------------------------------ #
#  Distillation Loss
# ------------------------------------------------------------------ #
class DistillationLoss(nn.Module):
    """
    Combined loss: hard label CE + soft label KL divergence.
    
    L = alpha * KL(softmax(teacher/T), softmax(student/T)) * T^2
      + (1 - alpha) * CE(student, labels)
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (B, C) student model output.
            teacher_logits: (B, C) teacher model output.
            labels: (B,) ground truth labels.

        Returns:
            Scalar combined loss.
        """
        # Soft labels from teacher
        soft_teacher = F.log_softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)

        # KL divergence (use log_softmax for numerical stability)
        kl_loss = F.kl_div(
            soft_student,
            soft_teacher.exp(),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Hard label cross-entropy
        hard_loss = self.ce_loss(student_logits, labels)

        return self.alpha * kl_loss + (1 - self.alpha) * hard_loss


# ------------------------------------------------------------------ #
#  Small ViT Student Model
# ------------------------------------------------------------------ #
def build_student_vit(config: dict) -> VisionTransformer:
    """
    Build a lightweight ViT student model for distillation.
    
    Args:
        config: Configuration with 'distillation.student_config' key.
    
    Returns:
        Smaller VisionTransformer suitable for edge deployment.
    """
    student_cfg = config.get("distillation", {}).get("student_config", {})
    vit_cfg = config.get("vit", {})

    return VisionTransformer(
        image_size=vit_cfg.get("image_size", 224),
        patch_size=vit_cfg.get("patch_size", 16),
        in_channels=3,
        num_classes=vit_cfg.get("num_classes", 4),
        embed_dim=student_cfg.get("embedding_dim", 384),
        num_heads=student_cfg.get("num_heads", 6),
        num_layers=student_cfg.get("num_layers", 6),
        mlp_dim=student_cfg.get("embedding_dim", 384) * 4,
        dropout=0.1,
    )


# ------------------------------------------------------------------ #
#  Distillation Trainer
# ------------------------------------------------------------------ #
def train_distillation(
    teacher: VisionTransformer,
    student: VisionTransformer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 3e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/distilled_student.pth",
) -> Dict:
    """
    Knowledge distillation training loop.

    Args:
        teacher: Pretrained large ViT (frozen).
        student: Small ViT to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Training epochs.
        lr: Learning rate.
        temperature: Distillation temperature.
        alpha: Balance between soft/hard loss.
        device: Device.
        save_path: Path to save student weights.

    Returns:
        Training history.
    """
    import os
    from pathlib import Path

    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    teacher = teacher.to(device).eval()
    student = student.to(device)

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    history = {"epoch": [], "train_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print("  Knowledge Distillation Training")
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")
    param_reduction = 1 - sum(p.numel() for p in student.parameters()) / sum(p.numel() for p in teacher.parameters())
    print(f"  Compression: {param_reduction:.1%} fewer parameters")
    print(f"  Temperature: {temperature} | Alpha: {alpha}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        student.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            loss = criterion(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)

        # ---- Validation ----
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = correct / max(total, 1)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch [{epoch:3d}/{epochs}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "val_acc": best_val_acc,
            }, save_path)
            print(f"  → Saved best student (acc: {best_val_acc:.4f})")

    print(f"\n  Distillation complete. Best val acc: {best_val_acc:.4f}")
    return history
