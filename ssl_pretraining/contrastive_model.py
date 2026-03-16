"""
Contrastive Self-Supervised Learning for Vision Transformers.
Implements SimCLR-style contrastive pretraining for financial document images.

Learn representations by maximizing agreement between differently augmented
views of the same image using a contrastive loss (NT-Xent).
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# ------------------------------------------------------------------ #
#  Projection Head
# ------------------------------------------------------------------ #
class ProjectionHead(nn.Module):
    """MLP projection head mapping encoder features to contrastive space."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 2048, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ #
#  ViT Encoder (lightweight, for contrastive learning)
# ------------------------------------------------------------------ #
class ContrastiveViTEncoder(nn.Module):
    """
    Vision Transformer encoder for contrastive pretraining.
    Outputs CLS token representation.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels

        self.patch_embed = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            CLS token features (B, embed_dim)
        """
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token


# ------------------------------------------------------------------ #
#  NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy)
# ------------------------------------------------------------------ #
class NTXentLoss(nn.Module):
    """Contrastive loss for SimCLR-style learning."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) normalized projections from view 1.
            z2: (B, D) normalized projections from view 2.

        Returns:
            Scalar contrastive loss.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        B = z1.shape[0]

        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        # Cosine similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


# ------------------------------------------------------------------ #
#  Full Contrastive Model
# ------------------------------------------------------------------ #
class ContrastiveModel(nn.Module):
    """
    SimCLR-style contrastive learning with ViT backbone.
    
    Usage:
        model = ContrastiveModel()
        loss = model(view1, view2)
        encoder = model.get_encoder()
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.encoder = ContrastiveViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
        )
        self.projector = ProjectionHead(
            input_dim=embed_dim,
            output_dim=projection_dim,
        )
        self.criterion = NTXentLoss(temperature=temperature)

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two augmented views.

        Args:
            view1: (B, C, H, W) first augmented view.
            view2: (B, C, H, W) second augmented view.

        Returns:
            Scalar contrastive loss.
        """
        h1 = self.encoder(view1)
        h2 = self.encoder(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return self.criterion(z1, z2)

    def get_encoder(self) -> ContrastiveViTEncoder:
        """Return pretrained encoder for downstream fine-tuning."""
        return self.encoder


# ------------------------------------------------------------------ #
#  Contrastive Pretraining Loop
# ------------------------------------------------------------------ #
def pretrain_contrastive(
    model: ContrastiveModel,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/contrastive_pretrained.pth",
) -> Dict:
    """
    Contrastive pretraining loop.

    Args:
        model: ContrastiveModel instance.
        dataloader: DataLoader yielding (view1, view2) pairs.
        epochs: Training epochs.
        lr: Peak learning rate.
        weight_decay: Weight decay.
        warmup_epochs: Warmup epochs.
        device: Device.
        save_path: Checkpoint save path.

    Returns:
        Training history dictionary.
    """
    import os
    from pathlib import Path

    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"epoch": [], "loss": [], "lr": []}
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print("  Contrastive Self-Supervised Pretraining")
    print(f"  Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            view1, view2 = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            loss = model(view1, view2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["lr"].append(current_lr)

        print(
            f"  Epoch [{epoch:3d}/{epochs}] | "
            f"Loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "loss": best_loss,
            }, save_path)
            print(f"  → Saved best model (loss: {best_loss:.6f})")

    print(f"\n  Contrastive pretraining complete. Best loss: {best_loss:.6f}")
    return history
