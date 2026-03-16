"""
Masked Autoencoder (MAE) for Self-Supervised Pretraining.
Reference: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)

Learns document structure representations without labels by:
  1. Splitting images into patches
  2. Randomly masking 75% of patches
  3. Reconstructing the masked patches via a decoder
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ------------------------------------------------------------------ #
#  Patch Embedding
# ------------------------------------------------------------------ #
class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_size * patch_size * in_channels),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ------------------------------------------------------------------ #
#  Transformer Encoder Block
# ------------------------------------------------------------------ #
class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------------ #
#  MAE Encoder
# ------------------------------------------------------------------ #
class MAEEncoder(nn.Module):
    """
    Encoder that processes only visible (unmasked) patches.
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
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input images.
            mask: (B, N) boolean mask. True = keep, False = mask out.

        Returns:
            encoded: (B, L_visible+1, D) encoder output for visible patches + CLS.
            ids_restore: Indices to restore original patch order.
        """
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, N, D)
        N = patches.shape[1]

        # Add positional embedding (skip cls position)
        patches = patches + self.pos_embed[:, 1:, :]

        # Apply mask: keep only visible patches
        ids_keep = mask.nonzero(as_tuple=False)  # fallback
        # Efficient gather
        len_keep = mask.sum(dim=1)[0].int().item()
        ids_shuffle = torch.argsort(mask.float(), dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        patches = torch.gather(
            patches, 1, ids_keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        )

        # Prepend CLS token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        patches = torch.cat([cls_tokens, patches], dim=1)

        # Transformer blocks
        for block in self.blocks:
            patches = block(patches)
        patches = self.norm(patches)

        return patches, ids_restore


# ------------------------------------------------------------------ #
#  MAE Decoder
# ------------------------------------------------------------------ #
class MAEDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs masked patches.
    """

    def __init__(
        self,
        num_patches: int = 196,
        encoder_dim: int = 768,
        decoder_dim: int = 512,
        decoder_heads: int = 16,
        decoder_depth: int = 8,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(
            decoder_dim, patch_size * patch_size * in_channels
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L_visible+1, D) encoder output.
            ids_restore: (B, N) indices to restore patch order.

        Returns:
            pred: (B, N, patch_size^2 * C) reconstructed patches.
        """
        B, L, _ = x.shape
        x = self.decoder_embed(x)

        # Append mask tokens for masked positions
        N = ids_restore.shape[1]
        num_mask = N - (L - 1)  # subtract CLS
        mask_tokens = repeat(self.mask_token, "1 1 d -> b n d", b=B, n=num_mask)

        # Remove CLS, combine visible + mask, restore order
        x_no_cls = x[:, 1:, :]
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1])
        )

        # Re-add CLS
        x_full = torch.cat([x[:, :1, :], x_full], dim=1)

        # Add positional embedding
        x_full = x_full + self.decoder_pos_embed

        # Decoder blocks
        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)

        # Predict pixel values (remove CLS)
        pred = self.pred(x_full[:, 1:, :])
        return pred


# ------------------------------------------------------------------ #
#  Full MAE Model
# ------------------------------------------------------------------ #
class MaskedAutoencoder(nn.Module):
    """
    Complete Masked Autoencoder for self-supervised pretraining.
    
    Usage:
        model = MaskedAutoencoder()
        loss = model(images)           # training
        encoder = model.get_encoder()  # extract pretrained encoder
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        encoder_dim: int = 768,
        encoder_heads: int = 12,
        encoder_depth: int = 12,
        decoder_dim: int = 512,
        decoder_heads: int = 16,
        decoder_depth: int = 8,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.encoder = MAEEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_dim,
            num_heads=encoder_heads,
            depth=encoder_depth,
        )
        self.decoder = MAEDecoder(
            num_patches=self.num_patches,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_heads=decoder_heads,
            decoder_depth=decoder_depth,
            patch_size=patch_size,
            in_channels=in_channels,
        )

    def _create_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create random mask: True = keep, False = mask."""
        N = self.num_patches
        len_keep = int(N * (1 - self.mask_ratio))
        mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        for i in range(batch_size):
            idx = torch.randperm(N, device=device)[:len_keep]
            mask[i, idx] = True
        return mask

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images to patch targets for loss computation."""
        p = self.patch_size
        B, C, H, W = images.shape
        h, w = H // p, W // p
        patches = rearrange(
            images, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p
        )
        return patches

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: mask → encode → decode → MSE loss on masked patches.

        Args:
            images: (B, C, H, W) input images.

        Returns:
            Scalar reconstruction loss (MSE on masked patches only).
        """
        B = images.shape[0]
        device = images.device

        mask = self._create_mask(B, device)

        # Encode visible patches
        encoded, ids_restore = self.encoder(images, mask)

        # Decode all patches
        pred = self.decoder(encoded, ids_restore)

        # Compute loss only on masked patches
        target = self._patchify(images)
        loss_mask = ~mask  # True where masked (need reconstruction)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) per-patch MSE
        loss = (loss * loss_mask.float()).sum() / loss_mask.float().sum()

        return loss

    def get_encoder(self) -> MAEEncoder:
        """Return the pretrained encoder for fine-tuning."""
        return self.encoder


# ------------------------------------------------------------------ #
#  Pretraining Trainer
# ------------------------------------------------------------------ #
def pretrain_mae(
    model: MaskedAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    save_path: str = "checkpoints/mae_pretrained.pth",
) -> dict:
    """
    MAE pretraining loop.

    Args:
        model: MaskedAutoencoder instance.
        dataloader: DataLoader for unlabeled images.
        epochs: Number of training epochs.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: Linear warmup epochs.
        device: Training device.
        save_path: Path to save pretrained weights.

    Returns:
        Dictionary with training history.
    """
    import os
    from pathlib import Path

    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine schedule with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"epoch": [], "loss": [], "lr": []}
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print("  MAE Self-Supervised Pretraining")
    print(f"  Mask Ratio: {model.mask_ratio}")
    print(f"  Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)

            optimizer.zero_grad()
            loss = model(images)
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

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, save_path)
            print(f"  → Saved best model (loss: {best_loss:.6f})")

    print(f"\n  Pretraining complete. Best loss: {best_loss:.6f}")
    return history
