"""
Vision Transformer (ViT) for Financial Document Classification.
Reference: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)

Architecture:
  1. Split image → 16×16 patches
  2. Linear patch embedding + positional encoding
  3. Transformer Encoder (12 layers, 12 heads)
  4. Classification head → {Genuine, Fraud, Tampered, Forged}
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ------------------------------------------------------------------ #
#  Patch Embedding
# ------------------------------------------------------------------ #
class PatchEmbedding(nn.Module):
    """Convert image into a sequence of patch embeddings."""

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
        patch_dim = patch_size * patch_size * in_channels

        self.projection = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


# ------------------------------------------------------------------ #
#  Multi-Head Self-Attention
# ------------------------------------------------------------------ #
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with attention weight storage for XAI."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Store attention weights for explainability
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Save attention weights for visualization
        self.attention_weights = attn.detach().clone()

        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj_dropout(self.proj(out))
        return out


# ------------------------------------------------------------------ #
#  Feed-Forward Network (MLP)
# ------------------------------------------------------------------ #
class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(
        self,
        embed_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ #
#  Transformer Encoder Block
# ------------------------------------------------------------------ #
class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder layer with pre-norm."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------------ #
#  Transformer Encoder (stack of blocks)
# ------------------------------------------------------------------ #
class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder blocks."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads, mlp_dim, dropout, attention_dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ------------------------------------------------------------------ #
#  Vision Transformer (ViT) - Full Model
# ------------------------------------------------------------------ #
class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer for financial document classification.
    
    Architecture:
        Image → Patches → Embedding → Positional Encoding
        → Transformer Encoder → CLS Token → Classification Head
    
    Output classes: Genuine / Fraud / Tampered / Forged
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 4,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, num_layers, mlp_dim, dropout, attention_dropout
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input images.
            return_features: If True, return CLS features instead of logits.

        Returns:
            (B, num_classes) logits, or (B, embed_dim) features.
        """
        B = x.shape[0]

        # Patch embedding
        patches = self.patch_embed(x)  # (B, N, D)

        # Prepend CLS token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls_tokens, patches], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer encoder
        x = self.encoder(x)
        x = self.norm(x)

        # CLS token output
        cls_output = x[:, 0]

        if return_features:
            return cls_output

        return self.head(cls_output)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS embeddings for downstream tasks like active learning."""
        return self.forward(x, return_features=True)

    def get_attention_maps(self) -> list:
        """
        Extract attention weights from all transformer layers.
        
        Returns:
            List of attention tensors (B, Heads, N, N) per layer.
        """
        attention_maps = []
        for layer in self.encoder.layers:
            assert isinstance(layer, TransformerEncoderBlock)
            if layer.attn.attention_weights is not None:
                attention_maps.append(layer.attn.attention_weights)
        return attention_maps

    def get_last_attention_map(self) -> Optional[torch.Tensor]:
        """Get attention map from the last encoder layer."""
        maps = self.get_attention_maps()
        return maps[-1] if maps else None

    def load_pretrained_encoder(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained encoder weights (from MAE or contrastive pretraining).

        Args:
            checkpoint_path: Path to pretrained checkpoint.
            strict: Whether to enforce exact key matching.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "encoder_state_dict" in checkpoint:
            state_dict = checkpoint["encoder_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Try to load compatible keys
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(
            f"[MODEL] Loaded {len(pretrained_dict)}/{len(model_dict)} "
            f"pretrained parameters from {checkpoint_path}"
        )


# ------------------------------------------------------------------ #
#  Factory function
# ------------------------------------------------------------------ #
def build_vit(config: dict) -> VisionTransformer:
    """
    Build ViT model from config dictionary.

    Args:
        config: Must contain 'vit' key with model hyperparameters.

    Returns:
        VisionTransformer instance.
    """
    vit_cfg = config["vit"]
    data_cfg = config.get("data", {})

    return VisionTransformer(
        image_size=vit_cfg.get("image_size", 224),
        patch_size=vit_cfg.get("patch_size", 16),
        in_channels=data_cfg.get("num_channels", 3),
        num_classes=vit_cfg.get("num_classes", 4),
        embed_dim=vit_cfg.get("embedding_dim", 768),
        num_heads=vit_cfg.get("num_heads", 12),
        num_layers=vit_cfg.get("num_layers", 12),
        mlp_dim=vit_cfg.get("mlp_dim", 3072),
        dropout=vit_cfg.get("dropout", 0.1),
        attention_dropout=vit_cfg.get("attention_dropout", 0.0),
    )
