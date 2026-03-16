"""
Hybrid CNN + ViT Model for Financial Document Analysis.

Architecture:
  - CNN backbone (ResNet50) extracts low-level texture features
    (signature strokes, document edges, tampered pixel patterns)
  - ViT processes global document structure understanding
  - Combined features → Classification head
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange, repeat


# ------------------------------------------------------------------ #
#  CNN Feature Extractor
# ------------------------------------------------------------------ #
class CNNFeatureExtractor(nn.Module):
    """
    ResNet50 backbone for low-level feature extraction.
    Removes the final classification layer, outputs spatial feature maps.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        # Remove avgpool and fc
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.output_dim = 2048  # ResNet50 final conv output channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, 2048, H/32, W/32) feature maps
        """
        return self.features(x)


# ------------------------------------------------------------------ #
#  Feature Map to Patches Converter
# ------------------------------------------------------------------ #
class FeatureToPatches(nn.Module):
    """Convert CNN spatial feature maps into patch tokens for ViT."""

    def __init__(self, cnn_dim: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(cnn_dim),
            nn.Linear(cnn_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) CNN feature maps.
        Returns:
            (B, H*W, embed_dim) patch sequence.
        """
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.proj(x)


# ------------------------------------------------------------------ #
#  Transformer Block (reused from ViT)
# ------------------------------------------------------------------ #
class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
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
        # Store attention weights
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        self.attention_weights = weights.detach()
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------------ #
#  Hybrid CNN + ViT Model
# ------------------------------------------------------------------ #
class HybridCNNViT(nn.Module):
    """
    Hybrid model combining CNN and Vision Transformer.
    
    CNN provides:
      - Low-level texture features (signature strokes, edges)
      - Local pattern recognition
    
    ViT provides:
      - Global document structure understanding
      - Long-range dependencies between document regions
    
    Output: {Genuine, Fraud, Tampered, Forged}
    """

    def __init__(
        self,
        cnn_pretrained: bool = True,
        cnn_features: int = 2048,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 4,
    ):
        super().__init__()

        # CNN backbone
        self.cnn = CNNFeatureExtractor(pretrained=cnn_pretrained)

        # Project CNN features to ViT embedding dimension
        self.feature_proj = FeatureToPatches(cnn_features, embed_dim)

        # CLS token and positional embedding
        # For 224x224 input, ResNet50 produces 7x7=49 spatial positions
        max_patches = 49 + 1  # +1 for CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))

        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
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

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, 224, 224) input images.
            return_features: If True, return features instead of logits.

        Returns:
            (B, num_classes) logits, or (B, embed_dim) features.
        """
        B = x.shape[0]

        # Extract CNN features → patch tokens
        cnn_features = self.cnn(x)  # (B, 2048, 7, 7)
        patches = self.feature_proj(cnn_features)  # (B, 49, 768)

        # Prepend CLS token
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 50, 768)

        # Add positional embedding
        x = x + self.pos_embed[:, : x.shape[1], :]

        # Transformer blocks
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)

        # CLS token
        cls_output = x[:, 0]

        if return_features:
            return cls_output
        return self.head(cls_output)

    def get_attention_maps(self) -> list:
        """Extract attention maps from all transformer layers."""
        return [
            block.attention_weights
            for block in self.transformer
            if block.attention_weights is not None
        ]

    def freeze_cnn(self):
        """Freeze CNN backbone for transfer learning."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("[HYBRID] CNN backbone frozen")

    def unfreeze_cnn(self):
        """Unfreeze CNN backbone for end-to-end fine-tuning."""
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("[HYBRID] CNN backbone unfrozen")


# ------------------------------------------------------------------ #
#  CNN-only baseline model for comparison
# ------------------------------------------------------------------ #
class CNNBaseline(nn.Module):
    """
    ResNet50 baseline classifier for comparison study.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(  # type: ignore[assignment]
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------ #
def build_hybrid(config: dict) -> HybridCNNViT:
    """Build Hybrid CNN+ViT model from config."""
    h = config.get("hybrid", {})
    return HybridCNNViT(
        cnn_pretrained=True,
        cnn_features=h.get("cnn_features", 2048),
        embed_dim=h.get("vit_embedding_dim", 768),
        num_heads=h.get("vit_num_heads", 12),
        num_layers=h.get("vit_num_layers", 6),
        num_classes=h.get("num_classes", 4),
    )


def build_cnn_baseline(config: dict) -> CNNBaseline:
    """Build CNN baseline from config."""
    c = config.get("cnn", {})
    return CNNBaseline(
        pretrained=c.get("pretrained", True),
        num_classes=c.get("num_classes", 4),
    )
