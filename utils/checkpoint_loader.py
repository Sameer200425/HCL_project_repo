"""Utilities for loading ViT checkpoints across scripts.

Supports both modern checkpoints (with config metadata) and
legacy checkpoints where architecture must be inferred from tensor shapes.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

import torch

from models.vit_model import VisionTransformer


def resolve_checkpoint_path(checkpoint_candidates: list[str]) -> str:
    """Return first existing checkpoint path from candidates."""
    checkpoint_path = next((path for path in checkpoint_candidates if os.path.exists(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found. Tried: {checkpoint_candidates}")
    return checkpoint_path


def infer_vit_config_from_state_dict(
    state_dict: dict[str, Any],
    image_size: int,
    default_patch_size: int = 16,
    default_num_layers: int = 4,
) -> dict[str, Any]:
    """Infer ViT architecture from checkpoint tensors."""
    cls_key = "cls_token" if "cls_token" in state_dict else "cls"
    pos_key = "pos_embed" if "pos_embed" in state_dict else "pos_embedding"

    embed_dim = int(state_dict[cls_key].shape[-1])
    pos_tokens = int(state_dict[pos_key].shape[1])
    num_patches = max(pos_tokens - 1, 1)
    grid = int(round(math.sqrt(num_patches)))
    patch_size = image_size // grid if grid > 0 and image_size % grid == 0 else default_patch_size

    layer_ids: list[int] = []
    for key in state_dict.keys():
        match = re.match(r"encoder\.layers\.(\d+)\.attn\.qkv\.weight", key)
        if match:
            layer_ids.append(int(match.group(1)))
    num_layers = (max(layer_ids) + 1) if layer_ids else default_num_layers

    mlp_key = "encoder.layers.0.mlp.net.0.weight"
    mlp_dim = int(state_dict[mlp_key].shape[0]) if mlp_key in state_dict else embed_dim * 2

    if embed_dim >= 512:
        num_heads = 12
    elif embed_dim >= 256:
        num_heads = 8
    elif embed_dim >= 128:
        num_heads = 4
    else:
        num_heads = 2

    return {
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_dim": mlp_dim,
        "dropout": 0.1,
    }


def load_vit_from_checkpoint(
    checkpoint_candidates: list[str],
    device: torch.device,
    image_size: int,
    num_classes: int,
    default_config: dict[str, Any] | None = None,
) -> tuple[VisionTransformer, dict[str, Any], str, dict[str, Any]]:
    """Load a ViT model from checkpoint candidates.

    Returns: (model, checkpoint_dict, checkpoint_path, resolved_config)
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_candidates)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    base_default = {
        "patch_size": 16,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 4,
        "mlp_dim": 256,
        "dropout": 0.1,
    }
    if default_config:
        base_default.update(default_config)

    model_config = checkpoint.get("config", {})
    if not isinstance(model_config, dict) or not model_config:
        model_config = infer_vit_config_from_state_dict(
            state_dict,
            image_size=image_size,
            default_patch_size=int(base_default["patch_size"]),
            default_num_layers=int(base_default["num_layers"]),
        )

    resolved_config = {**base_default, **model_config}

    model = VisionTransformer(
        image_size=image_size,
        patch_size=int(resolved_config["patch_size"]),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=int(resolved_config["embed_dim"]),
        num_heads=int(resolved_config["num_heads"]),
        num_layers=int(resolved_config["num_layers"]),
        mlp_dim=int(resolved_config["mlp_dim"]),
        dropout=float(resolved_config["dropout"]),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint, checkpoint_path, resolved_config