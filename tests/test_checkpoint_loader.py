"""Tests for shared ViT checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from models.vit_model import VisionTransformer
from utils.checkpoint_loader import load_vit_from_checkpoint


def _build_test_vit(embed_dim: int = 64, num_heads: int = 2, num_layers: int = 2) -> VisionTransformer:
    return VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=embed_dim * 2,
        dropout=0.1,
    )


def test_load_vit_checkpoint_with_config(tmp_path: Path) -> None:
    model = _build_test_vit(embed_dim=64, num_heads=2, num_layers=2)
    ckpt_path = tmp_path / "with_config.pth"

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "config": {
                "patch_size": 16,
                "embed_dim": 64,
                "num_heads": 2,
                "num_layers": 2,
                "mlp_dim": 128,
                "dropout": 0.1,
            },
        },
        ckpt_path,
    )

    loaded_model, _, loaded_path, cfg = load_vit_from_checkpoint(
        checkpoint_candidates=[str(ckpt_path)],
        device=torch.device("cpu"),
        image_size=224,
        num_classes=4,
    )

    assert loaded_path == str(ckpt_path)
    assert int(cfg["embed_dim"]) == 64
    assert int(cfg["num_heads"]) == 2
    assert int(cfg["num_layers"]) == 2

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = loaded_model(x)
    assert out.shape == (1, 4)


def test_load_vit_checkpoint_without_config_infers_architecture(tmp_path: Path) -> None:
    # Use dimensions that match inference heuristic for stable load.
    model = _build_test_vit(embed_dim=128, num_heads=4, num_layers=2)
    ckpt_path = tmp_path / "legacy_no_config.pth"

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            # intentionally no `config`
        },
        ckpt_path,
    )

    loaded_model, _, _, cfg = load_vit_from_checkpoint(
        checkpoint_candidates=[str(ckpt_path)],
        device=torch.device("cpu"),
        image_size=224,
        num_classes=4,
    )

    assert int(cfg["embed_dim"]) == 128
    assert int(cfg["num_heads"]) == 4
    assert int(cfg["num_layers"]) == 2
    assert int(cfg["patch_size"]) == 16

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = loaded_model(x)
    assert out.shape == (2, 4)
