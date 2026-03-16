"""
Data augmentation pipelines for financial document images.
Includes standard augmentations and document-specific transforms.
"""

from typing import Tuple

import torchvision.transforms as T


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Training augmentation pipeline.
    
    Includes:
        - Random resized crop
        - Horizontal flip
        - Color jitter
        - Random rotation (slight)
        - Random erasing (simulates occlusion)
        - Normalization
    
    Args:
        image_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
    
    Returns:
        Composed transform pipeline.
    """
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        T.RandomRotation(degrees=5),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Validation/test transform pipeline (no augmentation).
    
    Args:
        image_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
    
    Returns:
        Composed transform pipeline.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_ssl_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Self-supervised learning augmentation pipeline.
    Stronger augmentations for contrastive learning.
    
    Args:
        image_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
    
    Returns:
        Composed transform pipeline.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_contrastive_pair_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
):
    """
    Returns two different augmentation views for contrastive learning.
    
    Returns:
        Tuple of two transform pipelines.
    """
    transform = get_ssl_transforms(image_size, mean, std)
    return transform, transform  # Two independent random augmentations
