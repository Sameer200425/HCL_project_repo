"""
Dataset classes for financial document fraud detection.
Supports folder-based datasets with train/val/test splitting.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.augmentation import get_train_transforms, get_val_transforms


# ------------------------------------------------------------------ #
#  Core Dataset
# ------------------------------------------------------------------ #
class FinancialDocumentDataset(Dataset):
    """
    Dataset for financial document images organized in class folders:
        data/processed/genuine/
        data/processed/fraud/
        data/processed/tampered/
        data/processed/forged/
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir: Path to the image root directory.
            transform: Torchvision transforms to apply.
            class_names: Ordered list of class names.
                         If None, inferred from subdirectory names.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Discover classes from folder names
        if class_names is None:
            class_names = sorted(
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )
        self.class_names = class_names
        self.class_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(class_names)
        }

        # Collect (path, label) pairs
        self.samples: List[Tuple[str, int]] = []
        for cls_name in class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if Path(fname).suffix.lower() in self.EXTENSIONS:
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name])
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        assert isinstance(image, torch.Tensor)
        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count per class."""
        dist: Dict[str, int] = {c: 0 for c in self.class_names}
        for _, label in self.samples:
            dist[self.class_names[label]] += 1
        return dist


# ------------------------------------------------------------------ #
#  SSL Dataset (No labels required)
# ------------------------------------------------------------------ #
class SSLImageDataset(Dataset):
    """Unlabeled dataset for self-supervised pretraining."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths: List[str] = []

        for root, _, files in os.walk(root_dir):
            for fname in files:
                if Path(fname).suffix.lower() in self.EXTENSIONS:
                    self.image_paths.append(os.path.join(root, fname))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        assert isinstance(image, torch.Tensor)
        return image


class ContrastiveDataset(Dataset):
    """Dataset returning two augmented views of the same image."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        root_dir: str,
        transform1: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.image_paths: List[str] = []

        for root, _, files in os.walk(root_dir):
            for fname in files:
                if Path(fname).suffix.lower() in self.EXTENSIONS:
                    self.image_paths.append(os.path.join(root, fname))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        view1: torch.Tensor = self.transform1(image) if self.transform1 else image  # type: ignore[assignment]
        view2: torch.Tensor = self.transform2(image) if self.transform2 else image  # type: ignore[assignment]
        return view1, view2


# ------------------------------------------------------------------ #
#  Data Splitting & Loader Utilities
# ------------------------------------------------------------------ #
def split_dataset(
    dataset: FinancialDocumentDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split into train / val / test indices.

    Args:
        dataset: The full dataset.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )
    temp_labels = [labels[i] for i in temp_idx]
    relative_val = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_val,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def get_weighted_sampler(
    dataset: FinancialDocumentDataset,
    indices: List[int],
) -> WeightedRandomSampler:
    """
    Create a weighted sampler to handle class imbalance.

    Args:
        dataset: The dataset.
        indices: Subset indices.

    Returns:
        WeightedRandomSampler for the DataLoader.
    """
    labels = [dataset.samples[i][1] for i in indices]
    class_counts = np.bincount(labels, minlength=len(dataset.class_names))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
    balance_classes: bool = True,
) -> Dict[str, DataLoader]:
    """
    End-to-end: build dataset → split → create train/val/test loaders.

    Args:
        data_dir: Root directory with class subfolders.
        image_size: Target image size.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        seed: Random seed.
        class_names: Ordered class names.
        balance_classes: Whether to use weighted sampling.

    Returns:
        Dict with keys "train", "val", "test" mapping to DataLoaders.
    """
    full_dataset = FinancialDocumentDataset(
        root_dir=data_dir,
        transform=None,  # transforms applied per-split below
        class_names=class_names,
    )

    train_idx, val_idx, test_idx = split_dataset(full_dataset, seed=seed)

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_ds = _SubsetWithTransform(full_dataset, train_idx, train_transform)
    val_ds = _SubsetWithTransform(full_dataset, val_idx, val_transform)
    test_ds = _SubsetWithTransform(full_dataset, test_idx, val_transform)

    sampler = None
    shuffle = True
    if balance_classes:
        sampler = get_weighted_sampler(full_dataset, train_idx)
        shuffle = False  # sampler and shuffle are mutually exclusive

    import torch
    use_pin = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=use_pin,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin,
        ),
    }

    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[DATA] Classes: {full_dataset.class_names}")
    print(f"[DATA] Distribution: {full_dataset.get_class_distribution()}")

    return loaders


# ------------------------------------------------------------------ #
#  Internal helper
# ------------------------------------------------------------------ #
class _SubsetWithTransform(Dataset):
    """Subset wrapper that applies a specific transform."""

    def __init__(self, dataset: FinancialDocumentDataset, indices: List[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
