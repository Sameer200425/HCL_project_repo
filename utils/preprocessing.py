"""
Image preprocessing utilities for financial document images.
Handles cleaning, normalization, resizing, and corruption removal.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image


def clean_image(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Validate and clean an image. Returns None if corrupted.
    
    Args:
        image: Input image as numpy array.
    
    Returns:
        Cleaned image or None if invalid.
    """
    if image is None or image.size == 0:
        return None
    if len(image.shape) < 2:
        return None
    # Convert grayscale to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = (224, 224),
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """
    Resize image to target size with high-quality interpolation.
    
    Args:
        image: Input image.
        size: Target (height, width).
        interpolation: OpenCV interpolation method.
    
    Returns:
        Resized image.
    """
    return cv2.resize(image, (size[1], size[0]), interpolation=interpolation)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Normalize image with ImageNet statistics (or custom).
    
    Args:
        image: Input image (H, W, C) in [0, 255].
        mean: Per-channel mean.
        std: Per-channel std.
    
    Returns:
        Normalized image as float32.
    """
    image = image.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    image = (image - mean_arr) / std_arr
    return image


def remove_corrupted_images(image_dir: str) -> List[str]:
    """
    Scan directory and remove corrupted/unreadable images.
    
    Args:
        image_dir: Path to image directory.
    
    Returns:
        List of removed file paths.
    """
    removed = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if Path(fname).suffix.lower() not in extensions:
                continue
            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath)
                img.verify()  # Check integrity
                img.close()
                # Re-open to also verify readability
                img = Image.open(fpath)
                img.load()
                img.close()
            except Exception:
                os.remove(fpath)
                removed.append(fpath)
    
    return removed


def process_dataset(
    input_dir: str,
    output_dir: str,
    image_size: Tuple[int, int] = (224, 224),
) -> dict:
    """
    Full preprocessing pipeline: clean, resize, and save images.
    
    Args:
        input_dir: Raw images directory.
        output_dir: Processed images directory.
        image_size: Target image size.
    
    Returns:
        Dictionary with processing statistics.
    """
    stats = {"processed": 0, "skipped": 0, "total": 0}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if Path(fname).suffix.lower() not in extensions:
                continue
            stats["total"] += 1
            
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(root, input_dir)
            out_folder = os.path.join(output_dir, rel_path)
            Path(out_folder).mkdir(parents=True, exist_ok=True)
            
            image = cv2.imread(fpath)
            if image is None:
                stats["skipped"] += 1
                continue
            
            cleaned = clean_image(image)
            
            if cleaned is None:
                stats["skipped"] += 1
                continue
            
            cleaned = resize_image(cleaned, image_size)
            out_path = os.path.join(out_folder, fname)
            cv2.imwrite(out_path, cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))
            stats["processed"] += 1
    
    return stats
