"""
Download RVL-CDIP Sample Dataset
================================
Downloads a smaller version of RVL-CDIP (1600 images, 100 per class)
for SSL pretraining testing.
"""

import os
from pathlib import Path
from typing import Any, cast
from datasets import load_dataset

DATA_DIR = Path("data/rvl_cdip")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# RVL-CDIP class names (16 classes)
CLASS_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

def download_rvl_cdip_sample():
    print("=" * 60)
    print("  Downloading RVL-CDIP Sample (100 per class)")
    print("=" * 60)
    
    # Check if already downloaded
    existing = sum(1 for _ in DATA_DIR.rglob("*.png")) + sum(1 for _ in DATA_DIR.rglob("*.jpg"))
    if existing > 500:
        print(f"Already have {existing} images in {DATA_DIR}")
        return
    
    print("Loading from HuggingFace...")
    ds = load_dataset("jordyvl/rvl_cdip_100_examples_per_class", split="train")
    
    print(f"Downloaded {len(ds)} samples")
    
    # Create class directories
    for name in CLASS_NAMES:
        (DATA_DIR / name).mkdir(exist_ok=True)
    
    # Save images
    class_counts = {i: 0 for i in range(16)}
    
    print("Saving images...")
    for idx, sample in enumerate(ds):
        row = cast(dict[str, Any], sample)
        label: int = row["label"]
        img = row["image"]
        
        class_name = CLASS_NAMES[label]
        img_path = DATA_DIR / class_name / f"{class_counts[label]:04d}.png"
        img.save(img_path)
        class_counts[label] += 1
        
        if (idx + 1) % 200 == 0:
            print(f"  Saved {idx + 1}/{len(ds)} images...")
    
    print(f"\n✅ Saved {sum(class_counts.values())} images to {DATA_DIR}")
    print("\nClass distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {class_counts[i]} images")


if __name__ == "__main__":
    download_rvl_cdip_sample()
