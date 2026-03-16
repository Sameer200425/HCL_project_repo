"""
Sample Data Creator for Dataset Integration Testing
====================================================
Creates sample datasets for testing the fraud detection pipeline
without requiring large external downloads.

Usage:
    python create_sample_data.py --all          # Create all samples
    python create_sample_data.py --creditcard   # Credit card transactions only
    python create_sample_data.py --rvl-sample   # Small RVL-CDIP sample
"""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
#  1. Sample Credit Card Fraud Dataset
# ============================================================
def create_creditcard_sample(n_samples: int = 10000, fraud_ratio: float = 0.02):
    """
    Create a sample credit card fraud dataset matching Kaggle format.
    
    Kaggle format:
      - Time (seconds from first transaction)
      - V1-V28 (PCA components)
      - Amount
      - Class (0=legitimate, 1=fraud)
    """
    print("=" * 60)
    print("  Creating Sample Credit Card Fraud Dataset")
    print("=" * 60)
    
    output_dir = DATA_DIR / "transactions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "creditcard.csv"
    
    if output_path.exists():
        print(f"File already exists: {output_path}")
        print("Use --force to overwrite")
        return output_path
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    print(f"Generating {n_samples} transactions ({n_fraud} fraud, {n_legit} legitimate)...")
    
    # Generate time (seconds over ~48 hours)
    time_vals = np.sort(np.random.randint(0, 172800, size=n_samples))
    
    # Generate PCA components V1-V28
    # Legitimate transactions: centered around 0
    # Fraudulent transactions: some outlier patterns
    
    data: dict[str, Any] = {
        'Time': time_vals,
    }
    
    # V1-V28 PCA components
    for i in range(1, 29):
        # Base values (normal distribution)
        base = np.random.normal(0, 1, size=n_samples)
        
        # Add fraud patterns (outliers in certain components)
        if i in [1, 3, 4, 7, 10, 12, 14, 17]:
            # These components typically differ for fraud in real data
            fraud_idx = random.sample(range(n_samples), n_fraud)
            for idx in fraud_idx:
                # Shift fraud values
                base[idx] = np.random.normal(-3.5 if i % 2 == 0 else 3.5, 0.8)
        
        data[f'V{i}'] = base
    
    # Amount (fraud tends to have different patterns)
    amounts = np.random.exponential(scale=88, size=n_samples)
    amounts = np.clip(amounts, 0, 25000)  # Cap at 25000
    
    # Fraud amounts: some very small (testing), some large
    fraud_indices = sorted(random.sample(range(n_samples), n_fraud))
    for idx in fraud_indices:
        if random.random() < 0.3:
            amounts[idx] = random.uniform(0.5, 10)  # Small test transactions
        else:
            amounts[idx] = random.uniform(200, 5000)  # Larger fraud amounts
    
    data['Amount'] = np.round(amounts, 2)
    
    # Class labels (0=legitimate, 1=fraud)
    labels = np.zeros(n_samples, dtype=int)
    labels[fraud_indices] = 1
    data['Class'] = labels
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created {output_path}")
    print(f"   Total rows: {len(df)}")
    print(f"   Fraud cases: {df['Class'].sum()} ({100*df['Class'].mean():.2f}%)")
    print(f"   Legitimate: {(df['Class'] == 0).sum()}")
    print(f"   Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
    
    return output_path


# ============================================================
#  2. Sample RVL-CDIP Dataset (Small)
# ============================================================
def download_rvl_sample(n_per_class: int = 50):
    """
    Download a small sample of RVL-CDIP for testing.
    Total: 50 * 16 classes = 800 images (vs 400K full dataset)
    """
    print("=" * 60)
    print("  Downloading RVL-CDIP Sample")
    print("=" * 60)
    
    output_dir = DATA_DIR / "rvl_cdip"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset
    
    # Check if already downloaded
    existing = sum(1 for _ in output_dir.rglob("*.png"))
    if existing > 100:
        print(f"Already have {existing} images in {output_dir}")
        return output_dir
    
    print(f"Loading RVL-CDIP (streaming mode)...")
    print(f"Downloading {n_per_class} images per class (16 classes)...")
    
    # Use streaming to avoid downloading full dataset
    try:
        ds = load_dataset("rvl_cdip", split="train", streaming=True)
    except Exception as e:
        print(f"Could not stream dataset: {e}")
        print("\nAlternative: Download manually from HuggingFace")
        return None
    
    # RVL-CDIP class names
    class_names = [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific_report", "scientific_publication", "specification",
        "file_folder", "news_article", "budget", "invoice",
        "presentation", "questionnaire", "resume", "memo"
    ]
    
    # Track how many we have per class
    class_counts = {i: 0 for i in range(16)}
    saved = 0
    
    for sample in ds:
        label = sample["label"]
        
        if class_counts[label] >= n_per_class:
            continue
        
        # Save image
        class_dir = output_dir / class_names[label]
        class_dir.mkdir(exist_ok=True)
        
        img = sample["image"]
        img_path = class_dir / f"{class_counts[label]:04d}.png"
        img.save(img_path)
        
        class_counts[label] += 1
        saved += 1
        
        if saved % 100 == 0:
            print(f"  Saved {saved} images...")
        
        # Check if we have enough
        if all(c >= n_per_class for c in class_counts.values()):
            break
    
    print(f"\n✅ Saved {saved} images to {output_dir}")
    for i, name in enumerate(class_names):
        print(f"   {name}: {class_counts[i]} images")
    
    return output_dir


# ============================================================
#  3. Sample CEDAR-style Signatures (Synthetic)
# ============================================================
def create_signature_samples(n_per_class: int = 50):
    """
    Create synthetic signature-like images for testing.
    This simulates the CEDAR dataset structure.
    """
    print("=" * 60)
    print("  Creating Synthetic Signature Samples")
    print("=" * 60)
    
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available")
        return None
    
    output_dir = DATA_DIR / "cedar_signatures"
    genuine_dir = output_dir / "genuine"
    forged_dir = output_dir / "forged"
    
    genuine_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    existing = sum(1 for _ in genuine_dir.glob("*.png")) + sum(1 for _ in forged_dir.glob("*.png"))
    if existing > 20:
        print(f"Already have {existing} signature images")
        return output_dir
    
    # Sample names to generate signatures for
    names = [
        "John Smith", "Sarah Johnson", "Michael Chen", "Emily Davis",
        "Robert Wilson", "Jennifer Brown", "David Lee", "Lisa Anderson",
        "James Taylor", "Maria Garcia"
    ]
    
    def draw_signature(name: str, is_forged: bool = False) -> Image.Image:
        """Generate a signature-like image."""
        # Random dimensions (signatures vary in size)
        width = random.randint(300, 500)
        height = random.randint(100, 180)
        
        # White/off-white background
        bg_color = (255, 255, 255) if not is_forged else (250, 250, 248)
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to use a handwriting-style font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", random.randint(28, 42))
        except:
            font = ImageFont.load_default()
        
        # Signature color (dark blue/black)
        color = (10, 10, 80) if random.random() > 0.5 else (0, 0, 0)
        
        # Add some style variations
        x_offset = random.randint(20, 50)
        y_offset = random.randint(20, 50)
        
        # Draw the name in a "signature" style (slightly italic/slanted feel via positioning)
        parts = name.split()
        
        for i, part in enumerate(parts):
            x = x_offset + i * 120 + random.randint(-10, 10)
            y = y_offset + random.randint(-5, 5) + i * 10
            draw.text((x, y), part, fill=color, font=font)
        
        # Add some signature-like flourishes (simple lines)
        if random.random() > 0.3:
            # Underline
            line_y = y_offset + 45 + random.randint(0, 20)
            draw.line(
                [(x_offset, line_y), (x_offset + 150 + random.randint(0, 100), line_y + random.randint(-5, 5))],
                fill=color,
                width=1
            )
        
        # For forged signatures, add some imperfections
        if is_forged:
            # Slight blur or noise effect (simulated with extra strokes)
            if random.random() > 0.5:
                # Add hesitation marks
                for _ in range(random.randint(2, 5)):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    draw.point((x, y), fill=(150, 150, 150))
        
        return img
    
    print(f"Generating {n_per_class} genuine signatures...")
    for i in range(n_per_class):
        name = random.choice(names)
        img = draw_signature(name, is_forged=False)
        img.save(genuine_dir / f"genuine_{i:04d}.png")
    
    print(f"Generating {n_per_class} forged signatures...")
    for i in range(n_per_class):
        name = random.choice(names)
        img = draw_signature(name, is_forged=True)
        img.save(forged_dir / f"forged_{i:04d}.png")
    
    print(f"\n✅ Created {2 * n_per_class} signature images in {output_dir}")
    return output_dir


# ============================================================
#  Main CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Create sample datasets for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--all', action='store_true', help='Create all sample data')
    parser.add_argument('--creditcard', action='store_true', help='Create credit card sample')
    parser.add_argument('--rvl-sample', action='store_true', help='Download RVL-CDIP sample')
    parser.add_argument('--signatures', action='store_true', help='Create synthetic signatures')
    parser.add_argument('--samples', type=int, default=10000, help='Number of credit card samples')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()
    
    if not any([args.all, args.creditcard, args.rvl_sample, args.signatures]):
        args.all = True  # Default to all
    
    if args.all or args.creditcard:
        if args.force and (DATA_DIR / "transactions" / "creditcard.csv").exists():
            os.remove(DATA_DIR / "transactions" / "creditcard.csv")
        create_creditcard_sample(n_samples=args.samples)
    
    if args.all or args.signatures:
        create_signature_samples(n_per_class=50)
    
    if args.all or args.rvl_sample:
        download_rvl_sample(n_per_class=50)
    
    print("\n" + "=" * 60)
    print("  Sample Data Creation Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python setup_datasets.py --check")
    print("  2. Run: python data_integration/unified_loader.py --check")
    print("  3. Test the pipeline: python run_pipeline.py")


if __name__ == "__main__":
    main()
