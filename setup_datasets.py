"""
Real Dataset Setup Script
=========================
Downloads and prepares REAL datasets for the fraud detection pipeline:

1. RVL-CDIP     — 400K+ document images for ViT pretraining
2. CEDAR        — 2,640 genuine/forged signatures
3. Credit Card  — 284,807 transactions from Kaggle
4. User Images  — Your own bank documents placed in data/raw_images/

Usage:
    python setup_datasets.py --all           # Download everything
    python setup_datasets.py --rvl-cdip      # Only RVL-CDIP
    python setup_datasets.py --cedar         # Only CEDAR signatures
    python setup_datasets.py --creditcard    # Only credit card CSV
    python setup_datasets.py --check         # Check what's available
    python setup_datasets.py --prepare       # Prepare data dirs only
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
#  Directory Structure
# ============================================================
REQUIRED_DIRS = [
    DATA_DIR / "raw_images" / "genuine",
    DATA_DIR / "raw_images" / "fraud",
    DATA_DIR / "raw_images" / "tampered",
    DATA_DIR / "raw_images" / "forged",
    DATA_DIR / "processed",
    DATA_DIR / "cedar_signatures" / "genuine",
    DATA_DIR / "cedar_signatures" / "forged",
    DATA_DIR / "rvl_cdip",
    DATA_DIR / "transactions",
    DATA_DIR / "uploads",           # real-time uploads land here
]


def ensure_directories():
    """Create all required data directories."""
    print("Creating directory structure...")
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.relative_to(PROJECT_ROOT)}")
    print()


# ============================================================
#  1. RVL-CDIP (HuggingFace)
# ============================================================
def download_rvl_cdip(max_samples: int = 0):
    """
    Download RVL-CDIP dataset via HuggingFace `datasets` library.
    Set max_samples=0 for all data (~400 K images).
    """
    print("=" * 60)
    print("  Downloading RVL-CDIP Dataset")
    print("=" * 60)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset  # type: ignore

    output = DATA_DIR / "rvl_cdip"
    output.mkdir(parents=True, exist_ok=True)

    print("Loading from HuggingFace (this may take a while)...")
    ds = load_dataset("rvl_cdip")

    for split in ["train", "validation", "test"]:
        split_dir = output / split
        split_dir.mkdir(exist_ok=True)
        n = len(ds[split])
        limit = min(n, max_samples) if max_samples > 0 else n
        print(f"\nSaving {split} — {limit}/{n} images ...")

        for idx in range(limit):
            sample = ds[split][idx]
            img = sample["image"]
            label = sample["label"]
            cls_dir = split_dir / str(label)
            cls_dir.mkdir(exist_ok=True)
            img.save(cls_dir / f"{idx:06d}.png")

            if (idx + 1) % 5000 == 0:
                print(f"  {idx + 1}/{limit}")

    print(f"\n✅ RVL-CDIP saved to {output}")


# ============================================================
#  2. CEDAR Signatures
# ============================================================
def download_cedar():
    """
    Provide instructions for obtaining CEDAR signatures.
    Automatic download is not possible—dataset requires a request form.
    """
    print("=" * 60)
    print("  CEDAR Signature Dataset")
    print("=" * 60)
    print("""
The CEDAR dataset requires a request form.

Steps:
  1. Visit  https://www.cedar.buffalo.edu/NIJ/data/
  2. Fill the request form and agree to terms.
  3. After approval, download the ZIP.
  4. Extract into:
         data/cedar_signatures/genuine/   (original_*.png)
         data/cedar_signatures/forged/    (forgeries_*.png)

Alternative open datasets:
  • GPDS Synthetic  — https://www.gpds.ulpgc.es/
  • BHSig260        — Bengali/Hindi signatures (open access)
  • ICDAR 2011 SigComp

After placing images, re-run:
    python setup_datasets.py --check
""")


# ============================================================
#  3. Kaggle Credit Card Fraud
# ============================================================
def download_creditcard():
    """
    Download Kaggle Credit Card Fraud dataset.
    Requires `kaggle` CLI configured with API token.
    """
    print("=" * 60)
    print("  Kaggle Credit Card Fraud Dataset")
    print("=" * 60)

    dest = DATA_DIR / "transactions"
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "creditcard.csv"

    if csv_path.exists():
        print(f"Already exists: {csv_path}")
        return

    try:
        import kaggle  # type: ignore
        print("Downloading via Kaggle API ...")
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud",
            path=str(dest),
            unzip=True,
        )
        print(f"✅ Saved to {csv_path}")
    except Exception as e:
        print(f"Kaggle API failed: {e}")
        print("""
Manual download:
  1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. Download creditcard.csv (~144 MB)
  3. Place it at:  data/transactions/creditcard.csv

To use Kaggle CLI:
  pip install kaggle
  # Place ~/.kaggle/kaggle.json with your API key
  kaggle datasets download mlg-ulb/creditcardfraud -p data/transactions --unzip
""")


# ============================================================
#  4. Check Dataset Status
# ============================================================
def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in exts)


def check_datasets():
    """Print status of every dataset."""
    print("=" * 60)
    print("  Dataset Status")
    print("=" * 60)

    # Raw images (your own bank documents)
    raw = DATA_DIR / "raw_images"
    classes = ["genuine", "fraud", "tampered", "forged"]
    total_raw = 0
    for cls in classes:
        n = count_images(raw / cls)
        total_raw += n
        status = "✅" if n > 0 else "❌"
        print(f"  {status} raw_images/{cls}: {n} images")
    print(f"       Total raw documents: {total_raw}")

    # Uploads
    uploads = DATA_DIR / "uploads"
    n_up = count_images(uploads)
    print(f"\n  {'✅' if n_up else '—'} uploads/: {n_up} images (real-time uploads)")

    # CEDAR
    cedar = DATA_DIR / "cedar_signatures"
    n_gen = count_images(cedar / "genuine")
    n_forg = count_images(cedar / "forged")
    print(f"\n  {'✅' if n_gen else '❌'} cedar_signatures/genuine: {n_gen}")
    print(f"  {'✅' if n_forg else '❌'} cedar_signatures/forged:  {n_forg}")

    # RVL-CDIP
    rvl = DATA_DIR / "rvl_cdip"
    n_rvl = count_images(rvl)
    print(f"\n  {'✅' if n_rvl else '❌'} rvl_cdip/: {n_rvl} images")

    # Credit Card CSV
    cc = DATA_DIR / "transactions" / "creditcard.csv"
    if cc.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cc, nrows=5)
            import os as _os
            size_mb = _os.path.getsize(cc) / (1024 * 1024)
            print(f"\n  ✅ transactions/creditcard.csv: {size_mb:.1f} MB")
        except Exception:
            print(f"\n  ✅ transactions/creditcard.csv exists")
    else:
        print(f"\n  ❌ transactions/creditcard.csv: NOT FOUND")

    print()
    if total_raw == 0:
        print("⚠️  No documents in data/raw_images/!")
        print("   Place your real bank statements / cheques there,")
        print("   organised into genuine/ fraud/ tampered/ forged/ subfolders.")
    print()


# ============================================================
#  5. Prepare Data for Training
# ============================================================
def prepare_training_data():
    """
    Copy / symlink raw_images into data/processed/ for training.
    Performs basic validation (image opens correctly).
    """
    from PIL import Image

    src = DATA_DIR / "raw_images"
    dst = DATA_DIR / "processed"

    classes = ["genuine", "fraud", "tampered", "forged"]
    total = 0

    print("Preparing training data ...")
    for cls in classes:
        cls_src = src / cls
        cls_dst = dst / cls
        cls_dst.mkdir(parents=True, exist_ok=True)

        if not cls_src.exists():
            continue

        for img_path in cls_src.iterdir():
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                try:
                    img = Image.open(img_path)
                    img.verify()  # ensure image is valid
                    # copy (not move) so originals stay intact
                    shutil.copy2(img_path, cls_dst / img_path.name)
                    total += 1
                except Exception as e:
                    print(f"  ⚠️  Skipping invalid image {img_path.name}: {e}")

    print(f"✅ Prepared {total} images in data/processed/")
    return total


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Setup real datasets for fraud detection")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--rvl-cdip", action="store_true", help="Download RVL-CDIP")
    parser.add_argument("--rvl-cdip-limit", type=int, default=0, help="Max samples per split (0=all)")
    parser.add_argument("--cedar", action="store_true", help="CEDAR download instructions")
    parser.add_argument("--creditcard", action="store_true", help="Download Kaggle credit card data")
    parser.add_argument("--check", action="store_true", help="Check dataset status")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data from raw_images")

    args = parser.parse_args()

    ensure_directories()

    if args.check:
        check_datasets()
        return

    if args.prepare:
        check_datasets()
        prepare_training_data()
        return

    if args.all or args.rvl_cdip:
        download_rvl_cdip(args.rvl_cdip_limit)

    if args.all or args.cedar:
        download_cedar()

    if args.all or args.creditcard:
        download_creditcard()

    if not any([args.all, args.rvl_cdip, args.cedar, args.creditcard]):
        # Default: just check status
        check_datasets()
        print("Usage examples:")
        print("  python setup_datasets.py --all              # download everything")
        print("  python setup_datasets.py --creditcard        # download Kaggle CSV")
        print("  python setup_datasets.py --prepare           # prepare raw_images for training")
        print("  python setup_datasets.py --check             # status report")


if __name__ == "__main__":
    main()
