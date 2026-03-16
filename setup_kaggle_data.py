"""
Kaggle Credit Card Fraud Dataset Setup
======================================
Downloads and sets up the Kaggle Credit Card Fraud dataset.

Dataset Information:
- Source: Kaggle (mlg-ulb/creditcardfraud)
- Size: 284,807 transactions
- Fraud cases: 492 (0.17% - highly imbalanced)
- Features: Time, V1-V28 (PCA), Amount, Class

SETUP OPTIONS:
==============

Option 1: Kaggle API (Recommended)
----------------------------------
1. Create Kaggle account at https://www.kaggle.com
2. Go to Settings -> API -> Create New Token
3. Place the downloaded kaggle.json in:
   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
   - Linux/Mac: ~/.kaggle/kaggle.json
4. Run: python setup_kaggle_data.py --download

Option 2: Manual Download
-------------------------
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" button
3. Extract creditcard.csv to: data/transactions/creditcard.csv

Option 3: Use Sample Data (Testing)
-----------------------------------
Run: python setup_kaggle_data.py --sample
Creates a realistic 50,000 transaction sample dataset

"""

import os
import sys
import argparse
from pathlib import Path
import random
from typing import Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "transactions"


def check_kaggle_credentials():
    """Check if Kaggle API credentials exist."""
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_path.exists()


def download_via_kaggle():
    """Download dataset using Kaggle API."""
    if not check_kaggle_credentials():
        print("❌ Kaggle credentials not found!")
        print(f"   Expected location: {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("\nTo set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print("  4. Move downloaded kaggle.json to ~/.kaggle/")
        return False
    
    print("Downloading via Kaggle API...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    import kaggle
    kaggle.api.dataset_download_files(
        "mlg-ulb/creditcardfraud",
        path=str(DATA_DIR),
        unzip=True
    )
    
    csv_path = DATA_DIR / "creditcard.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n✅ Downloaded {len(df)} transactions")
        print(f"   Fraud cases: {df['Class'].sum()}")
        return True
    return False


def create_realistic_sample(n_samples: int = 50000, fraud_ratio: float = 0.00173):
    """
    Create a realistic large sample dataset matching Kaggle format.
    Uses statistical distributions similar to the real dataset.
    """
    print("=" * 60)
    print(f"  Creating Realistic Credit Card Sample ({n_samples:,} transactions)")
    print("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "creditcard.csv"
    
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud
    
    print(f"Generating {n_samples:,} transactions...")
    print(f"  Legitimate: {n_legit:,}")
    print(f"  Fraud: {n_fraud:,} ({100*fraud_ratio:.3f}%)")
    
    # Time: seconds over 2 days (172800 seconds)
    time_vals = np.sort(np.random.randint(0, 172800, size=n_samples))
    
    # Generate PCA components V1-V28 with realistic distributions
    # Based on statistical properties of the real dataset
    data: dict[str, Any] = {'Time': time_vals}
    
    # Real dataset statistics (approximate means and stds)
    pca_stats = {
        'V1': (-0.1, 1.95), 'V2': (0.0, 1.65), 'V3': (-0.1, 1.52),
        'V4': (0.0, 1.42), 'V5': (-0.1, 1.38), 'V6': (0.0, 1.33),
        'V7': (-0.05, 1.24), 'V8': (0.0, 1.19), 'V9': (-0.05, 1.10),
        'V10': (-0.08, 1.09), 'V11': (0.0, 1.02), 'V12': (-0.1, 1.0),
        'V13': (0.0, 1.0), 'V14': (-0.03, 1.0), 'V15': (0.0, 1.0),
        'V16': (-0.03, 0.87), 'V17': (-0.02, 0.85), 'V18': (0.0, 0.84),
        'V19': (0.0, 0.81), 'V20': (0.0, 0.77), 'V21': (0.0, 0.73),
        'V22': (0.0, 0.73), 'V23': (0.0, 0.62), 'V24': (0.0, 0.61),
        'V25': (0.0, 0.52), 'V26': (0.0, 0.48), 'V27': (0.0, 0.40),
        'V28': (0.0, 0.33)
    }
    
    # Fraud-indicative components (where fraud has different distribution)
    fraud_shift = {
        'V1': -4.0, 'V2': 2.5, 'V3': -5.0, 'V4': 3.5, 'V5': -2.0,
        'V7': -5.0, 'V9': -3.0, 'V10': -4.0, 'V11': 3.0, 'V12': -5.0,
        'V14': -6.0, 'V16': -4.0, 'V17': -5.0
    }
    
    # Generate indices for fraud samples
    fraud_indices = set(random.sample(range(n_samples), n_fraud))
    
    for v_name, (mean, std) in pca_stats.items():
        # Generate normal values
        values = np.random.normal(mean, std, size=n_samples)
        
        # Modify fraud samples
        if v_name in fraud_shift:
            for idx in fraud_indices:
                shift = fraud_shift[v_name]
                values[idx] = np.random.normal(mean + shift, std * 0.8)
        
        data[v_name] = values
    
    # Amount: exponential distribution
    # Real dataset: mean ~88, most transactions small
    amounts = np.random.exponential(scale=88, size=n_samples)
    amounts = np.clip(amounts, 0.01, 25000)
    
    # Fraud amounts tend to be either very small (testing) or moderate
    for idx in fraud_indices:
        if random.random() < 0.2:
            amounts[idx] = random.uniform(0.5, 5.0)  # Test transactions
        elif random.random() < 0.6:
            amounts[idx] = random.uniform(100, 1500)  # Moderate fraud
        else:
            amounts[idx] = random.uniform(1500, 5000)  # Large fraud
    
    data['Amount'] = np.round(amounts, 2)
    
    # Class labels
    labels = np.zeros(n_samples, dtype=int)
    for idx in fraud_indices:
        labels[idx] = 1
    data['Class'] = labels
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle to mix fraud with legitimate
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Total rows: {len(df):,}")
    print(f"   Fraud cases: {df['Class'].sum()} ({100*df['Class'].mean():.3f}%)")
    print(f"   Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
    print(f"   Mean amount: ${df['Amount'].mean():.2f}")
    
    return output_path


def check_status():
    """Check current dataset status."""
    csv_path = DATA_DIR / "creditcard.csv"
    
    print("=" * 60)
    print("  Credit Card Fraud Dataset Status")
    print("=" * 60)
    
    if not csv_path.exists():
        print(f"\n❌ Dataset not found at {csv_path}")
        print("\nOptions:")
        print("  1. python setup_kaggle_data.py --download  (requires Kaggle API)")
        print("  2. python setup_kaggle_data.py --sample    (create sample data)")
        print("  3. Manual download from Kaggle website")
        return
    
    df = pd.read_csv(csv_path)
    file_size = csv_path.stat().st_size / 1024 / 1024
    
    print(f"\n✅ Dataset found: {csv_path}")
    print(f"   File size: {file_size:.1f} MB")
    print(f"   Total rows: {len(df):,}")
    print(f"   Fraud cases: {df['Class'].sum():,} ({100*df['Class'].mean():.3f}%)")
    print(f"   Legitimate: {(df['Class']==0).sum():,}")
    
    # Check if it's the real dataset or sample
    if len(df) >= 200000:
        print("\n   ✅ This appears to be the FULL Kaggle dataset")
    else:
        print("\n   ⚠️  This appears to be a SAMPLE dataset")
        print("      For production, download the full dataset from Kaggle")


def main():
    parser = argparse.ArgumentParser(
        description="Kaggle Credit Card Fraud Dataset Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--download', action='store_true', 
                        help='Download via Kaggle API')
    parser.add_argument('--sample', action='store_true',
                        help='Create realistic sample (50K transactions)')
    parser.add_argument('--large-sample', action='store_true',
                        help='Create large sample (200K transactions)')
    parser.add_argument('--status', action='store_true',
                        help='Check current dataset status')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing file')
    
    args = parser.parse_args()
    
    if args.force and (DATA_DIR / "creditcard.csv").exists():
        os.remove(DATA_DIR / "creditcard.csv")
    
    if args.download:
        download_via_kaggle()
    elif args.sample:
        create_realistic_sample(n_samples=50000)
    elif args.large_sample:
        create_realistic_sample(n_samples=200000)
    elif args.status:
        check_status()
    else:
        # Default: check status and suggest options
        check_status()


if __name__ == "__main__":
    main()
