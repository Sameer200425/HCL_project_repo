"""
Quick verification of trained models and integrated data.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import pandas as pd

def verify_checkpoints():
    """Verify all model checkpoints load correctly."""
    print("=" * 60)
    print("  VERIFYING MODEL CHECKPOINTS")
    print("=" * 60)
    
    checkpoints = {
        'cnn_best.pth': 'CNN (ResNet50)',
        'vit_best.pth': 'ViT (from scratch)',
        'vit_ssl_best.pth': 'ViT + SSL (MAE)',
        'hybrid_best.pth': 'Hybrid CNN+ViT',
    }
    
    for fname, name in checkpoints.items():
        path = Path('checkpoints') / fname
        if path.exists():
            try:
                cp = torch.load(path, map_location='cpu', weights_only=True)
                val_acc = cp.get('val_acc', 'N/A')
                if isinstance(val_acc, float):
                    val_acc = f"{val_acc:.4f}"
                print(f"  OK  {name}: val_acc = {val_acc}")
            except Exception as e:
                print(f"  ERR {name}: {e}")
        else:
            print(f"  --  {name}: Not found")


def verify_datasets():
    """Verify all datasets are available."""
    print("\n" + "=" * 60)
    print("  VERIFYING DATASETS")
    print("=" * 60)
    
    datasets = {
        'data/raw_images': ('Bank Documents', 'dir'),
        'data/cedar_signatures': ('CEDAR Signatures', 'dir'),
        'data/rvl_cdip': ('RVL-CDIP Documents', 'dir'),
        'data/transactions/creditcard.csv': ('Credit Card Transactions', 'file'),
    }
    
    for path, (name, dtype) in datasets.items():
        p = Path(path)
        if dtype == 'dir':
            if p.exists():
                count = sum(1 for f in p.rglob('*') if f.suffix.lower() in {'.png', '.jpg', '.jpeg'})
                print(f"  OK  {name}: {count} images")
            else:
                print(f"  --  {name}: Not found")
        else:
            if p.exists():
                if path.endswith('.csv'):
                    df = pd.read_csv(p)
                    print(f"  OK  {name}: {len(df):,} rows")
                else:
                    size = p.stat().st_size / 1024 / 1024
                    print(f"  OK  {name}: {size:.1f} MB")
            else:
                print(f"  --  {name}: Not found")


def verify_credit_card_sample():
    """Verify credit card data quality."""
    print("\n" + "=" * 60)
    print("  CREDIT CARD DATA ANALYSIS")
    print("=" * 60)
    
    csv_path = Path('data/transactions/creditcard.csv')
    if not csv_path.exists():
        print("  Credit card data not found!")
        return
    
    df = pd.read_csv(csv_path)
    fraud_count = df['Class'].sum()
    total = len(df)
    fraud_pct = 100 * fraud_count / total
    
    print(f"  Total transactions: {total:,}")
    print(f"  Fraud cases: {fraud_count:,} ({fraud_pct:.3f}%)")
    print(f"  Legitimate: {total - fraud_count:,}")
    print(f"  Features: {len(df.columns)} columns")
    print(f"  Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")


def verify_model_results():
    """Show model comparison results."""
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    csv_path = Path('results/model_comparison.csv')
    if not csv_path.exists():
        print("  No comparison results found!")
        return
    
    df = pd.read_csv(csv_path)
    
    # Format nicely
    print(f"\n  {'Model':<20} {'Accuracy':>10} {'F1-Score':>10} {'Size (MB)':>10}")
    print("  " + "-" * 52)
    for _, row in df.iterrows():
        print(f"  {row['Model']:<20} {row['Accuracy']:>10.4f} {row['F1-Score']:>10.4f} {row['Size (MB)']:>10.1f}")
    
    # Best model
    best_idx = int(df['Accuracy'].idxmax())
    best = df.iloc[best_idx]
    print(f"\n  BEST: {best['Model']} with {best['Accuracy']:.2%} accuracy")


if __name__ == "__main__":
    verify_checkpoints()
    verify_datasets()
    verify_credit_card_sample()
    verify_model_results()
    
    print("\n" + "=" * 60)
    print("  VERIFICATION COMPLETE")
    print("=" * 60)
