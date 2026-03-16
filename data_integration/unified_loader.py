"""
Unified Data Integration Module
================================
Central interface for loading ALL real datasets used by the fraud detection system.

Supported datasets:
  - raw_images       : Your own bank documents (genuine/fraud/tampered/forged)
  - cedar_signatures : CEDAR signature verification dataset
  - creditcard       : Kaggle credit-card transaction CSV
  - rvl_cdip         : RVL-CDIP document images (optional, for SSL pretraining)
  - uploads          : Real-time uploaded documents via the API

Usage:
    python data_integration/unified_loader.py --check
    python data_integration/unified_loader.py --train-all
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))


# ============================================================
#  Dataset Registry
# ============================================================
class UnifiedDataManager:
    """
    Single entry-point for every dataset in the project.
    No synthetic / demo data — everything is real or user-provided.
    """

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, base_dir: str = 'data'):
        self.base_dir = Path(base_dir)

        self.config: Dict[str, dict] = {
            'raw_images': {
                'path': self.base_dir / 'raw_images',
                'type': 'image',
                'classes': ['genuine', 'fraud', 'tampered', 'forged'],
                'description': 'Bank documents organised by fraud class'
            },
            'uploads': {
                'path': self.base_dir / 'uploads',
                'type': 'image',
                'classes': [],
                'description': 'Real-time uploaded documents (API / frontend)'
            },
            'cedar_signatures': {
                'path': self.base_dir / 'cedar_signatures',
                'type': 'image',
                'classes': ['genuine', 'forged'],
                'description': 'CEDAR handwritten-signature dataset'
            },
            'rvl_cdip': {
                'path': self.base_dir / 'rvl_cdip',
                'type': 'image',
                'classes': ['16 document types'],
                'description': 'RVL-CDIP document images (SSL pretraining)'
            },
            'creditcard': {
                'path': self.base_dir / 'transactions' / 'creditcard.csv',
                'type': 'tabular',
                'classes': ['legitimate', 'fraud'],
                'description': 'Kaggle credit-card transaction data'
            }
        }
    
    # --------------------------------------------------------
    #  Status helpers
    # --------------------------------------------------------
    def _count_images(self, folder: Path) -> int:
        if not folder.exists():
            return 0
        return sum(1 for f in folder.rglob("*") if f.suffix.lower() in self.IMAGE_EXTS)

    def check_datasets(self) -> Dict[str, Dict]:
        """Return availability & size for every registered dataset."""
        status: Dict[str, Dict] = {}
        for name, cfg in self.config.items():
            p = cfg['path']
            if cfg['type'] == 'tabular':
                exists = p.is_file()
                size = 0
                if exists:
                    try:
                        import pandas as pd
                        size = len(pd.read_csv(p))
                    except Exception:
                        size = -1
            else:
                exists = p.is_dir() and self._count_images(p) > 0
                size = self._count_images(p) if exists else 0

            status[name] = {
                'exists': exists,
                'path': str(p),
                'type': cfg['type'],
                'size': size,
                'description': cfg['description']
            }
        return status

    # --------------------------------------------------------
    #  DataLoader factories
    # --------------------------------------------------------
    def get_dataloader(self, dataset_name: str, split: str = 'train',
                       batch_size: int = 32, **kwargs) -> DataLoader:
        """
        Get a PyTorch DataLoader for specified dataset.
        
        Args:
            dataset_name: Name of dataset
            split: 'train', 'val', or 'test'
            batch_size: Batch size
            **kwargs: Additional arguments for DataLoader
        """
        if dataset_name not in self.config:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        cfg = self.config[dataset_name]

        if cfg['type'] == 'image':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            data_path = cfg['path'] / split if (cfg['path'] / split).exists() else cfg['path']
            dataset = datasets.ImageFolder(str(data_path), transform=transform)

        elif cfg['type'] == 'tabular':
            import pandas as pd
            from torch.utils.data import TensorDataset

            df = pd.read_csv(cfg['path'])
            feature_cols = [c for c in df.columns if c.startswith('V') or c in ('Amount', 'Time')]
            X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            y = torch.tensor(df['Class'].values, dtype=torch.long)
            dataset = TensorDataset(X, y)
        else:
            raise ValueError(f"Unsupported type: {cfg['type']}")

        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=(split == 'train'), **kwargs)

    # --------------------------------------------------------
    #  Training helpers
    # --------------------------------------------------------
    def train_all_models(self):
        """Train sub-models on whichever real datasets are present."""
        status = self.check_datasets()
        results: Dict[str, object] = {}

        print("=" * 60)
        print("Training Models on Real Datasets")
        print("=" * 60)

        # 1. Signature verification
        if status['cedar_signatures']['exists']:
            print("\n[1] Training Signature Verification ViT ...")
            from data_integration.cedar_signature_loader import train_signature_vit
            train_signature_vit(
                str(self.config['cedar_signatures']['path']),
                epochs=30,
            )
            results['signature_vit'] = True
        else:
            print("\n⚠️  CEDAR signatures not found — skipping.")
            results['signature_vit'] = False

        # 2. Transaction risk scorer
        if status['creditcard']['exists']:
            print("\n[2] Training Transaction Risk Scorer ...")
            from data_integration.creditcard_fraud_loader import train_risk_model
            train_risk_model(
                str(self.config['creditcard']['path']),
                'checkpoints/risk_model.pkl',
            )
            results['risk_scorer'] = True
        else:
            print("\n⚠️  Credit-card CSV not found — skipping.")
            results['risk_scorer'] = False

        # 3. RVL-CDIP pretraining hint
        if status['rvl_cdip']['exists'] and status['rvl_cdip']['size'] > 100:
            print("\n[3] RVL-CDIP available for MAE pretraining.")
            print("    Run:  python data_integration/rvl_cdip_loader.py --pretrain")
            results['rvl_cdip_pretrain'] = 'available'
        else:
            print("\n⚠️  RVL-CDIP not found — SSL pretraining skipped.")
            results['rvl_cdip_pretrain'] = False

        # Summary
        print("\n" + "=" * 60)
        for name, r in results.items():
            mark = "✅" if r is True else ("⏭️" if r == 'available' else "❌")
            print(f"  {mark} {name}")
        print("=" * 60)
        return results


# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Unified Data Integration (real data only)')
    parser.add_argument('--check', action='store_true', help='Check dataset status')
    parser.add_argument('--train-all', action='store_true', help='Train models on available data')
    args = parser.parse_args()

    mgr = UnifiedDataManager()

    if args.check:
        status = mgr.check_datasets()
        print("\n📊 Dataset Status:")
        for name, info in status.items():
            mark = "✅" if info['exists'] else "❌"
            print(f"  {mark} {name}: {info['size']} samples — {info['description']}")
        print()
    elif args.train_all:
        mgr.train_all_models()
    else:
        status = mgr.check_datasets()
        print("\n📊 Dataset Status:")
        for name, info in status.items():
            mark = "✅" if info['exists'] else "❌"
            print(f"  {mark} {name}: {info['size']} samples — {info['description']}")
        print("\nNext steps:")
        print("  python setup_datasets.py --check       # detailed setup guide")
        print("  python setup_datasets.py --prepare      # copy raw_images → processed/")


if __name__ == '__main__':
    main()
