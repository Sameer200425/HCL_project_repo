"""
RVL-CDIP Dataset Integration for ViT Pretraining
================================================
Downloads and prepares RVL-CDIP dataset (400K+ document images)
for self-supervised pretraining with MAE.

Usage:
    python data_integration/rvl_cdip_loader.py --download
    python data_integration/rvl_cdip_loader.py --prepare
    python data_integration/rvl_cdip_loader.py --pretrain
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class RVLCDIPDataset(Dataset):
    """RVL-CDIP Dataset wrapper for PyTorch."""
    
    # 16 document categories
    CLASSES = [
        'letter', 'form', 'email', 'handwritten', 'advertisement',
        'scientific_report', 'scientific_publication', 'specification',
        'file_folder', 'news_article', 'budget', 'invoice',
        'presentation', 'questionnaire', 'resume', 'memo'
    ]
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        """
        Args:
            data_dir: Path to RVL-CDIP data directory
            split: 'train', 'val', or 'test'
            transform: torchvision transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or self._default_transform()
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self):
        """Load sample paths and labels from label file."""
        samples = []
        label_file = self.data_dir / f'labels/{self.split}.txt'
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_path, label = parts
                        full_path = self.data_dir / 'images' / img_path
                        if full_path.exists():
                            samples.append((str(full_path), int(label)))
        else:
            # Fallback: scan directory structure
            for class_idx, class_name in enumerate(self.CLASSES):
                class_dir = self.data_dir / self.split / class_name
                if class_dir.exists():
                    for img_file in class_dir.glob('*.tif'):
                        samples.append((str(img_file), class_idx))
                        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (224, 224), color='white')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label


def download_rvl_cdip(output_dir: str):
    """
    Download RVL-CDIP dataset using HuggingFace datasets library.
    
    Args:
        output_dir: Directory to save dataset
    """
    print("=" * 60)
    print("RVL-CDIP Dataset Downloader")
    print("=" * 60)
    
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset  # type: ignore
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nDownloading RVL-CDIP dataset from HuggingFace...")
    print("This may take a while (dataset is ~40GB)...\n")
    
    # Load dataset
    dataset = load_dataset("rvl_cdip")
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Val samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Save to disk
    for split in ['train', 'validation', 'test']:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving {split} split...")
        for idx, sample in enumerate(tqdm(dataset[split])):
            img = sample['image']
            label = sample['label']
            
            # Create class directory
            class_dir = split_dir / str(label)
            class_dir.mkdir(exist_ok=True)
            
            # Save image
            img_path = class_dir / f"{idx:06d}.png"
            img.save(img_path)
            
            # Limit for demo (remove this in production)
            if idx >= 10000:
                print(f"  (Limited to 10,000 samples for demo)")
                break
    
    print("\n✅ Download complete!")
    print(f"Dataset saved to: {output_path}")
    
    return output_path


def prepare_for_mae_pretraining(data_dir: str, output_dir: str, num_samples: int = 50000):
    """
    Prepare RVL-CDIP images for MAE self-supervised pretraining.
    
    Args:
        data_dir: Path to RVL-CDIP dataset
        output_dir: Path to save prepared images
        num_samples: Number of samples to prepare
    """
    print("=" * 60)
    print("Preparing RVL-CDIP for MAE Pretraining")
    print("=" * 60)
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    all_images = []
    for ext in ['*.png', '*.jpg', '*.tif', '*.tiff']:
        all_images.extend(data_path.rglob(ext))
    
    print(f"Found {len(all_images)} images")
    
    # Sample subset
    if len(all_images) > num_samples:
        np.random.seed(42)
        indices = np.random.choice(len(all_images), num_samples, replace=False)
        all_images = [all_images[i] for i in indices]
    
    print(f"Preparing {len(all_images)} images for pretraining...")
    
    # Transform for MAE
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
    ])
    
    # Process images
    for idx, img_path in enumerate(tqdm(all_images)):
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            
            output_file = output_path / f"doc_{idx:06d}.png"
            img.save(output_file)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\n✅ Prepared {len(all_images)} images for MAE pretraining")
    print(f"Output directory: {output_path}")


def pretrain_vit_with_rvl_cdip(data_dir: str, checkpoint_dir: str, epochs: int = 50):
    """
    Pretrain ViT using MAE on RVL-CDIP dataset.
    
    Args:
        data_dir: Path to prepared images
        checkpoint_dir: Path to save checkpoints
        epochs: Number of pretraining epochs
    """
    from models.vit_model import VisionTransformer
    from ssl_pretraining.mae_model import MAEEncoder, MAEDecoder
    
    print("=" * 60)
    print("MAE Pretraining on RVL-CDIP")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Simple image folder dataset
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print(f"Training samples: {len(dataset)}")
    
    # Create MAE components
    encoder = MAEEncoder(
        image_size=224,
        patch_size=16,
        embed_dim=192,
        num_heads=3,
        depth=4
    ).to(device)
    
    decoder = MAEDecoder(
        patch_size=16,
        encoder_dim=192,
        decoder_dim=96,
        decoder_heads=3,
        decoder_depth=2,
        num_patches=(224 // 16) ** 2
    ).to(device)
    
    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    mask_ratio = 0.75
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, _ in pbar:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Create random mask
            num_patches = (224 // 16) ** 2
            num_masked = int(num_patches * mask_ratio)
            
            # Random mask for each sample
            noise = torch.rand(batch_size, num_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            mask = torch.zeros(batch_size, num_patches, device=device)
            mask[:, :num_masked] = 1
            mask = torch.gather(mask, dim=1, index=ids_restore)
            
            # Encode
            encoded, _ = encoder(images, mask)
            
            # Decode
            reconstructed = decoder(encoded, ids_restore)
            
            # Compute reconstruction loss
            # Patchify original image
            patches = images.unfold(2, 16, 16).unfold(3, 16, 16)
            patches = patches.contiguous().view(batch_size, 3, num_patches, 16, 16)
            patches = patches.permute(0, 2, 3, 4, 1).contiguous()
            patches = patches.view(batch_size, num_patches, -1)
            
            # Loss only on masked patches
            loss = ((reconstructed - patches) ** 2).mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, checkpoint_path / 'mae_rvl_cdip_best.pth')
            print(f"  Saved best model (loss: {best_loss:.4f})")
    
    print("\n✅ Pretraining complete!")
    print(f"Best checkpoint saved to: {checkpoint_path / 'mae_rvl_cdip_best.pth'}")
    
    return encoder


def create_sample_rvl_cdip():
    """
    DEPRECATED: This project now uses real RVL-CDIP data only.
    See setup_datasets.py --rvl-cdip for download instructions.
    """
    print("⚠️  Synthetic sample generation has been removed.")
    print("    Download the real RVL-CDIP dataset instead:")
    print("      python setup_datasets.py --rvl-cdip")
    return None


def main():
    parser = argparse.ArgumentParser(description='RVL-CDIP Dataset Integration')
    parser.add_argument('--download', action='store_true', help='Download RVL-CDIP from HuggingFace')
    parser.add_argument('--prepare', action='store_true', help='Prepare images for MAE pretraining')
    parser.add_argument('--pretrain', action='store_true', help='Run MAE pretraining')
    parser.add_argument('--data-dir', type=str, default='data/rvl_cdip', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='data/rvl_cdip_prepared', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Pretraining epochs')
    
    args = parser.parse_args()
    
    if args.download:
        download_rvl_cdip(args.data_dir)
    elif args.prepare:
        prepare_for_mae_pretraining(args.data_dir, args.output_dir)
    elif args.pretrain:
        if not Path(args.output_dir).exists() or not any(Path(args.output_dir).rglob('*.png')):
            print(f"❌ No data found in {args.output_dir}")
            print("   Download and prepare first:")
            print("     python data_integration/rvl_cdip_loader.py --download")
            print("     python data_integration/rvl_cdip_loader.py --prepare")
            return
        pretrain_vit_with_rvl_cdip(args.output_dir, 'checkpoints', args.epochs)
    else:
        print("RVL-CDIP Dataset Integration (real data only)")
        print("\nCommands:")
        print("  --download   Download from HuggingFace (~40GB)")
        print("  --prepare    Resize & prepare for MAE pretraining")
        print("  --pretrain   Run MAE self-supervised pretraining")


if __name__ == '__main__':
    main()
