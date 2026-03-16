"""
CEDAR Signature Dataset Integration
====================================
Loads and processes CEDAR signature dataset for signature 
verification in bank fraud detection.

Dataset: 55 signers × (24 genuine + 24 forged) = 2,640 signatures

Usage:
    python data_integration/cedar_signature_loader.py --download-info
    python data_integration/cedar_signature_loader.py --prepare
    python data_integration/cedar_signature_loader.py --train
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))


class CEDARSignatureDataset(Dataset):
    """
    CEDAR Signature Dataset for genuine vs forged classification.
    
    Directory structure expected:
        cedar_signatures/
        ├── genuine/
        │   ├── original_1_1.png
        │   ├── original_1_2.png
        │   └── ...
        └── forged/
            ├── forgeries_1_1.png
            ├── forgeries_1_2.png
            └── ...
    """
    
    CLASSES = ['genuine', 'forged']
    
    def __init__(self, data_dir: str, transform=None, writer_ids: Optional[List[int]] = None):
        """
        Args:
            data_dir: Path to CEDAR dataset
            transform: Image transforms
            writer_ids: Specific writer IDs to include (for train/test split)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or self._default_transform()
        self.writer_ids = writer_ids
        
        self.samples = self._load_samples()
        
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_samples(self) -> List[Tuple[str, int, int]]:
        """Load (image_path, label, writer_id) tuples."""
        samples = []
        
        for label, class_name in enumerate(self.CLASSES):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_file in class_dir.glob('*.*'):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']:
                    # Extract writer ID from filename (e.g., original_1_1.png -> writer 1)
                    try:
                        parts = img_file.stem.split('_')
                        writer_id = int(parts[1])
                    except:
                        writer_id = 0
                    
                    # Filter by writer_ids if specified
                    if self.writer_ids is None or writer_id in self.writer_ids:
                        samples.append((str(img_file), label, writer_id))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, writer_id = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, writer_id


class SignatureVerificationViT(nn.Module):
    """
    Vision Transformer for Signature Verification.
    Binary classification: genuine (0) vs forged (1)
    """
    
    def __init__(self, image_size=224, patch_size=16, embed_dim=192, 
                 num_heads=3, depth=6, mlp_dim=384):
        super().__init__()
        
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head (binary)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 2)  # Binary: genuine vs forged
        )
        
    def forward(self, x):
        B = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x


def create_sample_cedar_data(output_dir: str, num_writers: int = 10):
    """
    DEPRECATED: This project now uses real CEDAR data only.
    See setup_datasets.py --cedar for download instructions.
    """
    print("⚠️  Synthetic sample generation has been removed.")
    print("    Download the real CEDAR dataset instead:")
    print("      python setup_datasets.py --cedar")
    print("    Then place images in data/cedar_signatures/genuine/ and data/cedar_signatures/forged/")
    return None
def train_signature_vit(data_dir: str, checkpoint_dir: str = 'checkpoints', 
                        epochs: int = 30, batch_size: int = 16):
    """
    Train a ViT model for signature verification.
    
    Args:
        data_dir: Path to signature dataset
        checkpoint_dir: Path to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
    """
    print("=" * 60)
    print("Training Signature Verification ViT")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get unique writer IDs for train/val split
    full_dataset = CEDARSignatureDataset(data_dir)
    writer_ids = list(set([s[2] for s in full_dataset.samples]))
    np.random.shuffle(writer_ids)
    
    # 80/20 writer split (writer-independent validation)
    split_idx = int(len(writer_ids) * 0.8)
    train_writers = writer_ids[:split_idx]
    val_writers = writer_ids[split_idx:]
    
    print(f"Training writers: {len(train_writers)}")
    print(f"Validation writers: {len(val_writers)}")
    
    # Create datasets
    train_dataset = CEDARSignatureDataset(data_dir, train_transform, train_writers)
    val_dataset = CEDARSignatureDataset(data_dir, val_transform, val_writers)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = SignatureVerificationViT(
        image_size=224,
        patch_size=16,
        embed_dim=192,
        num_heads=3,
        depth=6,
        mlp_dim=384
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.1f}%, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': best_val_acc,
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path / 'signature_vit_best.pth')
            print(f"  ✅ Saved best model (Val Acc: {best_val_acc:.1f}%)")
    
    # Save training history
    with open(checkpoint_path / 'signature_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Checkpoint saved to: {checkpoint_path / 'signature_vit_best.pth'}")
    print("=" * 60)
    
    return model, history


def extract_signature_from_cheque(cheque_image: Image.Image) -> Image.Image:
    """
    Extract signature region from a cheque image.
    
    In production, use object detection or template matching.
    This is a simplified version using fixed region extraction.
    
    Args:
        cheque_image: Full cheque image
        
    Returns:
        Cropped signature region
    """
    width, height = cheque_image.size
    
    # Typical signature location: bottom-right quadrant
    # Adjust these percentages based on your cheque format
    left = int(width * 0.55)
    top = int(height * 0.65)
    right = int(width * 0.95)
    bottom = int(height * 0.90)
    
    signature = cheque_image.crop((left, top, right, bottom))
    return signature


def verify_signature(model, image_path: str, device='cpu') -> Dict:
    """
    Verify if a signature is genuine or forged.
    
    Args:
        model: Trained SignatureVerificationViT
        image_path: Path to signature image
        device: torch device
        
    Returns:
        Dictionary with prediction and confidence
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    img_transformed = transform(image)
    image_tensor = img_transformed.unsqueeze(0).to(device)  # type: ignore[union-attr]
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    return {
        'prediction': 'genuine' if pred_class == 0 else 'forged',
        'confidence': confidence,
        'genuine_prob': probs[0, 0].item(),
        'forged_prob': probs[0, 1].item()
    }


def download_info():
    """Print download instructions for CEDAR dataset."""
    print("=" * 60)
    print("CEDAR Signature Dataset Download Instructions")
    print("=" * 60)
    
    print("""
The CEDAR signature dataset is available from:
  https://www.cedar.buffalo.edu/NIJ/data/

⚠️ IMPORTANT: This is a research dataset. You may need to:
  1. Fill out a request form
  2. Agree to usage terms
  3. Wait for approval

Alternative datasets:
  - GPDS Synthetic: https://www.gpds.ulpgc.es/
  - ICDAR 2011 Signature: https://www.imag.fr/~icdar2011/
  - BHSig260: Hand-drawn Bengali/Hindi signatures

After downloading, organize the data as:
  data/cedar_signatures/
  ├── genuine/
  │   ├── original_1_1.png
  │   ├── original_1_2.png
  │   └── ...
  └── forged/
      ├── forgeries_1_1.png
      ├── forgeries_1_2.png
      └── ...

For testing without the real dataset, use:
  python data_integration/cedar_signature_loader.py --create-sample
""")


def main():
    parser = argparse.ArgumentParser(description='CEDAR Signature Dataset Integration')
    parser.add_argument('--download-info', action='store_true', help='Show download instructions')
    parser.add_argument('--train', action='store_true', help='Train signature verification model')
    parser.add_argument('--data-dir', type=str, default='data/cedar_signatures', help='Data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    if args.download_info:
        download_info()
    elif args.train:
        if not Path(args.data_dir).exists() or not any(Path(args.data_dir).rglob('*.png')):
            print(f"❌ No real data found in: {args.data_dir}")
            print("   Download CEDAR dataset first:")
            download_info()
            return
        train_signature_vit(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
    else:
        download_info()


if __name__ == '__main__':
    main()
