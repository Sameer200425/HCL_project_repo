"""
Robustness Testing Module
==========================
Tests model performance on challenging conditions to validate
that high accuracy is not due to overfitting or pattern leakage.

Tests Include:
- Noise injection (Gaussian, Salt & Pepper)
- Blur (Gaussian, Motion)
- Rotation and geometric transforms
- JPEG compression artifacts
- Brightness/Contrast variations
- Real-world simulation (phone camera, scanner artifacts)

Usage:
    python robustness_testing.py --model cnn
    python robustness_testing.py --model hybrid
    python robustness_testing.py --full-report
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))


class RobustnessTransforms:
    """Collection of robustness test transforms."""
    
    @staticmethod
    def gaussian_noise(image: Image.Image, severity: float = 0.1) -> Image.Image:
        """Add Gaussian noise."""
        img_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, severity, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    @staticmethod
    def salt_pepper_noise(image: Image.Image, amount: float = 0.05) -> Image.Image:
        """Add salt and pepper noise."""
        img_array = np.array(image)
        # Salt
        salt_mask = np.random.random(img_array.shape[:2]) < amount / 2
        img_array[salt_mask] = 255
        # Pepper
        pepper_mask = np.random.random(img_array.shape[:2]) < amount / 2
        img_array[pepper_mask] = 0
        return Image.fromarray(img_array)
    
    @staticmethod
    def gaussian_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
        """Apply Gaussian blur."""
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def motion_blur(image: Image.Image, size: int = 15) -> Image.Image:
        """Simulate motion blur."""
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size
        from PIL import ImageFilter
        # Simplified motion blur using box blur
        return image.filter(ImageFilter.BoxBlur(radius=size // 3))
    
    @staticmethod
    def jpeg_compression(image: Image.Image, quality: int = 10) -> Image.Image:
        """Simulate JPEG compression artifacts."""
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    
    @staticmethod
    def rotation(image: Image.Image, angle: float = 15) -> Image.Image:
        """Rotate image."""
        return image.rotate(angle, fillcolor=(255, 255, 255))
    
    @staticmethod
    def perspective_transform(image: Image.Image, magnitude: float = 0.1) -> Image.Image:
        """Apply perspective transformation."""
        width, height = image.size
        m = magnitude * min(width, height)
        
        # Random perspective points
        coeffs = [
            np.random.uniform(-m, m), np.random.uniform(-m, m),
            width + np.random.uniform(-m, m), np.random.uniform(-m, m),
            width + np.random.uniform(-m, m), height + np.random.uniform(-m, m),
            np.random.uniform(-m, m), height + np.random.uniform(-m, m)
        ]
        
        return image.transform((width, height), Image.Transform.QUAD, coeffs, Image.Resampling.BILINEAR)
    
    @staticmethod
    def brightness_change(image: Image.Image, factor: float = 0.5) -> Image.Image:
        """Change brightness."""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def contrast_change(image: Image.Image, factor: float = 0.5) -> Image.Image:
        """Change contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def phone_camera_simulation(image: Image.Image) -> Image.Image:
        """Simulate phone camera capture (blur + noise + compression)."""
        # Add slight blur
        img = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        # Add noise
        img = RobustnessTransforms.gaussian_noise(img, severity=0.03)
        # JPEG compression
        img = RobustnessTransforms.jpeg_compression(img, quality=70)
        return img
    
    @staticmethod
    def scanner_simulation(image: Image.Image) -> Image.Image:
        """Simulate scanned document artifacts."""
        # Slight rotation (misalignment)
        angle = np.random.uniform(-2, 2)
        img = image.rotate(angle, fillcolor=(255, 255, 255))
        # Slight blur
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        # Brightness variation
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(np.random.uniform(0.9, 1.1))
        return img


class RobustnessDataset(Dataset):
    """Dataset with robustness transformations applied."""
    
    CORRUPTION_TYPES = [
        ('clean', None),
        ('gaussian_noise_low', lambda img: RobustnessTransforms.gaussian_noise(img, 0.05)),
        ('gaussian_noise_med', lambda img: RobustnessTransforms.gaussian_noise(img, 0.1)),
        ('gaussian_noise_high', lambda img: RobustnessTransforms.gaussian_noise(img, 0.2)),
        ('salt_pepper_low', lambda img: RobustnessTransforms.salt_pepper_noise(img, 0.02)),
        ('salt_pepper_high', lambda img: RobustnessTransforms.salt_pepper_noise(img, 0.1)),
        ('blur_low', lambda img: RobustnessTransforms.gaussian_blur(img, 1.0)),
        ('blur_med', lambda img: RobustnessTransforms.gaussian_blur(img, 2.0)),
        ('blur_high', lambda img: RobustnessTransforms.gaussian_blur(img, 4.0)),
        ('motion_blur', lambda img: RobustnessTransforms.motion_blur(img, 10)),
        ('jpeg_q50', lambda img: RobustnessTransforms.jpeg_compression(img, 50)),
        ('jpeg_q20', lambda img: RobustnessTransforms.jpeg_compression(img, 20)),
        ('jpeg_q10', lambda img: RobustnessTransforms.jpeg_compression(img, 10)),
        ('rotation_5', lambda img: RobustnessTransforms.rotation(img, 5)),
        ('rotation_15', lambda img: RobustnessTransforms.rotation(img, 15)),
        ('rotation_30', lambda img: RobustnessTransforms.rotation(img, 30)),
        ('bright_low', lambda img: RobustnessTransforms.brightness_change(img, 0.5)),
        ('bright_high', lambda img: RobustnessTransforms.brightness_change(img, 1.5)),
        ('contrast_low', lambda img: RobustnessTransforms.contrast_change(img, 0.5)),
        ('contrast_high', lambda img: RobustnessTransforms.contrast_change(img, 1.5)),
        ('phone_camera', RobustnessTransforms.phone_camera_simulation),
        ('scanner', RobustnessTransforms.scanner_simulation),
    ]
    
    def __init__(self, data_dir: str, corruption_type: str = 'clean'):
        self.data_dir = Path(data_dir)
        self.corruption_type = corruption_type
        
        # Get corruption function
        self.corruption_fn = None
        for name, fn in self.CORRUPTION_TYPES:
            if name == corruption_type:
                self.corruption_fn = fn
                break
        
        # Standard transform after corruption
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load samples
        self.samples = []
        self.class_names = ['genuine', 'fraud', 'tampered', 'forged']
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        # Apply corruption
        if self.corruption_fn is not None:
            image = self.corruption_fn(image)
        
        # Apply standard transform
        image = self.transform(image)
        
        return image, label


def evaluate_robustness(model, data_dir: str, device: Union[str, torch.device] = 'cpu', 
                        save_dir: str = 'results') -> Dict:
    """
    Evaluate model robustness across all corruption types.
    
    Args:
        model: Trained PyTorch model
        data_dir: Path to test dataset
        device: Device to use
        save_dir: Directory to save results
        
    Returns:
        Dictionary with results per corruption type
    """
    print("=" * 70)
    print("ROBUSTNESS EVALUATION")
    print("=" * 70)
    print("Testing model resilience to real-world image corruptions")
    print("-" * 70)
    
    model.eval()
    model.to(device)
    
    results = {}
    
    for corruption_name, _ in tqdm(RobustnessDataset.CORRUPTION_TYPES, desc="Testing corruptions"):
        dataset = RobustnessDataset(data_dir, corruption_type=corruption_name)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        correct = 0
        total = 0
        confidences = []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Track confidence
                max_probs = probs.max(dim=1)[0]
                confidences.extend(max_probs.cpu().numpy().tolist())
        
        accuracy = correct / total if total > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        results[corruption_name] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_confidence': avg_confidence
        }
        
        # Print progress
        emoji = "✅" if accuracy > 0.9 else ("⚠️" if accuracy > 0.7 else "❌")
        print(f"  {emoji} {corruption_name:20s}: {accuracy*100:5.1f}% (conf: {avg_confidence:.3f})")
    
    # Calculate summary statistics
    accuracies = [r['accuracy'] for r in results.values()]
    results['summary'] = {
        'mean_accuracy': np.mean(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'std_accuracy': np.std(accuracies),
        'clean_accuracy': results['clean']['accuracy'],
        'robustness_gap': results['clean']['accuracy'] - np.mean(accuracies[1:])  # Gap from corrupted
    }
    
    # Save results
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(save_path / 'robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(f"  Clean Accuracy:     {results['summary']['clean_accuracy']*100:.1f}%")
    print(f"  Mean Corrupted:     {results['summary']['mean_accuracy']*100:.1f}%")
    print(f"  Robustness Gap:     {results['summary']['robustness_gap']*100:.1f}%")
    print(f"  Min Accuracy:       {results['summary']['min_accuracy']*100:.1f}%")
    print(f"  Std Deviation:      {results['summary']['std_accuracy']*100:.1f}%")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    if results['summary']['robustness_gap'] < 0.05:
        print("  ✅ Model is ROBUST - minimal accuracy drop under corruptions")
    elif results['summary']['robustness_gap'] < 0.15:
        print("  ⚠️ Model has MODERATE robustness - some sensitivity to corruptions")
    else:
        print("  ❌ Model is FRAGILE - significant accuracy drop under corruptions")
        print("     Consider: data augmentation, adversarial training, or ensemble methods")
    
    return results


def generate_robustness_report(results: Dict, output_path: str = 'reports/robustness_report.txt'):
    """Generate detailed robustness report."""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("=" * 70)
    report.append("MODEL ROBUSTNESS EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Summary
    if 'summary' in results:
        s = results['summary']
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Clean Accuracy:        {s['clean_accuracy']*100:.2f}%")
        report.append(f"Average Corrupted:     {s['mean_accuracy']*100:.2f}%")
        report.append(f"Robustness Gap:        {s['robustness_gap']*100:.2f}%")
        report.append(f"Worst Case:            {s['min_accuracy']*100:.2f}%")
        report.append("")
    
    # Detailed results by category
    categories = {
        'Noise': ['gaussian_noise_low', 'gaussian_noise_med', 'gaussian_noise_high', 
                  'salt_pepper_low', 'salt_pepper_high'],
        'Blur': ['blur_low', 'blur_med', 'blur_high', 'motion_blur'],
        'Compression': ['jpeg_q50', 'jpeg_q20', 'jpeg_q10'],
        'Geometric': ['rotation_5', 'rotation_15', 'rotation_30'],
        'Illumination': ['bright_low', 'bright_high', 'contrast_low', 'contrast_high'],
        'Real-World': ['phone_camera', 'scanner']
    }
    
    report.append("DETAILED ANALYSIS BY CATEGORY")
    report.append("-" * 40)
    
    for category, corruptions in categories.items():
        report.append(f"\n{category}:")
        for c in corruptions:
            if c in results:
                r = results[c]
                report.append(f"  {c:25s}: {r['accuracy']*100:5.1f}% (n={r['total']})")
    
    # Recommendations
    report.append("\n" + "=" * 70)
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    
    weak_areas = []
    for c, r in results.items():
        if c not in ['summary', 'clean'] and r['accuracy'] < 0.8:
            weak_areas.append((c, r['accuracy']))
    
    if weak_areas:
        report.append("Model shows weakness in following areas:")
        for area, acc in sorted(weak_areas, key=lambda x: x[1]):
            report.append(f"  - {area}: {acc*100:.1f}%")
        report.append("\nSuggested mitigations:")
        report.append("  1. Add corresponding augmentations during training")
        report.append("  2. Consider ensemble with specialized models")
        report.append("  3. Apply preprocessing to normalize inputs")
    else:
        report.append("Model demonstrates strong robustness across all test categories.")
        report.append("The high accuracy is likely NOT due to overfitting.")
    
    report.append("\n" + "=" * 70)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ Report saved to: {output_path}")
    
    return '\n'.join(report)


def visualize_corruptions(image_path: str, output_dir: str = 'results/corruption_samples'):
    """Generate visualization of all corruption types on a sample image."""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    original = Image.open(image_path).convert('RGB')
    original = original.resize((224, 224))
    
    # Create grid visualization
    n_corruptions = len(RobustnessDataset.CORRUPTION_TYPES)
    cols = 5
    rows = (n_corruptions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()
    
    for idx, (name, fn) in enumerate(RobustnessDataset.CORRUPTION_TYPES):
        if fn is None:
            corrupted = original
        else:
            corrupted = fn(original.copy())
        
        axes[idx].imshow(corrupted)
        axes[idx].set_title(name, fontsize=8)
        axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(n_corruptions, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'corruption_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Corruption samples saved to: {output_path / 'corruption_samples.png'}")


def main():
    parser = argparse.ArgumentParser(description='Model Robustness Testing')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'hybrid', 'vit'],
                        help='Model to test')
    parser.add_argument('--data-dir', type=str, default='data/fraud_dataset/test',
                        help='Test data directory')
    parser.add_argument('--full-report', action='store_true', help='Generate full report')
    parser.add_argument('--visualize', action='store_true', help='Visualize corruptions')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading {args.model} model...")
    
    if args.model == 'cnn':
        from models.hybrid_model import CNNBaseline
        model = CNNBaseline(num_classes=4, pretrained=False)
        checkpoint = torch.load('checkpoints/cnn_best.pth', map_location=device)
    elif args.model == 'hybrid':
        from models.hybrid_model import HybridCNNViT
        model = HybridCNNViT(num_classes=4)
        checkpoint = torch.load('checkpoints/hybrid_best.pth', map_location=device)
    else:
        from models.vit_model import VisionTransformer
        model = VisionTransformer(num_classes=4)
        checkpoint = torch.load('checkpoints/vit_best.pth', map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        # Try alternate path
        data_dir = Path('data/fraud_dataset')
        if not data_dir.exists():
            print(f"Error: Data directory not found")
            return
    
    # Visualize corruptions
    if args.visualize:
        sample_images = list(data_dir.rglob('*.png'))
        if sample_images:
            visualize_corruptions(str(sample_images[0]))
    
    # Run robustness evaluation
    results = evaluate_robustness(model, str(data_dir), device)
    
    # Generate report
    if args.full_report:
        generate_robustness_report(results)


if __name__ == '__main__':
    main()
