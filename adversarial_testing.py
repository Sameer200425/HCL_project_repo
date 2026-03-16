"""
Adversarial Attack Testing Module
==================================
Tests model robustness against adversarial attacks.
Important for cybersecurity-level fraud detection research.

Attack Types:
1. FGSM (Fast Gradient Sign Method)
2. PGD (Projected Gradient Descent)
3. Noise Injection
4. Patch Attacks
5. Rotation/Scaling Attacks

Usage:
    python adversarial_testing.py --demo
    python adversarial_testing.py --attack fgsm --model cnn
    python adversarial_testing.py --full-report
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))


class AdversarialAttacks:
    """Collection of adversarial attack methods."""
    
    @staticmethod
    def fgsm_attack(model: nn.Module, image: torch.Tensor, 
                    label: torch.Tensor, epsilon: float = 0.03,
                    device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            model: Target model
            image: Input image tensor
            label: True label
            epsilon: Perturbation magnitude
            device: Device to use
            
        Returns:
            Adversarial image
        """
        image = image.to(device)
        label = label.to(device)
        
        image.requires_grad = True
        
        output = model(image)
        loss = F.cross_entropy(output, label)
        
        model.zero_grad()
        loss.backward()
        
        # Get sign of gradients
        if image.grad is None:
            raise RuntimeError("No gradient computed for input image")
        data_grad = image.grad.data
        sign_data_grad = data_grad.sign()
        
        # Create perturbed image
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image.detach()
    
    @staticmethod
    def pgd_attack(model: nn.Module, image: torch.Tensor,
                   label: torch.Tensor, epsilon: float = 0.03,
                   alpha: float = 0.007, num_steps: int = 10,
                   device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack.
        Stronger iterative version of FGSM.
        
        Args:
            model: Target model
            image: Input image tensor
            label: True label
            epsilon: Maximum perturbation
            alpha: Step size
            num_steps: Number of iterations
            device: Device to use
            
        Returns:
            Adversarial image
        """
        image = image.to(device)
        label = label.to(device)
        original_image = image.clone()
        
        # Random start within epsilon ball
        perturbed_image = image + torch.empty_like(image).uniform_(-epsilon, epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        for _ in range(num_steps):
            perturbed_image.requires_grad = True
            
            output = model(perturbed_image)
            loss = F.cross_entropy(output, label)
            
            model.zero_grad()
            loss.backward()
            
            # Update adversarial image
            if perturbed_image.grad is None:
                raise RuntimeError("No gradient computed for perturbed image")
            data_grad = perturbed_image.grad.data
            perturbed_image = perturbed_image + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
            perturbed_image = torch.clamp(original_image + perturbation, 0, 1).detach()
        
        return perturbed_image
    
    @staticmethod
    def gaussian_noise_attack(image: torch.Tensor, 
                               sigma: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(image) * sigma
        perturbed = image + noise
        return torch.clamp(perturbed, 0, 1)
    
    @staticmethod
    def patch_attack(image: torch.Tensor, 
                     patch_size: int = 32,
                     position: str = 'random') -> torch.Tensor:
        """
        Add adversarial patch to image.
        
        Args:
            image: Input image (C, H, W)
            patch_size: Size of patch
            position: 'random', 'center', or 'corner'
        """
        perturbed = image.clone()
        _, h, w = image.shape[-3:]
        
        # Determine position
        if position == 'center':
            y = (h - patch_size) // 2
            x = (w - patch_size) // 2
        elif position == 'corner':
            y, x = 0, 0
        else:  # random
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)
        
        # Create random patch
        patch = torch.rand(3, patch_size, patch_size)
        
        if perturbed.dim() == 4:
            perturbed[:, :, y:y+patch_size, x:x+patch_size] = patch
        else:
            perturbed[:, y:y+patch_size, x:x+patch_size] = patch
        
        return perturbed
    
    @staticmethod
    def rotation_attack(image: torch.Tensor, 
                        angle: float = 15) -> torch.Tensor:
        """Apply rotation attack."""
        # Convert to PIL, rotate, convert back
        if image.dim() == 4:
            image = image[0]
        
        # Denormalize if needed
        img_np = image.permute(1, 2, 0).numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img_np)
        rotated = pil_img.rotate(angle, fillcolor=(255, 255, 255))
        
        # Convert back to tensor
        rotated_np = np.array(rotated).astype(np.float32) / 255.0
        rotated_tensor = torch.from_numpy(rotated_np).permute(2, 0, 1)
        
        return rotated_tensor.unsqueeze(0) if image.dim() == 4 else rotated_tensor


class AdversarialEvaluator:
    """Evaluates model robustness against adversarial attacks."""
    
    def __init__(self, model: nn.Module, device: Union[str, torch.device] = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.attacks = AdversarialAttacks()
    
    def evaluate_attack(self, dataloader: DataLoader, 
                        attack_fn, attack_name: str,
                        **attack_kwargs) -> Dict:
        """
        Evaluate model against specific attack.
        
        Args:
            dataloader: Test data loader
            attack_fn: Attack function
            attack_name: Name of attack
            **attack_kwargs: Attack parameters
            
        Returns:
            Dictionary with results
        """
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        clean_confidences = []
        adv_confidences = []
        perturbation_norms = []
        
        for images, labels in tqdm(dataloader, desc=f"Testing {attack_name}"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Clean prediction
            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_probs = F.softmax(clean_outputs, dim=1)
                clean_preds = clean_outputs.argmax(dim=1)
                clean_correct += (clean_preds == labels).sum().item()
                clean_confidences.extend(clean_probs.max(dim=1)[0].cpu().numpy())
            
            # Generate adversarial examples
            if 'model' in attack_fn.__code__.co_varnames:
                # Gradient-based attack
                adv_images = attack_fn(
                    model=self.model,
                    image=images,
                    label=labels,
                    device=self.device,
                    **attack_kwargs
                )
            else:
                # Simple perturbation attack
                adv_images = attack_fn(images, **attack_kwargs)
                adv_images = adv_images.to(self.device)
            
            # Adversarial prediction
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_probs = F.softmax(adv_outputs, dim=1)
                adv_preds = adv_outputs.argmax(dim=1)
                adv_correct += (adv_preds == labels).sum().item()
                adv_confidences.extend(adv_probs.max(dim=1)[0].cpu().numpy())
            
            # Calculate perturbation
            perturbation = (adv_images - images).view(images.size(0), -1)
            perturbation_norms.extend(perturbation.norm(dim=1).cpu().numpy())
            
            total += labels.size(0)
        
        clean_acc = clean_correct / total
        adv_acc = adv_correct / total
        
        return {
            'attack_name': attack_name,
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'accuracy_drop': clean_acc - adv_acc,
            'attack_success_rate': 1 - (adv_acc / clean_acc) if clean_acc > 0 else 0,
            'avg_clean_confidence': np.mean(clean_confidences),
            'avg_adv_confidence': np.mean(adv_confidences),
            'avg_perturbation_norm': np.mean(perturbation_norms),
            'total_samples': total
        }
    
    def run_full_evaluation(self, dataloader: DataLoader) -> Dict:
        """Run evaluation with all attack types."""
        results = {}
        
        # FGSM attacks with different epsilon
        for epsilon in [0.01, 0.03, 0.05, 0.1]:
            result = self.evaluate_attack(
                dataloader,
                self.attacks.fgsm_attack,
                f'FGSM (ε={epsilon})',
                epsilon=epsilon
            )
            results[f'fgsm_eps{epsilon}'] = result
        
        # PGD attacks
        for epsilon in [0.01, 0.03]:
            result = self.evaluate_attack(
                dataloader,
                self.attacks.pgd_attack,
                f'PGD (ε={epsilon})',
                epsilon=epsilon,
                alpha=epsilon/4,
                num_steps=10
            )
            results[f'pgd_eps{epsilon}'] = result
        
        # Noise attacks
        for sigma in [0.05, 0.1, 0.2]:
            result = self.evaluate_attack(
                dataloader,
                self.attacks.gaussian_noise_attack,
                f'Noise (σ={sigma})',
                sigma=sigma
            )
            results[f'noise_sigma{sigma}'] = result
        
        # Patch attacks
        for patch_size in [16, 32, 48]:
            result = self.evaluate_attack(
                dataloader,
                self.attacks.patch_attack,
                f'Patch ({patch_size}x{patch_size})',
                patch_size=patch_size
            )
            results[f'patch_{patch_size}'] = result
        
        # Rotation attacks
        for angle in [5, 10, 15, 30]:
            result = self.evaluate_attack(
                dataloader,
                lambda img, angle=angle: self.attacks.rotation_attack(img, angle),
                f'Rotation ({angle}°)'
            )
            results[f'rotation_{angle}'] = result
        
        # Calculate summary stats
        results['summary'] = self._calculate_summary(results)
        
        return results
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics."""
        valid_results = [v for k, v in results.items() 
                        if isinstance(v, dict) and 'adversarial_accuracy' in v]
        
        if not valid_results:
            return {}
        
        adv_accs = [r['adversarial_accuracy'] for r in valid_results]
        acc_drops = [r['accuracy_drop'] for r in valid_results]
        
        return {
            'mean_adversarial_accuracy': np.mean(adv_accs),
            'min_adversarial_accuracy': np.min(adv_accs),
            'max_accuracy_drop': np.max(acc_drops),
            'mean_accuracy_drop': np.mean(acc_drops),
            'most_effective_attack': max(valid_results, 
                                         key=lambda x: x['accuracy_drop'])['attack_name'],
            'robustness_score': np.mean(adv_accs) / valid_results[0]['clean_accuracy']
        }


def generate_adversarial_report(results: Dict, output_path: str = 'reports/adversarial_report.txt'):
    """Generate detailed adversarial testing report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("=" * 70)
    report.append("ADVERSARIAL ROBUSTNESS EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Summary
    if 'summary' in results:
        s = results['summary']
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Mean Adversarial Accuracy:  {s['mean_adversarial_accuracy']*100:.2f}%")
        report.append(f"Worst Case Accuracy:        {s['min_adversarial_accuracy']*100:.2f}%")
        report.append(f"Maximum Accuracy Drop:      {s['max_accuracy_drop']*100:.2f}%")
        report.append(f"Robustness Score:           {s['robustness_score']:.2f}")
        report.append(f"Most Effective Attack:      {s['most_effective_attack']}")
        report.append("")
        
        # Robustness assessment
        if s['robustness_score'] >= 0.8:
            report.append("✅ Model shows STRONG adversarial robustness")
        elif s['robustness_score'] >= 0.6:
            report.append("⚠️ Model shows MODERATE adversarial robustness")
        else:
            report.append("❌ Model is VULNERABLE to adversarial attacks")
        report.append("")
    
    # Detailed results by attack type
    report.append("DETAILED ATTACK RESULTS")
    report.append("-" * 40)
    
    attack_categories = {
        'Gradient-Based (FGSM)': [k for k in results if k.startswith('fgsm')],
        'Iterative (PGD)': [k for k in results if k.startswith('pgd')],
        'Noise Injection': [k for k in results if k.startswith('noise')],
        'Patch Attacks': [k for k in results if k.startswith('patch')],
        'Geometric': [k for k in results if k.startswith('rotation')]
    }
    
    for category, keys in attack_categories.items():
        if not keys:
            continue
            
        report.append(f"\n{category}:")
        for key in sorted(keys):
            r = results[key]
            if isinstance(r, dict) and 'adversarial_accuracy' in r:
                report.append(
                    f"  {r['attack_name']:25s}: "
                    f"Clean={r['clean_accuracy']*100:5.1f}% → "
                    f"Adv={r['adversarial_accuracy']*100:5.1f}% "
                    f"(↓{r['accuracy_drop']*100:.1f}%)"
                )
    
    # Security recommendations
    report.append("\n" + "=" * 70)
    report.append("SECURITY RECOMMENDATIONS")
    report.append("-" * 40)
    
    recommendations = []
    
    if 'summary' in results:
        s = results['summary']
        
        if s['robustness_score'] < 0.6:
            recommendations.append("1. Implement adversarial training to improve robustness")
            recommendations.append("2. Consider input preprocessing/denoising")
            recommendations.append("3. Add ensemble of diverse models")
        
        if any(r.get('accuracy_drop', 0) > 0.3 for k, r in results.items() 
               if isinstance(r, dict)):
            recommendations.append("4. Implement adversarial detection mechanism")
            recommendations.append("5. Add confidence thresholding before final decision")
        
        # Check specific vulnerabilities
        fgsm_results = [r for k, r in results.items() if k.startswith('fgsm')]
        if fgsm_results and max(r.get('accuracy_drop', 0) for r in fgsm_results) > 0.2:
            recommendations.append("6. Model is vulnerable to gradient-based attacks - consider TRADES or PGD training")
        
        patch_results = [r for k, r in results.items() if k.startswith('patch')]
        if patch_results and max(r.get('accuracy_drop', 0) for r in patch_results) > 0.15:
            recommendations.append("7. Model is vulnerable to patch attacks - implement spatial attention constraints")
    
    if recommendations:
        for rec in recommendations:
            report.append(rec)
    else:
        report.append("Model shows adequate robustness. Continue monitoring for new attack vectors.")
    
    report.append("\n" + "=" * 70)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Report saved to: {output_path}")
    
    return '\n'.join(report)


def visualize_adversarial_examples(model: nn.Module, dataloader: DataLoader,
                                   output_dir: str = 'results/adversarial_samples',
                                   device: Union[str, torch.device] = 'cpu'):
    """Generate visualization of adversarial examples."""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    attacks = AdversarialAttacks()
    
    # Get sample image
    images, labels = next(iter(dataloader))
    image, label = images[0:1].to(device), labels[0:1].to(device)
    
    attack_configs = [
        ('Clean', lambda x: x),
        ('FGSM ε=0.03', lambda x: attacks.fgsm_attack(model, x, label, 0.03, device)),
        ('FGSM ε=0.1', lambda x: attacks.fgsm_attack(model, x, label, 0.1, device)),
        ('PGD', lambda x: attacks.pgd_attack(model, x, label, 0.03, device=device)),
        ('Noise σ=0.1', lambda x: attacks.gaussian_noise_attack(x, 0.1)),
        ('Patch 32x32', lambda x: attacks.patch_attack(x, 32)),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    class_names = ['genuine', 'fraud', 'tampered', 'forged']
    
    for idx, (name, attack_fn) in enumerate(attack_configs):
        adv_image = attack_fn(image.clone())
        
        with torch.no_grad():
            output = model(adv_image.to(device))
            pred = output.argmax(dim=1).item()
            conf = F.softmax(output, dim=1).max().item()
        
        # Denormalize for display
        img_display = adv_image[0].cpu().permute(1, 2, 0).numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
        axes[idx].imshow(img_display)
        axes[idx].set_title(f'{name}\nPred: {class_names[pred]} ({conf:.1%})')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'adversarial_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Adversarial examples saved to: {output_path / 'adversarial_examples.png'}")


def main():
    parser = argparse.ArgumentParser(description='Adversarial Attack Testing')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'hybrid', 'vit'])
    parser.add_argument('--attack', type=str, default='all',
                        choices=['fgsm', 'pgd', 'noise', 'patch', 'rotation', 'all'])
    parser.add_argument('--data-dir', type=str, default='data/fraud_dataset')
    parser.add_argument('--full-report', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading {args.model} model...")
    
    if args.model == 'cnn':
        from models.hybrid_model import CNNBaseline
        model = CNNBaseline(num_classes=4, pretrained=False)
        checkpoint_path = 'checkpoints/cnn_best.pth'
    elif args.model == 'hybrid':
        from models.hybrid_model import HybridCNNViT
        model = HybridCNNViT(num_classes=4)
        checkpoint_path = 'checkpoints/hybrid_best.pth'
    else:
        from models.vit_model import VisionTransformer
        model = VisionTransformer(num_classes=4)
        checkpoint_path = 'checkpoints/vit_best.pth'
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("Running in demo mode with random weights...")
    
    model.to(device)
    model.eval()
    
    # Create dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        print("Creating synthetic test data...")
        # Create minimal test data
        data_path.mkdir(parents=True, exist_ok=True)
        for cls in ['genuine', 'fraud', 'tampered', 'forged']:
            cls_dir = data_path / cls
            cls_dir.mkdir(exist_ok=True)
            for i in range(10):
                img = Image.new('RGB', (224, 224), color=tuple(np.random.randint(0, 255, 3)))
                img.save(cls_dir / f'{i}.png')
    
    dataset = datasets.ImageFolder(str(data_path), transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create evaluator
    evaluator = AdversarialEvaluator(model, device)
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("ADVERSARIAL ATTACK EVALUATION")
    print("=" * 60)
    
    results = evaluator.run_full_evaluation(dataloader)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if 'summary' in results:
        s = results['summary']
        print(f"Mean Adversarial Accuracy: {s['mean_adversarial_accuracy']*100:.1f}%")
        print(f"Worst Case Accuracy:       {s['min_adversarial_accuracy']*100:.1f}%")
        print(f"Robustness Score:          {s['robustness_score']:.2f}")
        print(f"Most Effective Attack:     {s['most_effective_attack']}")
    
    # Save results
    results_path = Path('results/adversarial_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Generate report
    if args.full_report:
        generate_adversarial_report(results)
    
    # Visualize
    if args.visualize:
        visualize_adversarial_examples(model, dataloader, device=device)


if __name__ == '__main__':
    main()
