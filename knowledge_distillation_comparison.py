"""
Knowledge Distillation Comparison Module
==========================================
Compares teacher vs student models for edge deployment scenarios.

This module demonstrates:
1. Teacher-student knowledge distillation
2. Model compression comparison
3. Accuracy vs efficiency trade-offs
4. Edge deployment recommendations

Usage:
    python knowledge_distillation_comparison.py --demo
    python knowledge_distillation_comparison.py --evaluate
    python knowledge_distillation_comparison.py --report
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

sys.path.append(str(Path(__file__).parent))


@dataclass
class ModelStats:
    """Statistics for a model."""
    name: str
    accuracy: float
    parameters: int
    model_size_mb: float
    inference_time_ms: float
    flops_estimate: int
    memory_usage_mb: float
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (accuracy per MB)."""
        return self.accuracy / self.model_size_mb * 100
    
    @property
    def latency_class(self) -> str:
        """Classify latency for deployment."""
        if self.inference_time_ms < 10:
            return "Real-time (<10ms)"
        elif self.inference_time_ms < 50:
            return "Interactive (<50ms)"
        elif self.inference_time_ms < 200:
            return "Batch (<200ms)"
        else:
            return "Offline"


class DistillationTrainer:
    """Knowledge distillation training utility."""
    
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 temperature: float = 3.0, alpha: float = 0.7,
                 device: str = 'cpu'):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        self.teacher.eval()
    
    def distillation_loss(self, student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate knowledge distillation loss.
        
        Combines:
        - Soft target loss (KL divergence with temperature)
        - Hard target loss (cross entropy with true labels)
        """
        # Soft targets (knowledge from teacher)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Hard targets (true labels)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Train student for one epoch with distillation."""
        self.student.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Teacher forward (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(images)
            
            # Student forward
            student_logits = self.student(images)
            
            # Calculate loss
            loss = self.distillation_loss(student_logits, teacher_logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def measure_inference_time(model: nn.Module, input_shape: Tuple[int, ...],
                           device: str = 'cpu', num_runs: int = 100) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return float(np.mean(times))


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Rough FLOP estimation based on layer types."""
    total_flops = 0
    
    def hook_fn(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Conv2d):
            # FLOPs = 2 * Cout * Hout * Wout * Cin * K * K
            _, c_out, h, w = output.shape
            c_in = module.in_channels
            k = module.kernel_size[0]
            total_flops += 2 * c_out * h * w * c_in * k * k
            
        elif isinstance(module, nn.Linear):
            # FLOPs = 2 * in_features * out_features
            total_flops += 2 * module.in_features * module.out_features
            
        elif isinstance(module, nn.MultiheadAttention):
            # Approximate for attention
            embed_dim = module.embed_dim
            seq_len = input[0].shape[1] if len(input[0].shape) > 2 else 1
            total_flops += 4 * embed_dim * embed_dim * seq_len
    
    hooks = []
    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook_fn))
    
    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return total_flops


def evaluate_model(model: nn.Module, dataloader: DataLoader,
                   name: str, device: str = 'cpu') -> ModelStats:
    """Comprehensive model evaluation."""
    model.eval()
    model.to(device)
    
    # Accuracy
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    
    # Model stats
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    inference_time = measure_inference_time(model, (1, 3, 224, 224), device)
    
    # FLOPS
    flops = estimate_flops(model, (1, 3, 224, 224))
    
    # Memory usage
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy)
        memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        memory = size_mb * 1.5  # Rough estimate
    
    return ModelStats(
        name=name,
        accuracy=accuracy,
        parameters=params,
        model_size_mb=size_mb,
        inference_time_ms=inference_time,
        flops_estimate=flops,
        memory_usage_mb=memory
    )


def create_comparison_table(models: List[ModelStats]) -> str:
    """Create markdown comparison table."""
    headers = [
        "Model", "Accuracy", "Params", "Size (MB)", 
        "Latency (ms)", "FLOPs", "Efficiency"
    ]
    
    rows = []
    for m in models:
        rows.append([
            m.name,
            f"{m.accuracy*100:.2f}%",
            f"{m.parameters/1e6:.2f}M",
            f"{m.model_size_mb:.2f}",
            f"{m.inference_time_ms:.1f}",
            f"{m.flops_estimate/1e9:.2f}G",
            f"{m.efficiency_score:.2f}"
        ])
    
    # Format table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    
    table = []
    
    # Header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    table.append(f"| {header_row} |")
    
    # Separator
    separator = " | ".join("-" * w for w in col_widths)
    table.append(f"| {separator} |")
    
    # Rows
    for row in rows:
        row_str = " | ".join(str(r).ljust(w) for r, w in zip(row, col_widths))
        table.append(f"| {row_str} |")
    
    return "\n".join(table)


def generate_comparison_report(models: List[ModelStats],
                               output_path: str = 'reports/distillation_comparison.md'):
    """Generate comprehensive comparison report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    report = []
    
    report.append("# Knowledge Distillation Model Comparison Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("This report compares teacher and student models for fraud detection,")
    report.append("evaluating the trade-offs between accuracy, model size, and inference speed")
    report.append("for potential edge deployment scenarios.")
    report.append("")
    
    # Find best models
    best_accuracy = max(models, key=lambda x: x.accuracy)
    best_efficiency = max(models, key=lambda x: x.efficiency_score)
    fastest = min(models, key=lambda x: x.inference_time_ms)
    smallest = min(models, key=lambda x: x.model_size_mb)
    
    report.append("### Key Findings")
    report.append("")
    report.append(f"- **Best Accuracy**: {best_accuracy.name} ({best_accuracy.accuracy*100:.2f}%)")
    report.append(f"- **Most Efficient**: {best_efficiency.name} ({best_efficiency.efficiency_score:.2f} acc/MB)")
    report.append(f"- **Fastest Inference**: {fastest.name} ({fastest.inference_time_ms:.1f}ms)")
    report.append(f"- **Smallest Model**: {smallest.name} ({smallest.model_size_mb:.2f}MB)")
    report.append("")
    
    # Comparison table
    report.append("## Model Comparison Table")
    report.append("")
    report.append(create_comparison_table(models))
    report.append("")
    
    # Detailed analysis
    report.append("## Detailed Analysis")
    report.append("")
    
    for model in models:
        report.append(f"### {model.name}")
        report.append("")
        report.append(f"- **Parameters**: {model.parameters:,}")
        report.append(f"- **Model Size**: {model.model_size_mb:.2f} MB")
        report.append(f"- **Accuracy**: {model.accuracy*100:.2f}%")
        report.append(f"- **Inference Time**: {model.inference_time_ms:.2f} ms")
        report.append(f"- **Latency Class**: {model.latency_class}")
        report.append(f"- **Estimated FLOPs**: {model.flops_estimate/1e9:.2f} GFLOPs")
        report.append(f"- **Efficiency Score**: {model.efficiency_score:.2f}")
        report.append("")
    
    # Deployment recommendations
    report.append("## Deployment Recommendations")
    report.append("")
    
    report.append("### Scenario 1: Cloud Server Deployment")
    report.append("")
    report.append(f"**Recommended**: {best_accuracy.name}")
    report.append("")
    report.append("- Priority: Maximum accuracy")
    report.append("- Resources: Unlimited GPU/CPU")
    report.append("- Use case: Batch processing of documents")
    report.append("")
    
    report.append("### Scenario 2: Edge Device / Mobile")
    report.append("")
    report.append(f"**Recommended**: {smallest.name}")
    report.append("")
    report.append("- Priority: Minimal footprint")
    report.append("- Resources: Limited memory (<1GB)")
    report.append("- Use case: On-device fraud detection")
    report.append("")
    
    report.append("### Scenario 3: Real-time API")
    report.append("")
    report.append(f"**Recommended**: {fastest.name}")
    report.append("")
    report.append("- Priority: Low latency")
    report.append("- Resources: Standard server")
    report.append("- Use case: Real-time transaction verification")
    report.append("")
    
    report.append("### Scenario 4: Balanced Production")
    report.append("")
    report.append(f"**Recommended**: {best_efficiency.name}")
    report.append("")
    report.append("- Priority: Best accuracy/efficiency trade-off")
    report.append("- Resources: Moderate")
    report.append("- Use case: Production fraud detection system")
    report.append("")
    
    # Compression analysis
    if len(models) >= 2:
        teacher = models[0]
        student = models[-1]
        
        report.append("## Compression Analysis")
        report.append("")
        report.append(f"**Teacher → Student Compression Results**")
        report.append("")
        report.append(f"- Size Reduction: {(1 - student.model_size_mb/teacher.model_size_mb)*100:.1f}%")
        report.append(f"- Parameter Reduction: {(1 - student.parameters/teacher.parameters)*100:.1f}%")
        report.append(f"- Speedup: {teacher.inference_time_ms/student.inference_time_ms:.2f}x")
        report.append(f"- Accuracy Drop: {(teacher.accuracy - student.accuracy)*100:.2f}%")
        report.append("")
        
        # Calculate compression ratio
        compression_ratio = teacher.model_size_mb / student.model_size_mb
        accuracy_retention = student.accuracy / teacher.accuracy
        
        report.append(f"- **Compression Ratio**: {compression_ratio:.2f}x")
        report.append(f"- **Accuracy Retention**: {accuracy_retention*100:.2f}%")
        report.append("")
        
        if accuracy_retention > 0.95 and compression_ratio > 2:
            report.append("✅ **Excellent distillation results!** Student maintains >95% accuracy with >2x compression.")
        elif accuracy_retention > 0.90:
            report.append("✅ **Good distillation results.** Student maintains >90% accuracy.")
        else:
            report.append("⚠️ **Moderate distillation results.** Consider tuning temperature/alpha parameters.")
    
    report.append("")
    report.append("---")
    report.append("*Generated by Knowledge Distillation Comparison Module*")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✅ Report saved to: {output_path}")
    
    return '\n'.join(report)


def create_visual_comparison(models: List[ModelStats], output_dir: str = 'results'):
    """Create visualizations comparing models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = [m.name for m in models]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63'][:len(models)]
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    accs = [m.accuracy * 100 for m in models]
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=10)
    
    # 2. Model size comparison
    ax = axes[0, 1]
    sizes = [m.model_size_mb for m in models]
    bars = ax.bar(names, sizes, color=colors)
    ax.set_ylabel('Size (MB)')
    ax.set_title('Model Size Comparison')
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{size:.1f}MB', ha='center', fontsize=10)
    
    # 3. Inference time comparison
    ax = axes[1, 0]
    times = [m.inference_time_ms for m in models]
    bars = ax.bar(names, times, color=colors)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Time Comparison')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{t:.1f}ms', ha='center', fontsize=10)
    
    # 4. Efficiency radar chart (simplified as bar)
    ax = axes[1, 1]
    effs = [m.efficiency_score for m in models]
    bars = ax.bar(names, effs, color=colors)
    ax.set_ylabel('Efficiency Score')
    ax.set_title('Model Efficiency (Accuracy/MB)')
    for bar, eff in zip(bars, effs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{eff:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path / 'model_comparison.png'}")
    
    # Create trade-off scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, m in enumerate(models):
        ax.scatter(m.model_size_mb, m.accuracy * 100, 
                   s=200, c=colors[i], label=m.name, alpha=0.7)
        ax.annotate(m.name, (m.model_size_mb, m.accuracy * 100),
                    xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Model Size Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_vs_size.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Trade-off plot saved to: {output_path / 'accuracy_vs_size.png'}")


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Comparison')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--data-dir', type=str, default='data/fraud_dataset')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing models')
    parser.add_argument('--report', action='store_true', help='Generate comparison report')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    from models.hybrid_model import CNNBaseline, HybridCNNViT
    from models.vit_model import VisionTransformer
    
    models_config = [
        ('ViT (Teacher)', VisionTransformer, 'checkpoints/vit_best.pth'),
        ('Hybrid CNN-ViT', HybridCNNViT, 'checkpoints/hybrid_best.pth'),
        ('CNN (Student)', CNNBaseline, 'checkpoints/cnn_best.pth'),
    ]
    
    models = []
    
    for name, model_class, checkpoint_path in models_config:
        print(f"\nLoading {name}...")
        
        try:
            if model_class == VisionTransformer:
                model = model_class(num_classes=4)
            elif model_class == HybridCNNViT:
                model = model_class(num_classes=4)
            else:
                model = model_class(num_classes=4, pretrained=False)
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✅ Loaded from {checkpoint_path}")
            except FileNotFoundError:
                print(f"  ⚠️ Checkpoint not found, using random weights")
            
            models.append((name, model))
        except Exception as e:
            print(f"  ❌ Error loading {name}: {e}")
    
    # Create dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    data_path = Path(args.data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"\nData directory not found or empty: {data_path}")
        print("Please add real training data first.")
        print("Run: python setup_datasets.py --check")
        print("See REAL_DATA_GUIDE.md for instructions.")
        return
    
    dataset = datasets.ImageFolder(str(data_path), transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Evaluate models
    print("\n" + "=" * 60)
    print("KNOWLEDGE DISTILLATION MODEL COMPARISON")
    print("=" * 60)
    
    model_stats = []
    
    for name, model in models:
        print(f"\nEvaluating {name}...")
        stats = evaluate_model(model, dataloader, name, str(device))
        model_stats.append(stats)
        
        print(f"  Accuracy:       {stats.accuracy*100:.2f}%")
        print(f"  Parameters:     {stats.parameters:,}")
        print(f"  Size:           {stats.model_size_mb:.2f} MB")
        print(f"  Inference Time: {stats.inference_time_ms:.2f} ms")
        print(f"  Efficiency:     {stats.efficiency_score:.2f}")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print()
    print(create_comparison_table(model_stats))
    
    # Generate report
    if args.report or args.demo:
        print("\n" + "=" * 60)
        print("GENERATING REPORT")
        print("=" * 60)
        generate_comparison_report(model_stats)
    
    # Create visualizations
    if args.visualize or args.demo:
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        create_visual_comparison(model_stats)
    
    # Save results
    results_path = Path('results/distillation_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'models': [
            {
                'name': s.name,
                'accuracy': s.accuracy,
                'parameters': s.parameters,
                'model_size_mb': s.model_size_mb,
                'inference_time_ms': s.inference_time_ms,
                'flops_estimate': s.flops_estimate,
                'efficiency_score': s.efficiency_score
            }
            for s in model_stats
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(model_stats) >= 2:
        teacher = model_stats[0]
        student = model_stats[-1]
        
        print(f"\nTeacher ({teacher.name}) vs Student ({student.name}):")
        print(f"  Size Reduction:      {(1 - student.model_size_mb/teacher.model_size_mb)*100:.1f}%")
        print(f"  Speedup:             {teacher.inference_time_ms/student.inference_time_ms:.2f}x")
        print(f"  Accuracy Retention:  {student.accuracy/teacher.accuracy*100:.1f}%")
    
    print("\n✅ Knowledge distillation comparison complete!")


if __name__ == '__main__':
    main()
