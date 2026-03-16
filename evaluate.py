"""
Evaluation Script for Banking Fraud Detection.
============================================================
Loads a trained model and runs comprehensive evaluation:
  - Classification metrics
  - Confusion matrix
  - ROC curves
  - Precision-Recall curves
  - Explainability analysis (Attention, Grad-CAM, SHAP)
  - Risk scoring
  - PDF report generation
============================================================
Usage:
  python evaluate.py --checkpoint checkpoints/vit_best.pth
  python evaluate.py --checkpoint checkpoints/vit_best.pth --explain
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from utils.seed import set_seed, get_device
from utils.dataset import create_dataloaders
from models.vit_model import build_vit
from models.hybrid_model import build_hybrid, build_cnn_baseline
from analytics.performance_metrics import (
    MetricsCalculator,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    export_metrics_csv,
    export_results_json,
)
from analytics.risk_scoring import FraudRiskScorer, batch_risk_assessment


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected dict from config, got {type(config)}")
    return config


def evaluate(args):
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()

    class_names = config["data"]["classes"]
    Path("results").mkdir(exist_ok=True)

    # Build model
    if args.model == "vit":
        model = build_vit(config)
    elif args.model == "hybrid":
        model = build_hybrid(config)
    elif args.model == "cnn":
        model = build_cnn_baseline(config)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"[EVAL] Loaded model from {args.checkpoint}")

    # Data
    dataloaders = create_dataloaders(
        data_dir=config["data"]["processed_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        seed=config.get("seed", 42),
        class_names=class_names,
    )

    # Metrics
    calc = MetricsCalculator(class_names, output_dir="results/")
    metrics, inf_time = calc.evaluate_model(model, dataloaders["test"], device)
    model_size = calc.get_model_size(model)
    metrics.update(model_size)

    # Collect predictions for plots
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in dataloaders["test"]:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Plots
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, save_path=f"results/{args.model}_confusion_matrix.png")
    plot_roc_curves(y_true, y_probs, class_names, save_path=f"results/{args.model}_roc_curves.png")
    plot_precision_recall_curves(y_true, y_probs, class_names, save_path=f"results/{args.model}_pr_curves.png")

    # Export
    export_metrics_csv(metrics, f"results/{args.model}_metrics.csv")
    export_results_json(metrics, f"results/{args.model}_evaluation.json")

    # Risk Assessment
    print("\n[EVAL] Running risk assessment on test set...")
    scorer = FraudRiskScorer(
        **config.get("analytics", {}).get("risk_scoring", {})
    )
    risk_results = batch_risk_assessment(model, dataloaders["test"], class_names, scorer, device)
    flagged = sum(1 for r in risk_results if r.get("risk_level") in ["HIGH", "CRITICAL"])
    print(f"  Flagged documents: {flagged}/{len(risk_results)}")

    # Print results
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS — {args.model.upper()}")
    print(f"{'='*55}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Precision:     {metrics['precision_macro']:.4f}")
    print(f"  Recall:        {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    if metrics.get("roc_auc_macro"):
        print(f"  ROC-AUC:       {metrics['roc_auc_macro']:.4f}")
    print(f"  Inference:     {metrics['inference_time_per_sample_ms']:.2f} ms")
    print(f"  Model Size:    {model_size['model_size_mb']:.2f} MB")
    print(f"  FPR:           {metrics['false_positive_rates']}")
    print(f"{'='*55}")

    # --- Explainability (optional) ---
    if args.explain and args.model in ["vit", "hybrid"]:
        print("\n[EVAL] Running explainability analysis...")
        from explainability.attention_visualization import visualize_attention
        from explainability.gradcam import visualize_gradcam, compare_attention_vs_gradcam
        from explainability.report_generator import ExplainabilityReport

        # Get a sample image
        test_iter = iter(dataloaders["test"])
        sample_images, sample_labels = next(test_iter)
        sample_img = sample_images[0:1].to(device)

        # Denormalize for display
        mean = np.array(config["data"]["augmentation"]["normalize"]["mean"])
        std = np.array(config["data"]["augmentation"]["normalize"]["std"])
        display_img = sample_images[0].permute(1, 2, 0).numpy()
        display_img = (display_img * std + mean) * 255
        display_img = display_img.clip(0, 255).astype(np.uint8)

        # Attention map
        visualize_attention(
            model, sample_img, display_img,
            image_size=config["data"]["image_size"],
            patch_size=config["vit"]["patch_size"],
            device=device,
            save_path=f"results/{args.model}_attention_map.png",
        )

        # Grad-CAM
        target_layer = config.get("explainability", {}).get(
            "gradcam", {}
        ).get("target_layer", "encoder.layers.11")

        overlay, pred_cls, conf = visualize_gradcam(
            model, sample_img, display_img, class_names,
            target_layer=target_layer,
            image_size=config["data"]["image_size"],
            patch_size=config["vit"]["patch_size"],
            save_path=f"results/{args.model}_gradcam.png",
        )

        # Attention vs Grad-CAM comparison
        compare_attention_vs_gradcam(
            model, sample_img, display_img, class_names,
            target_layer=target_layer,
            device=device,
            save_path=f"results/{args.model}_attn_vs_gradcam.png",
        )

        # Generate PDF report
        reporter = ExplainabilityReport(output_dir="reports/")
        probs_dict = {}
        with torch.no_grad():
            out = model(sample_img)
            p = torch.nn.functional.softmax(out, dim=1)
            for i, c in enumerate(class_names):
                probs_dict[c] = p[0, i].item()

        reporter.generate(
            document_image=display_img,
            prediction=class_names[pred_cls],
            confidence=conf,
            class_probabilities=probs_dict,
            attention_overlay=None,
            gradcam_overlay=overlay,
            document_id=f"EVAL_{args.model.upper()}_SAMPLE",
        )
        print("[EVAL] Explainability analysis complete. Reports saved to reports/")

    print("\n[DONE] Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "hybrid", "cnn"])
    parser.add_argument("--explain", action="store_true", help="Run XAI analysis")
    evaluate(parser.parse_args())
