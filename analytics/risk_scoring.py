"""
Fraud Risk Scoring Module for Banking Analytics.

Computes a composite risk score for each document using:
  Risk Score = Attention Intensity × Confidence × Anomaly Score

Provides risk-level classifications suitable for banking decision pipelines.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Risk Score Calculator
# ------------------------------------------------------------------ #
class FraudRiskScorer:
    """
    Multi-factor fraud risk scoring for financial documents.
    
    Combines model confidence, attention intensity, and anomaly
    detection for a comprehensive risk assessment.
    """

    # Risk level thresholds
    RISK_LEVELS = {
        "LOW": (0.0, 0.3),
        "MEDIUM": (0.3, 0.6),
        "HIGH": (0.6, 0.8),
        "CRITICAL": (0.8, 1.0),
    }

    def __init__(
        self,
        attention_weight: float = 0.4,
        confidence_weight: float = 0.35,
        anomaly_weight: float = 0.25,
        fraud_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            attention_weight: Weight for attention intensity component.
            confidence_weight: Weight for confidence component.
            anomaly_weight: Weight for anomaly score component.
            fraud_classes: List of class names considered as fraud.
        """
        total = attention_weight + confidence_weight + anomaly_weight
        self.attention_weight = attention_weight / total
        self.confidence_weight = confidence_weight / total
        self.anomaly_weight = anomaly_weight / total
        self.fraud_classes = fraud_classes or ["fraud", "tampered", "forged"]

    def compute_attention_intensity(
        self,
        attention_map: Optional[np.ndarray],
    ) -> float:
        """
        Compute attention intensity score.
        High concentration of attention in small area = suspicious.

        Args:
            attention_map: (H, W) normalized attention heatmap.

        Returns:
            Intensity score in [0, 1].
        """
        if attention_map is None:
            return 0.5

        # Entropy-based: low entropy = high concentration = more suspicious
        flat = attention_map.flatten()
        flat = flat / (flat.sum() + 1e-8)
        entropy = -np.sum(flat * np.log(flat + 1e-8))
        max_entropy = np.log(len(flat))
        normalized_entropy = entropy / (max_entropy + 1e-8)

        # Invert: low entropy → high intensity
        intensity = 1.0 - normalized_entropy
        return float(np.clip(intensity, 0, 1))

    def compute_confidence_score(
        self,
        class_probabilities: Dict[str, float],
        predicted_class: str,
    ) -> float:
        """
        Compute confidence-based risk score.
        High confidence in fraud class = high risk.

        Args:
            class_probabilities: Per-class probability dict.
            predicted_class: Model's prediction.

        Returns:
            Risk score in [0, 1].
        """
        fraud_prob = sum(
            class_probabilities.get(cls, 0.0)
            for cls in self.fraud_classes
        )
        # Scale: mostly fraud probability
        if predicted_class in self.fraud_classes:
            return float(np.clip(fraud_prob, 0, 1))
        else:
            return float(np.clip(fraud_prob * 0.5, 0, 1))

    def compute_anomaly_score(
        self,
        feature_vector: Optional[np.ndarray] = None,
        reference_features: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute anomaly score based on distance from genuine reference.

        Args:
            feature_vector: (D,) model features for the document.
            reference_features: (N, D) reference features from genuine docs.

        Returns:
            Anomaly score in [0, 1].
        """
        if feature_vector is None or reference_features is None:
            return 0.5

        # Cosine distance from centroid of genuine features
        centroid = reference_features.mean(axis=0)
        cos_sim = np.dot(feature_vector, centroid) / (
            np.linalg.norm(feature_vector) * np.linalg.norm(centroid) + 1e-8
        )
        # Low similarity = high anomaly
        anomaly = 1.0 - max(0, cos_sim)
        return float(np.clip(anomaly, 0, 1))

    def compute_risk_score(
        self,
        attention_intensity: float,
        confidence_score: float,
        anomaly_score: float,
    ) -> Tuple[float, str]:
        """
        Compute composite fraud risk score.

        Risk Score = Attention Intensity × Confidence × Anomaly Score

        Each factor is in [0, 1], so the product naturally stays in [0, 1].
        A small epsilon is added to each factor to avoid zero-collapse.

        Args:
            attention_intensity: Attention-based score [0, 1].
            confidence_score: Confidence-based score [0, 1].
            anomaly_score: Anomaly-based score [0, 1].

        Returns:
            (risk_score, risk_level) tuple.
        """
        eps = 0.01  # Prevent zero-collapse
        risk_score = (
            (attention_intensity + eps)
            * (confidence_score + eps)
            * (anomaly_score + eps)
        )
        # Normalize back to [0, 1] range (max product ≈ 1.03^3 ≈ 1.09)
        risk_score = float(np.clip(risk_score, 0, 1))

        # Determine risk level
        risk_level = "LOW"
        for level, (low, high) in self.RISK_LEVELS.items():
            if low <= risk_score < high:
                risk_level = level
                break
        if risk_score >= 0.8:
            risk_level = "CRITICAL"

        return risk_score, risk_level

    def assess_document(
        self,
        model,
        image_tensor: torch.Tensor,
        class_names: List[str],
        attention_map: Optional[np.ndarray] = None,
        feature_vector: Optional[np.ndarray] = None,
        reference_features: Optional[np.ndarray] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict:
        """
        Full risk assessment pipeline for a single document.

        Args:
            model: Trained ViT model.
            image_tensor: (1, C, H, W) preprocessed image.
            class_names: Class names.
            attention_map: Optional precomputed attention map.
            feature_vector: Optional model features.
            reference_features: Optional reference genuine features.
            device: Device.

        Returns:
            Comprehensive risk assessment dictionary.
        """
        model = model.to(device).eval()
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class_idx = logits.argmax(dim=1).item()
            pred_class = class_names[pred_class_idx]

        class_probs = {
            class_names[i]: probs[0, i].item()
            for i in range(len(class_names))
        }

        # Compute component scores
        attn_score = self.compute_attention_intensity(attention_map)
        conf_score = self.compute_confidence_score(class_probs, pred_class)
        anom_score = self.compute_anomaly_score(feature_vector, reference_features)

        # Composite risk
        risk_score, risk_level = self.compute_risk_score(
            attn_score, conf_score, anom_score
        )

        return {
            "predicted_class": pred_class,
            "predicted_class_idx": pred_class_idx,
            "class_probabilities": class_probs,
            "confidence": probs[0, pred_class_idx].item(),
            "attention_intensity": attn_score,
            "confidence_risk": conf_score,
            "anomaly_score": anom_score,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_components": {
                "attention": f"{attn_score:.4f} (weight: {self.attention_weight:.2f})",
                "confidence": f"{conf_score:.4f} (weight: {self.confidence_weight:.2f})",
                "anomaly": f"{anom_score:.4f} (weight: {self.anomaly_weight:.2f})",
            },
            "is_flagged": risk_level in ["HIGH", "CRITICAL"],
        }


def batch_risk_assessment(
    model,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    scorer: Optional[FraudRiskScorer] = None,
    device: torch.device = torch.device("cpu"),
) -> List[Dict]:
    """
    Run risk assessment on an entire dataset batch.

    Args:
        model: Trained model.
        dataloader: DataLoader.
        class_names: Class names.
        scorer: FraudRiskScorer instance.
        device: Device.

    Returns:
        List of risk assessment dicts.
    """
    if scorer is None:
        scorer = FraudRiskScorer()

    results = []
    model = model.to(device).eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            for i in range(images.shape[0]):
                pred_idx = logits[i].argmax().item()
                pred_class = class_names[pred_idx]
                class_probs = {
                    class_names[j]: probs[i, j].item()
                    for j in range(len(class_names))
                }
                conf_score = scorer.compute_confidence_score(class_probs, pred_class)
                risk_score, risk_level = scorer.compute_risk_score(
                    0.5, conf_score, 0.5  # Default attention/anomaly when not available
                )
                results.append({
                    "index": batch_idx * (dataloader.batch_size or 1) + i,
                    "true_label": class_names[labels[i].item()],
                    "predicted_class": pred_class,
                    "confidence": probs[i, pred_idx].item(),
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                })

    return results
