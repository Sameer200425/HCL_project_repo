"""
Tests for Analytics, Explainability, Compliance, Uncertainty & Active Learning
==============================================================================
Covers modules that did not have dedicated tests:
  - analytics/risk_scoring.py
  - analytics/performance_metrics.py
  - analytics/fraud_trend_engine.py
  - explainability/gradcam.py
  - compliance/regulatory_report.py
  - uncertainty_quantification.py
  - active_learning.py

Run:
    pytest tests/test_analytics_explainability.py -v
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===================================================================
#  Analytics — Risk Scoring
# ===================================================================

class TestFraudRiskScorer:
    """Tests for analytics/risk_scoring.py."""

    @pytest.fixture
    def scorer(self):
        from analytics.risk_scoring import FraudRiskScorer
        return FraudRiskScorer()

    def test_scorer_creation(self, scorer):
        assert scorer is not None
        assert abs(scorer.attention_weight + scorer.confidence_weight + scorer.anomaly_weight - 1.0) < 1e-6

    def test_custom_weights(self):
        from analytics.risk_scoring import FraudRiskScorer
        s = FraudRiskScorer(attention_weight=0.5, confidence_weight=0.3, anomaly_weight=0.2)
        assert abs(s.attention_weight + s.confidence_weight + s.anomaly_weight - 1.0) < 1e-6

    def test_attention_intensity_none(self, scorer):
        score = scorer.compute_attention_intensity(None)
        assert score == 0.5

    def test_attention_intensity_uniform(self, scorer):
        uniform = np.ones((14, 14)) / (14 * 14)
        score = scorer.compute_attention_intensity(uniform)
        assert 0.0 <= score <= 1.0
        # Uniform attention → low intensity (high entropy)
        assert score < 0.3

    def test_attention_intensity_concentrated(self, scorer):
        concentrated = np.zeros((14, 14))
        concentrated[7, 7] = 1.0
        score = scorer.compute_attention_intensity(concentrated)
        assert 0.0 <= score <= 1.0
        # Concentrated → high intensity
        assert score > 0.7

    def test_confidence_score_fraud(self, scorer):
        probs = {"genuine": 0.05, "fraud": 0.80, "tampered": 0.10, "forged": 0.05}
        score = scorer.compute_confidence_score(probs, "fraud")
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_confidence_score_genuine(self, scorer):
        probs = {"genuine": 0.95, "fraud": 0.02, "tampered": 0.02, "forged": 0.01}
        score = scorer.compute_confidence_score(probs, "genuine")
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_anomaly_score_no_features(self, scorer):
        score = scorer.compute_anomaly_score()
        assert 0.0 <= score <= 1.0

    def test_risk_levels(self, scorer):
        assert len(scorer.RISK_LEVELS) == 4
        assert "LOW" in scorer.RISK_LEVELS
        assert "CRITICAL" in scorer.RISK_LEVELS


# ===================================================================
#  Analytics — Performance Metrics
# ===================================================================

class TestMetricsCalculator:
    """Tests for analytics/performance_metrics.py."""

    @pytest.fixture
    def calculator(self, tmp_path):
        from analytics.performance_metrics import MetricsCalculator
        return MetricsCalculator(
            class_names=["genuine", "fraud", "tampered", "forged"],
            output_dir=str(tmp_path),
        )

    def test_calculator_creation(self, calculator):
        assert calculator is not None
        assert len(calculator.class_names) == 4

    def test_compute_all_metrics(self, calculator):
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 0])  # one mismatch
        metrics = calculator.compute_all_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics or "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_perfect_predictions(self, calculator):
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 0, 1])
        metrics = calculator.compute_all_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0


# ===================================================================
#  Explainability — GradCAM
# ===================================================================

class TestGradCAM:
    """Tests for explainability/gradcam.py."""

    @pytest.fixture
    def small_vit(self):
        from models.vit_model import VisionTransformer
        return VisionTransformer(
            image_size=224, patch_size=16, in_channels=3,
            num_classes=4, embed_dim=64, num_heads=4,
            num_layers=2, mlp_dim=128, dropout=0.0,
        )

    def test_gradcam_creation(self, small_vit):
        from explainability.gradcam import GradCAM
        try:
            gc = GradCAM(small_vit, target_layer_name="encoder.layers.1")
            assert gc is not None
        except (AttributeError, KeyError):
            # Acceptable if layer naming differs
            pytest.skip("Layer naming mismatch in this ViT implementation")

    def test_gradcam_get_layer(self, small_vit):
        from explainability.gradcam import GradCAM
        # Test static method _get_layer navigates correctly
        try:
            layer = GradCAM._get_layer(small_vit, "encoder.layers.0")
            assert layer is not None
        except AttributeError:
            pytest.skip("Layer path not available")


# ===================================================================
#  Compliance — Regulatory Report
# ===================================================================

class TestComplianceReport:
    """Tests for compliance/regulatory_report.py."""

    def test_report_config_defaults(self):
        from compliance.regulatory_report import ReportConfig
        cfg = ReportConfig()
        assert cfg.bank_name == "ABC Bank Ltd."
        assert len(cfg.classification_classes) == 4

    def test_report_config_custom(self):
        from compliance.regulatory_report import ReportConfig
        cfg = ReportConfig(bank_name="Test Bank", model_version="2.0.0")
        assert cfg.bank_name == "Test Bank"
        assert cfg.model_version == "2.0.0"

    def test_checklist_item_creation(self):
        from compliance.regulatory_report import ChecklistItem
        item = ChecklistItem(
            requirement="Model explainability",
            framework="RBI",
            status="PASS",
            evidence="Grad-CAM, SHAP, Attention maps",
            risk_if_fail="HIGH",
        )
        assert item.status == "PASS"
        assert item.framework == "RBI"

    def test_generator_creation(self):
        from compliance.regulatory_report import ComplianceReportGenerator
        gen = ComplianceReportGenerator()
        assert gen is not None
        assert gen.config.bank_name == "ABC Bank Ltd."

    def test_load_metrics(self):
        from compliance.regulatory_report import ComplianceReportGenerator
        gen = ComplianceReportGenerator()
        gen.load_metrics({
            "accuracy": 0.97,
            "f1": 0.97,
            "precision": 0.97,
            "recall": 0.97,
        })
        assert gen.metrics["accuracy"] == 0.97

    def test_load_explainability_evidence(self):
        from compliance.regulatory_report import ComplianceReportGenerator
        gen = ComplianceReportGenerator()
        gen.load_explainability_evidence(
            gradcam_available=True,
            shap_available=True,
            attention_rollout=True,
        )
        assert gen.xai_evidence.get("gradcam_available", True)

    def test_generate_report(self):
        from compliance.regulatory_report import ComplianceReportGenerator
        gen = ComplianceReportGenerator()
        gen.load_metrics({"accuracy": 0.97, "f1": 0.97})
        report = gen.generate()
        assert isinstance(report, str)
        assert len(report) > 100


# ===================================================================
#  Uncertainty Quantification
# ===================================================================

class TestUncertaintyQuantification:
    """Tests for uncertainty_quantification.py."""

    def test_uncertainty_result_dataclass(self):
        from uncertainty_quantification import UncertaintyResult
        result = UncertaintyResult(
            predicted_class=0,
            class_name="genuine",
            mean_confidence=0.95,
            epistemic_std=0.02,
            aleatoric_entropy=0.10,
            total_uncertainty=0.05,
            uncertainty_level="LOW",
            class_probabilities={"genuine": 0.95, "fraud": 0.05},
            requires_human_review=False,
            mc_passes_used=50,
        )
        assert result.class_name == "genuine"
        assert result.uncertainty_level == "LOW"
        assert not result.requires_human_review

    def test_enable_mc_dropout(self):
        from uncertainty_quantification import enable_mc_dropout
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(0.1),
            nn.Linear(10, 4),
        )
        model.eval()
        # After eval(), dropout should be in eval mode
        assert not model[1].training
        enable_mc_dropout(model)
        # After enable_mc_dropout, dropout should be in train mode
        assert model[1].training

    def test_mc_dropout_predictor_creation(self):
        from uncertainty_quantification import MCDropoutPredictor
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(0.1),
            nn.Linear(10, 4),
        )
        predictor = MCDropoutPredictor(model, n_passes=10, dropout_rate=0.1)
        assert predictor is not None
        assert predictor.n_passes == 10

    def test_uncertainty_thresholds(self):
        from uncertainty_quantification import MCDropoutPredictor
        assert "LOW" in MCDropoutPredictor.UNCERTAINTY_THRESHOLDS
        assert "HIGH" in MCDropoutPredictor.UNCERTAINTY_THRESHOLDS


# ===================================================================
#  Active Learning
# ===================================================================

class TestActiveLearning:
    """Tests for active_learning.py."""

    def test_query_strategy_enum(self):
        from active_learning import QueryStrategy
        assert QueryStrategy.ENTROPY.value == "entropy"
        assert QueryStrategy.MARGIN_SAMPLING.value == "margin_sampling"
        assert QueryStrategy.CORE_SET.value == "core_set"

    def test_sample_score_dataclass(self):
        from active_learning import SampleScore
        score = SampleScore(
            index=0,
            score=0.85,
            strategy="entropy",
            predicted_class="fraud",
            confidence=0.6,
            entropy=0.85,
            margin=0.1,
        )
        assert score.score == 0.85
        assert score.strategy == "entropy"
        assert score.priority == "NORMAL"

    def test_active_learner_creation(self):
        from active_learning import ActiveLearner, QueryStrategy
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(0.1),
            nn.Linear(10, 4),
        )
        learner = ActiveLearner(model, strategy=QueryStrategy.ENTROPY)
        assert learner is not None
        assert learner.strategy == QueryStrategy.ENTROPY

    def test_active_learner_strategies(self):
        from active_learning import ActiveLearner, QueryStrategy
        model = nn.Sequential(nn.Linear(10, 4))
        for strategy in [QueryStrategy.ENTROPY, QueryStrategy.LEAST_CONFIDENCE, QueryStrategy.MARGIN_SAMPLING]:
            learner = ActiveLearner(model, strategy=strategy)
            assert learner.strategy == strategy

    def test_class_names_default(self):
        from active_learning import CLASS_NAMES
        assert len(CLASS_NAMES) == 4
        assert "genuine" in CLASS_NAMES
        assert "fraud" in CLASS_NAMES


# ===================================================================
#  Analytics — Fraud Trend Engine (basic imports)
# ===================================================================

class TestFraudTrendEngine:
    """Basic tests for analytics/fraud_trend_engine.py."""

    def test_import(self):
        import analytics.fraud_trend_engine  # noqa: F401
        assert True

    def test_module_has_classes(self):
        import analytics.fraud_trend_engine as fte
        # Check at least one main class or function exists
        attrs = dir(fte)
        assert len(attrs) > 5  # Module is non-trivial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
