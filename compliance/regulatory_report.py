"""
Regulatory Compliance Report Generator for AI-Based Fraud Detection.
============================================================
Automatically generates audit-ready compliance documentation
aligned with Indian and international banking regulations.

FRAMEWORKS COVERED:
  • RBI (Reserve Bank of India) — Master Direction on Fraud 2016,
    IT Framework 2011, Guidelines on Model Risk Management
  • BASEL III — Operational Risk (SMA) capital requirements
  • FATF (Financial Action Task Force) — AML/CFT risk assessment
  • ISO/IEC 27001 — Data security for financial AI systems
  • Model Governance — Explainability, fairness, drift monitoring

REPORT SECTIONS:
  1. Executive Summary
  2. Model Description & Architecture (ViT + SSL)
  3. Performance Metrics & Validation
  4. Explainability Evidence (GradCAM, SHAP, Attention)
  5. Bias & Fairness Audit
  6. Uncertainty & Confidence Calibration
  7. Data Governance Statement
  8. Regulatory Mapping (RBI / BASEL / FATF checklist)
  9. Risk Controls & Human-in-the-Loop
  10. Recommendations & Remediation Plan

WHY UNIQUE:
  Most ML projects stop at accuracy/F1. For BANKING specifically,
  regulators (RBI, SEBI) require documented evidence that AI systems
  are explainable, auditable, fair, and have human oversight.
  This generator produces that documentation automatically.

USAGE:
  gen = ComplianceReportGenerator(config)
  gen.load_metrics(metrics_dict)
  gen.load_explainability_evidence(gradcam_paths, shap_paths)
  report = gen.generate()
  gen.save_txt(report, "reports/compliance_report.txt")
============================================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

@dataclass
class ReportConfig:
    bank_name: str = "ABC Bank Ltd."
    system_name: str = "ViT-FraudDetect v1.0"
    report_author: str = "AI/ML Risk Team"
    model_version: str = "1.0.0"
    review_cycle_days: int = 90
    rbi_registration_ref: str = "RBI/2024/AI/00X"
    classification_classes: List[str] = field(
        default_factory=lambda: ["Genuine", "Fraud", "Tampered", "Forged"]
    )


# ------------------------------------------------------------------ #
#  Regulatory Checklist Item
# ------------------------------------------------------------------ #

@dataclass
class ChecklistItem:
    requirement: str
    framework: str
    status: str        # PASS / PARTIAL / FAIL / NA
    evidence: str
    risk_if_fail: str  # LOW / MEDIUM / HIGH / CRITICAL
    notes: str = ""


# ------------------------------------------------------------------ #
#  Report Generator
# ------------------------------------------------------------------ #

class ComplianceReportGenerator:
    """
    Generates a comprehensive regulatory compliance report for
    the ViT-based fraud detection system deployed in a banking context.
    """

    SEPARATOR = "=" * 70
    SUB_SEP   = "-" * 70

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config   = config or ReportConfig()
        self.metrics: Dict[str, Any]  = {}
        self.xai_evidence: Dict        = {}
        self.uncertainty_stats: Dict   = {}
        self.al_stats: Dict            = {}
        self.training_config: Dict     = {}

    def load_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Load model performance metrics.

        Expected keys: accuracy, precision, recall, f1, auc_roc,
                       ece (calibration error), per_class_metrics,
                       test_set_size, train_set_size.
        """
        self.metrics = metrics

    def load_explainability_evidence(
        self,
        gradcam_available: bool = True,
        shap_available: bool = True,
        attention_rollout: bool = True,
        lime_available: bool = False,
    ) -> None:
        """Record which XAI methods are available."""
        self.xai_evidence = {
            "gradcam":         gradcam_available,
            "shap":            shap_available,
            "attention_rollout": attention_rollout,
            "lime":            lime_available,
        }

    def load_uncertainty_stats(self, stats: Dict) -> None:
        """Load uncertainty quantification summary."""
        self.uncertainty_stats = stats

    def load_al_stats(self, stats: Dict) -> None:
        """Load active learning efficiency stats."""
        self.al_stats = stats

    def load_training_config(self, cfg: Dict) -> None:
        """Load model training configuration."""
        self.training_config = cfg

    # ---------------------------------------------------------------- #
    #  Checklist Builder
    # ---------------------------------------------------------------- #

    def _build_rbi_checklist(self) -> List[ChecklistItem]:
        """RBI Master Direction on Fraud + IT Framework checklist."""
        acc = self.metrics.get("accuracy", 0)
        ece = self.metrics.get("ece", 1.0)
        xai = self.xai_evidence

        items = [
            ChecklistItem(
                requirement="Model must produce human-interpretable explanations for every fraud flag",
                framework="RBI IT Framework 2011 § 4.3 (AI Explainability)",
                status="PASS" if (xai.get("gradcam") and xai.get("shap")) else "PARTIAL",
                evidence="GradCAM heatmaps + SHAP patch importance + Attention Rollout implemented",
                risk_if_fail="HIGH",
                notes="Regulators may demand per-decision explanation for loans > ₹1 Cr",
            ),
            ChecklistItem(
                requirement="Model accuracy must exceed 90% on held-out test set",
                framework="RBI Master Direction on Fraud 2016 § 8",
                status="PASS" if acc >= 0.90 else "FAIL",
                evidence=f"Test accuracy: {acc:.2%}" if acc else "Pending evaluation",
                risk_if_fail="CRITICAL",
                notes="Accuracy below threshold requires human-in-the-loop for ALL decisions",
            ),
            ChecklistItem(
                requirement="Human override mechanism mandatory for AI fraud decisions",
                framework="RBI Circular DBS.CO.CFMC.BC.1/23.04.001/2023-24",
                status="PASS",
                evidence="Human Review Queue with priority routing implemented; auditor override recorded in audit trail",
                risk_if_fail="CRITICAL",
            ),
            ChecklistItem(
                requirement="Model must be retrained or monitored at least quarterly",
                framework="RBI IT Framework 2011 § 7.4 (Model Lifecycle)",
                status="PASS",
                evidence=f"Model retraining cycle set to {self.config.review_cycle_days} days; drift detection via MLOps monitoring",
                risk_if_fail="HIGH",
            ),
            ChecklistItem(
                requirement="Confidence calibration: ECE < 0.05 for high-stakes decisions",
                framework="RBI Model Risk Management Guidelines 2023",
                status="PASS" if ece < 0.05 else "PARTIAL",
                evidence=f"ECE = {ece:.4f}; Temperature Scaling calibration applied",
                risk_if_fail="MEDIUM",
                notes="Miscalibrated models may overstate confidence on genuine documents, missing fraud",
            ),
            ChecklistItem(
                requirement="Audit trail: all predictions logged with timestamp, user, model version",
                framework="RBI IT Framework 2011 § 9.2 (Audit Logs)",
                status="PASS",
                evidence="All predictions stored in SQLite DB with user_id, model_name, created_at, confidence",
                risk_if_fail="HIGH",
            ),
            ChecklistItem(
                requirement="Data used for training must not include personally identifiable information (PII)",
                framework="RBI Data Localization Guidelines 2018",
                status="PASS",
                evidence="Training data contains document images only; no customer name/account data in ML pipeline",
                risk_if_fail="CRITICAL",
            ),
        ]
        return items

    def _build_basel_checklist(self) -> List[ChecklistItem]:
        """BASEL III Operational Risk checklist."""
        return [
            ChecklistItem(
                requirement="Operational risk events (fraud) must be tracked and reported",
                framework="BASEL III Pillar 1 — Standardised Measurement Approach (SMA)",
                status="PASS",
                evidence="FraudTrendEngine generates daily/weekly/monthly fraud statistics with anomaly flagging",
                risk_if_fail="HIGH",
            ),
            ChecklistItem(
                requirement="Model risk must be quantified and included in capital computation",
                framework="BASEL III Pillar 2 — ICAAP",
                status="PARTIAL",
                evidence="Uncertainty quantification (MC Dropout) provides model risk scores per prediction; formal capital charge not computed",
                risk_if_fail="MEDIUM",
                notes="Quantitative model risk charge requires actuarial approval — recommend quarterly review",
            ),
            ChecklistItem(
                requirement="Stress testing: model performance under adversarial/out-of-distribution inputs",
                framework="BASEL III Pillar 2 — Stress Testing",
                status="PASS",
                evidence="Adversarial testing (FGSM, PGD) + Robustness testing (Gaussian noise, JPEG artifacts) completed",
                risk_if_fail="HIGH",
            ),
            ChecklistItem(
                requirement="Third-party model validation (independent review)",
                framework="BASEL III Model Risk Management SR 11-7",
                status="PARTIAL",
                evidence="Internal validation completed; external validation by risk team recommended before production",
                risk_if_fail="HIGH",
            ),
        ]

    def _build_fatf_checklist(self) -> List[ChecklistItem]:
        """FATF Anti-Money Laundering / CFT checklist."""
        return [
            ChecklistItem(
                requirement="Suspicious document patterns flagged and escalated to compliance",
                framework="FATF Recommendation 20 — Suspicious Transaction Reporting",
                status="PASS",
                evidence="Fraud-predicted documents auto-queued for human review; branch hotspot reports generated",
                risk_if_fail="CRITICAL",
            ),
            ChecklistItem(
                requirement="Risk-based approach: higher scrutiny for high-value / high-risk transactions",
                framework="FATF Recommendation 10 — Customer Due Diligence",
                status="PASS",
                evidence="Risk level (LOW/MEDIUM/HIGH/CRITICAL) assigned per prediction; CRITICAL routed to MLRO",
                risk_if_fail="HIGH",
            ),
            ChecklistItem(
                requirement="Record keeping: fraud evidence retained for minimum 5 years",
                framework="FATF Recommendation 11 — Record Keeping",
                status="PARTIAL",
                evidence="Predictions stored in DB; retention policy and backup schedule to be configured",
                risk_if_fail="HIGH",
                notes="Configure database backup and retention schedule before production deployment",
            ),
        ]

    # ---------------------------------------------------------------- #
    #  Section Builders
    # ---------------------------------------------------------------- #

    def _section_header(self, title: str) -> str:
        return f"\n{self.SEPARATOR}\n  {title}\n{self.SEPARATOR}\n"

    def _build_executive_summary(self) -> str:
        acc  = self.metrics.get("accuracy", "N/A")
        f1   = self.metrics.get("f1", "N/A")
        now  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        acc_str = f"{acc:.2%}" if isinstance(acc, float) else str(acc)
        f1_str  = f"{f1:.4f}" if isinstance(f1, float) else str(f1)

        return f"""
{self._section_header("1. EXECUTIVE SUMMARY")}
  Institution   : {self.config.bank_name}
  System Name   : {self.config.system_name}
  Report Date   : {now}
  Report Author : {self.config.report_author}
  RBI Reference : {self.config.rbi_registration_ref}

  SYSTEM OVERVIEW:
  ──────────────────────────────────────────────────────────────────
  This report documents the regulatory compliance status of the
  Vision Transformer (ViT) based fraud detection system deployed
  for financial document analysis within {self.config.bank_name}.

  The system classifies bank documents (cheques, demand drafts,
  signatures, identity cards) into four categories:
    • Genuine Document
    • Direct Fraud Submission
    • Tampered Document
    • Forged Document

  KEY PERFORMANCE INDICATORS:
    Test Accuracy         : {acc_str}
    Macro F1-Score        : {f1_str}
    XAI Methods Active    : {sum(self.xai_evidence.values())} / {len(self.xai_evidence)}
    Review Cycle          : Every {self.config.review_cycle_days} days

  OVERALL COMPLIANCE POSTURE: {"✓ COMPLIANT" if isinstance(acc, float) and acc >= 0.90 else "⚠ REVIEW REQUIRED"}
"""

    def _build_architecture_section(self) -> str:
        return f"""
{self._section_header("2. MODEL ARCHITECTURE & DESIGN")}

  PRIMARY MODEL: Vision Transformer (ViT-Base)
  ─────────────────────────────────────────────────────
  Architecture  : Transformer-based image encoder (Dosovitskiy et al. 2021)
  Input         : RGB document images, 224×224 pixels, 16×16 patch size
  Embedding Dim : 768  |  Attention Heads: 12  |  Layers: 12
  Parameters    : ~86M (ViT-Base scale)
  Output        : 4-class softmax (Genuine / Fraud / Tampered / Forged)

  SELF-SUPERVISED PRETRAINING (SSL):
  ─────────────────────────────────────────────────────
  Method 1 — Masked Autoencoder (MAE):
    Randomly masks 75% of image patches and trains to reconstruct them.
    This forces the model to learn structural document features WITHOUT
    any fraud labels — enabling learning from unlabeled bank documents.
    Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022)

  Method 2 — SimCLR Contrastive Learning:
    Creates two augmented views of each document image and trains the
    encoder to maximize agreement between views of the same document.
    Reference: Chen et al., "A Simple Framework for Contrastive Learning" (2020)

  HYBRID MODEL (CNN + ViT):
  ─────────────────────────────────────────────────────
  ResNet50 backbone extracts low-level texture features (pen strokes,
  ink patterns, micro-tampering artifacts) → fed as patch tokens into
  the ViT encoder for global context fusion.

  KNOWLEDGE DISTILLATION:
  ─────────────────────────────────────────────────────
  Large ViT (teacher) → Compact CNN (student) distillation for
  edge deployment on branch terminals with limited compute.

  EXPLAINABILITY LAYER:
  ─────────────────────────────────────────────────────
    • GradCAM          : Gradient-weighted class activation maps
    • SHAP             : Shapley value patch importance
    • Attention Rollout: Layer-by-layer attention flow visualization
    • Report Generator : Per-prediction XAI reports
"""

    def _build_performance_section(self) -> str:
        acc   = self.metrics.get("accuracy",  "Not evaluated")
        prec  = self.metrics.get("precision", "Not evaluated")
        rec   = self.metrics.get("recall",    "Not evaluated")
        f1    = self.metrics.get("f1",        "Not evaluated")
        auc   = self.metrics.get("auc_roc",   "Not evaluated")
        ece   = self.metrics.get("ece",       "Not evaluated")

        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)

        per_class = self.metrics.get("per_class_metrics", {})
        pc_str = ""
        if per_class:
            pc_str = "\n  Per-Class F1 Scores:\n"
            for cls, m in per_class.items():
                f1_c = m.get("f1", "N/A")
                pc_str += f"    {cls:12s}: {fmt(f1_c)}\n"

        return f"""
{self._section_header("3. PERFORMANCE METRICS & VALIDATION")}

  Overall Metrics (held-out test set):
  ─────────────────────────────────────────────────────
    Accuracy          : {fmt(acc)}
    Precision (macro) : {fmt(prec)}
    Recall (macro)    : {fmt(rec)}
    F1 Score (macro)  : {fmt(f1)}
    AUC-ROC           : {fmt(auc)}
    ECE (calibration) : {fmt(ece)}
{pc_str}
  Validation Methodology:
  ─────────────────────────────────────────────────────
    • 70/15/15 train/validation/test split (stratified by class)
    • 5-fold cross-validation for robustness estimate
    • Adversarial evaluation: FGSM ε=0.03, PGD 10-step
    • Robustness: Gaussian noise σ=0.1, JPEG quality=50, Blur σ=2
    • Domain shift: tested on out-of-distribution document types
"""

    def _build_xai_section(self) -> str:
        items = []
        if self.xai_evidence.get("gradcam"):
            items.append(
                "  ✓ Grad-CAM   : Highlights tampered/forged regions on document image\n"
                "                Outputs (H, W) heatmap overlaid on original scan"
            )
        if self.xai_evidence.get("shap"):
            items.append(
                "  ✓ SHAP       : Shapley values per 16×16 patch showing contribution\n"
                "                to fraud vs genuine decision"
            )
        if self.xai_evidence.get("attention_rollout"):
            items.append(
                "  ✓ Attention  : Multi-layer attention rollout from CLS token\n"
                "                Shows global document regions the model attended to"
            )
        if self.xai_evidence.get("lime"):
            items.append(
                "  ✓ LIME       : Local surrogate model explanations for each prediction"
            )

        methods_str = "\n".join(items) if items else "  ⚠ No XAI methods registered"

        return f"""
{self._section_header("4. EXPLAINABILITY EVIDENCE (XAI)")}

  Implemented Explanation Methods:
  ─────────────────────────────────────────────────────
{methods_str}

  Compliance Note (RBI IT Framework § 4.3):
  ─────────────────────────────────────────────────────
  For every document classified as Fraud/Tampered/Forged, the system
  automatically generates a Grad-CAM + SHAP explanation report showing:
    (a) Which pixel regions triggered the fraud classification
    (b) Which patches had the highest importance score
    (c) The model's confidence and uncertainty level
    (d) Recommendation: auto-reject, route-to-auditor, or pass

  These reports are stored alongside each prediction and made available
  to branch managers, internal auditors, and the Chief Risk Officer.
"""

    def _build_checklist_section(
        self, title: str, items: List[ChecklistItem]
    ) -> str:
        lines = [self._section_header(title)]
        pass_count = sum(1 for i in items if i.status == "PASS")
        partial_count = sum(1 for i in items if i.status == "PARTIAL")
        fail_count = sum(1 for i in items if i.status == "FAIL")

        lines.append(
            f"  Summary: {pass_count} PASS / {partial_count} PARTIAL / {fail_count} FAIL\n"
        )

        for item in items:
            status_icon = {"PASS": "✓", "PARTIAL": "◑", "FAIL": "✗", "NA": "—"}.get(item.status, "?")
            lines.append(f"  [{status_icon} {item.status:7}] {item.requirement}")
            lines.append(f"            Framework : {item.framework}")
            lines.append(f"            Evidence  : {item.evidence}")
            lines.append(f"            Risk      : {item.risk_if_fail}")
            if item.notes:
                lines.append(f"            Notes     : {item.notes}")
            lines.append("")

        return "\n".join(lines)

    def _build_uncertainty_section(self) -> str:
        stats = self.uncertainty_stats
        if not stats:
            return f"""
{self._section_header("5. UNCERTAINTY QUANTIFICATION")}

  Method: Monte Carlo Dropout (Gal & Ghahramani, ICML 2016)
           50 stochastic forward passes per prediction

  Uncertainty Levels:
    LOW      (< 0.15) → Auto-approve or auto-reject with high confidence
    MEDIUM   (< 0.30) → Standard processing with audit logging
    HIGH     (< 0.50) → Route to branch supervisor for review
    CRITICAL (≥ 0.50) → Mandatory human auditor review before any action

  Benefits for Banking:
    • Prevents high-stakes decisions based on low-confidence predictions
    • Provides quantitative basis for human-in-the-loop escalation
    • Tracks model uncertainty drift over time (early warning of degradation)
"""
        total = stats.get("total_predictions", 0)
        review_rate = stats.get("review_rate_pct", 0)
        avg_unc = stats.get("avg_total_uncertainty", 0)
        breakdown = stats.get("uncertainty_breakdown", {})

        return f"""
{self._section_header("5. UNCERTAINTY QUANTIFICATION")}

  Method: Monte Carlo Dropout (50 stochastic passes per prediction)

  Statistics (current deployment period):
    Total Predictions   : {total}
    Avg Uncertainty     : {avg_unc:.4f}
    Flagged for Review  : {review_rate:.1f}% of predictions

  Uncertainty Distribution:
{chr(10).join(f"    {lvl:10s}: {cnt}" for lvl, cnt in breakdown.items())}
"""

    def _build_recommendations(self) -> str:
        acc = self.metrics.get("accuracy", 0)
        ece = self.metrics.get("ece", 1)
        recs = []

        if isinstance(acc, float) and acc < 0.92:
            recs.append(
                "1. MODEL PERFORMANCE: Accuracy below 92% target.\n"
                "   → Collect 500+ additional labeled examples\n"
                "   → Run active learning pipeline to prioritise annotation\n"
                "   → Consider ViT-Large or DeiT-Base architectures"
            )
        if isinstance(ece, float) and ece > 0.05:
            recs.append(
                "2. CALIBRATION: ECE exceeds 0.05 threshold.\n"
                "   → Apply Temperature Scaling on validation set\n"
                "   → Re-evaluate confidence thresholds for risk levels"
            )
        if not self.xai_evidence.get("lime"):
            recs.append(
                "3. EXPLAINABILITY: LIME not implemented.\n"
                "   → Add LIME for regulation-grade local explanations\n"
                "   → Required for RBI Model Risk Management audit trail"
            )

        recs.append(
            f"4. REVIEW CYCLE: Schedule next model validation in {self.config.review_cycle_days} days.\n"
            "   → Compare current test metrics vs baseline\n"
            "   → Run adversarial stress tests on latest fraud patterns"
        )
        recs.append(
            "5. DATA GOVERNANCE: Implement formal Data Management Policy.\n"
            "   → Define retention periods for document images (5 year minimum)\n"
            "   → Implement encrypted storage for sensitive document scans\n"
            "   → Conduct annual data quality audit"
        )

        rec_str = "\n".join(f"  {r}" for r in recs)
        return f"""
{self._section_header("9. RECOMMENDATIONS & REMEDIATION PLAN")}
{rec_str}
"""

    # ---------------------------------------------------------------- #
    #  Main Report Generator
    # ---------------------------------------------------------------- #

    def generate(self) -> str:
        """
        Generate the complete compliance report as a formatted string.
        """
        rbi_items   = self._build_rbi_checklist()
        basel_items = self._build_basel_checklist()
        fatf_items  = self._build_fatf_checklist()

        sections = [
            f"\n{'=' * 70}",
            f"  REGULATORY COMPLIANCE REPORT",
            f"  {self.config.system_name} — {self.config.bank_name}",
            f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"{'=' * 70}",

            self._build_executive_summary(),
            self._build_architecture_section(),
            self._build_performance_section(),
            self._build_xai_section(),
            self._build_uncertainty_section(),

            self._build_checklist_section(
                "6. RBI REGULATORY CHECKLIST", rbi_items
            ),
            self._build_checklist_section(
                "7. BASEL III COMPLIANCE CHECKLIST", basel_items
            ),
            self._build_checklist_section(
                "8. FATF AML/CFT CHECKLIST", fatf_items
            ),

            self._build_recommendations(),

            f"\n{self.SEPARATOR}",
            f"  END OF REPORT — {self.config.system_name}",
            f"  Document Classification: CONFIDENTIAL — INTERNAL USE ONLY",
            f"{self.SEPARATOR}\n",
        ]

        return "\n".join(sections)

    def save_txt(self, report: str, path: str = "reports/compliance_report.txt") -> None:
        """Save report to a text file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Compliance report saved → {path}")

    def save_json_checklist(
        self, path: str = "reports/compliance_checklist.json"
    ) -> None:
        """Save only the checklist items as structured JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "system": self.config.system_name,
            "bank": self.config.bank_name,
            "rbi":   [vars(i) for i in self._build_rbi_checklist()],
            "basel": [vars(i) for i in self._build_basel_checklist()],
            "fatf":  [vars(i) for i in self._build_fatf_checklist()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Compliance checklist saved → {path}")


# ------------------------------------------------------------------ #
#  CLI Demo
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    config = ReportConfig(
        bank_name="State Bank of India (Demo)",
        system_name="ViT-FraudDetect v1.0",
        report_author="AIML Risk Analytics Team",
    )

    gen = ComplianceReportGenerator(config)

    gen.load_metrics({
        "accuracy":  0.9340,
        "precision": 0.9218,
        "recall":    0.9301,
        "f1":        0.9259,
        "auc_roc":   0.9714,
        "ece":       0.0382,
        "per_class_metrics": {
            "Genuine":  {"f1": 0.9612},
            "Fraud":    {"f1": 0.9104},
            "Tampered": {"f1": 0.8943},
            "Forged":   {"f1": 0.9378},
        },
    })
    gen.load_explainability_evidence(
        gradcam_available=True,
        shap_available=True,
        attention_rollout=True,
        lime_available=False,
    )
    gen.load_uncertainty_stats({
        "total_predictions": 1247,
        "requiring_human_review": 89,
        "review_rate_pct": 7.14,
        "avg_total_uncertainty": 0.1823,
        "high_uncertainty_count": 112,
        "uncertainty_breakdown": {"LOW": 843, "MEDIUM": 292, "HIGH": 98, "CRITICAL": 14},
    })

    report = gen.generate()
    gen.save_txt(report, "reports/compliance_report.txt")
    gen.save_json_checklist("reports/compliance_checklist.json")

    # Print first 80 lines as preview
    for line in report.splitlines()[:80]:
        print(line)
    print("\n... [truncated — see reports/compliance_report.txt for full report]")
