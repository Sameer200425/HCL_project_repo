"""
Explainability Report Generator for Banking Compliance.

Generates PDF audit reports containing:
  - Document image
  - Prediction with confidence score
  - Attention heatmap
  - Grad-CAM heatmap
  - SHAP feature importance
  - Decision traceability log

Essential for banking regulations and audit compliance.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from fpdf import FPDF  # type: ignore[import-untyped]
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ------------------------------------------------------------------ #
#  Report Generator
# ------------------------------------------------------------------ #
class ExplainabilityReport:
    """
    Generate PDF compliance reports for fraud detection decisions.
    Provides decision traceability for bank auditors.
    """

    def __init__(
        self,
        output_dir: str = "reports/",
        title: str = "Financial Document Fraud Detection Report",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title

    def generate(
        self,
        document_image: np.ndarray,
        prediction: str,
        confidence: float,
        class_probabilities: Dict[str, float],
        attention_overlay: Optional[np.ndarray] = None,
        gradcam_overlay: Optional[np.ndarray] = None,
        shap_importance: Optional[np.ndarray] = None,
        document_id: str = "DOC-UNKNOWN",
        analyst_notes: str = "",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Generate a comprehensive PDF report.

        Args:
            document_image: Original document image (RGB).
            prediction: Predicted class label.
            confidence: Prediction confidence (0-1).
            class_probabilities: Per-class probabilities.
            attention_overlay: Attention heatmap overlay image.
            gradcam_overlay: Grad-CAM overlay image.
            shap_importance: SHAP importance map.
            document_id: Document identifier.
            analyst_notes: Optional notes.
            metadata: Additional metadata dict.

        Returns:
            Path to generated PDF report.
        """
        if not FPDF_AVAILABLE:
            return self._generate_text_report(
                prediction, confidence, class_probabilities,
                document_id, analyst_notes, metadata
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"report_{document_id}_{timestamp}.pdf"
        report_path = self.output_dir / report_name
        temp_dir = self.output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Save images to temp files for FPDF
        img_paths = {}
        if document_image is not None:
            path = str(temp_dir / "original.png")
            plt.imsave(path, document_image)
            img_paths["original"] = path

        if attention_overlay is not None:
            path = str(temp_dir / "attention.png")
            plt.imsave(path, attention_overlay)
            img_paths["attention"] = path

        if gradcam_overlay is not None:
            path = str(temp_dir / "gradcam.png")
            plt.imsave(path, gradcam_overlay)
            img_paths["gradcam"] = path

        if shap_importance is not None:
            path = str(temp_dir / "shap.png")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(shap_importance, cmap="hot")
            ax.axis("off")
            fig.savefig(path, bbox_inches="tight", dpi=100)
            plt.close(fig)
            img_paths["shap"] = path

        # Build PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Title Page ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 15, self.title, new_x="LMARGIN", new_y="NEXT", align="C")

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, f"Document ID: {document_id}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(
            0, 8,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.cell(0, 8, f"Model: Vision Transformer (ViT-Base)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # --- Prediction Summary ---
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "1. Prediction Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)

        risk_color = (220, 50, 50) if prediction in ["fraud", "tampered", "forged"] else (50, 150, 50)
        pdf.set_text_color(*risk_color)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"Prediction: {prediction.upper()}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 10, f"Confidence: {confidence:.1%}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 12)
        pdf.ln(3)

        pdf.cell(0, 8, "Class Probabilities:", new_x="LMARGIN", new_y="NEXT")
        for cls, prob in class_probabilities.items():
            bar_width = int(prob * 100)
            pdf.cell(0, 7, f"  {cls}: {prob:.4f} {'|' * bar_width}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # --- Document Image ---
        if "original" in img_paths:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "2. Document Image", new_x="LMARGIN", new_y="NEXT")
            pdf.image(img_paths["original"], w=100)
            pdf.ln(5)

        # --- Attention Map ---
        if "attention" in img_paths:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "3. Attention Map Analysis", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(
                0, 6,
                "The attention map shows which regions of the document the model "
                "focused on during classification. Brighter regions indicate higher "
                "attention, suggesting areas most relevant to the fraud decision."
            )
            pdf.ln(3)
            pdf.image(img_paths["attention"], w=140)
            pdf.ln(5)

        # --- Grad-CAM ---
        if "gradcam" in img_paths:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "4. Grad-CAM Analysis", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(
                0, 6,
                "Grad-CAM highlights specific regions that most influenced the "
                "classification decision. Red/warm areas are most important. "
                "This can reveal tampered areas, edited text, or forged signatures."
            )
            pdf.ln(3)
            pdf.image(img_paths["gradcam"], w=140)
            pdf.ln(5)

        # --- SHAP ---
        if "shap" in img_paths:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "5. SHAP Feature Importance", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(
                0, 6,
                "SHAP values provide feature-level importance scores based on "
                "Shapley values from game theory. Each patch's contribution to "
                "the final decision is quantified, providing a rigorous basis "
                "for the model's explanation."
            )
            pdf.ln(3)
            pdf.image(img_paths["shap"], w=100)
            pdf.ln(5)

        # --- Decision Traceability ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "6. Decision Traceability", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)

        trace_items = [
            f"Timestamp: {datetime.now().isoformat()}",
            f"Document ID: {document_id}",
            f"Model Architecture: Vision Transformer (ViT-Base/16)",
            f"Image Size: 224x224 pixels",
            f"Patch Size: 16x16",
            f"Number of Patches: 196",
            f"Encoder Layers: 12",
            f"Attention Heads: 12",
            f"Predicted Class: {prediction}",
            f"Confidence: {confidence:.4f}",
            f"XAI Methods Applied: Attention Rollout, Grad-CAM, SHAP",
        ]

        if metadata:
            for k, v in metadata.items():
                trace_items.append(f"{k}: {v}")

        for item in trace_items:
            pdf.cell(0, 7, f"  * {item}", new_x="LMARGIN", new_y="NEXT")

        # --- Analyst Notes ---
        if analyst_notes:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Analyst Notes:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, analyst_notes)

        # --- Footer ---
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(
            0, 6,
            "This report is auto-generated by the ViT Fraud Detection System. "
            "For audit purposes only.",
            new_x="LMARGIN", new_y="NEXT",
            align="C",
        )

        # Save
        pdf.output(str(report_path))

        # Cleanup temp images
        for p in img_paths.values():
            if os.path.exists(p):
                os.remove(p)

        print(f"[REPORT] Generated: {report_path}")
        return str(report_path)

    def _generate_text_report(
        self,
        prediction: str,
        confidence: float,
        class_probabilities: Dict[str, float],
        document_id: str,
        analyst_notes: str,
        metadata: Optional[Dict],
    ) -> str:
        """Fallback text report when fpdf2 is not available."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"report_{document_id}_{timestamp}.txt"

        lines = [
            "=" * 60,
            self.title,
            "=" * 60,
            f"Document ID: {document_id}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "PREDICTION SUMMARY",
            "-" * 30,
            f"Prediction: {prediction.upper()}",
            f"Confidence: {confidence:.4f} ({confidence:.1%})",
            "",
            "Class Probabilities:",
        ]
        for cls, prob in class_probabilities.items():
            lines.append(f"  {cls}: {prob:.4f}")

        lines += ["", "DECISION TRACE", "-" * 30]
        if metadata:
            for k, v in metadata.items():
                lines.append(f"  {k}: {v}")

        if analyst_notes:
            lines += ["", "ANALYST NOTES", "-" * 30, analyst_notes]

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        print(f"[REPORT] Generated text report: {report_path}")
        return str(report_path)
