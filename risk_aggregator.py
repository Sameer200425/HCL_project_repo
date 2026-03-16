"""
Multi-Modal Banking Fraud Risk Aggregator
==========================================
Combines multiple fraud detection signals into unified risk score:

1. Image Fraud Detection (cheque classification)
2. Signature Verification (genuine vs forged)
3. Transaction Anomaly Detection (risk scoring)

Formula:
  Final Risk Score = w1 * Image_Fraud_Prob 
                   + w2 * Signature_Mismatch_Score
                   + w3 * Transaction_Risk_Score

This creates an Enterprise-Level Multi-Modal Fraud Detection System.

Usage:
    python risk_aggregator.py --demo
    python risk_aggregator.py --analyze path/to/cheque.png
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).parent))


@dataclass
class FraudSignal:
    """Individual fraud detection signal."""
    source: str
    score: float  # 0-1, higher = more suspicious
    confidence: float
    details: Dict = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    final_score: float  # 0-1
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommendation: str
    signals: List[FraudSignal] = field(default_factory=list)
    explanation: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'final_score': self.final_score,
            'risk_level': self.risk_level,
            'recommendation': self.recommendation,
            'signals': [
                {
                    'source': s.source,
                    'score': s.score,
                    'confidence': s.confidence,
                    'details': s.details
                } for s in self.signals
            ],
            'explanation': self.explanation,
            'timestamp': self.timestamp
        }


class ImageFraudAnalyzer:
    """Analyzes cheque images for fraud indicators."""
    
    def __init__(self, model_path: str = 'checkpoints/cnn_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = ['genuine', 'fraud', 'tampered', 'forged']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load classification model."""
        try:
            from models.hybrid_model import CNNBaseline
            self.model = CNNBaseline(num_classes=4, pretrained=False)
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Loaded image classifier from {path}")
        except Exception as e:
            print(f"⚠️ Could not load image classifier: {e}")
    
    def analyze(self, image: Image.Image) -> FraudSignal:
        """Analyze image for fraud indicators."""
        if self.model is None:
            return FraudSignal(
                source='image_classifier',
                score=0.5,
                confidence=0.0,
                details={'error': 'Model not loaded'}
            )
        
        image_tensor: torch.Tensor = self.transform(image).unsqueeze(0).to(self.device)  # type: ignore[union-attr]
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
        
        # Fraud score = 1 - P(genuine)
        genuine_prob = probs[0, 0].item()
        fraud_score = 1 - genuine_prob
        
        return FraudSignal(
            source='image_classifier',
            score=fraud_score,
            confidence=probs[0, pred_class].item(),
            details={
                'predicted_class': self.class_names[pred_class],
                'class_probabilities': {
                    name: probs[0, i].item() 
                    for i, name in enumerate(self.class_names)
                }
            }
        )


class SignatureAnalyzer:
    """Analyzes signatures for verification."""
    
    def __init__(self, model_path: str = 'checkpoints/signature_vit_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load signature verification model."""
        try:
            from data_integration.cedar_signature_loader import SignatureVerificationViT
            self.model = SignatureVerificationViT()
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Loaded signature verifier from {path}")
        except Exception as e:
            print(f"⚠️ Could not load signature verifier: {e}")
    
    def extract_signature_region(self, cheque_image: Image.Image) -> Image.Image:
        """Extract signature region from cheque."""
        width, height = cheque_image.size
        # Typical signature location: bottom-right quadrant
        left = int(width * 0.55)
        top = int(height * 0.65)
        right = int(width * 0.95)
        bottom = int(height * 0.90)
        
        signature = cheque_image.crop((left, top, right, bottom))
        return signature
    
    def analyze(self, signature_image: Image.Image) -> FraudSignal:
        """Analyze signature for authenticity."""
        if self.model is None:
            # Return moderate risk if model not available
            return FraudSignal(
                source='signature_verifier',
                score=0.5,
                confidence=0.0,
                details={'error': 'Model not loaded', 'status': 'unavailable'}
            )
        
        image_tensor: torch.Tensor = self.transform(signature_image).unsqueeze(0).to(self.device)  # type: ignore[union-attr]
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
        
        # Forgery score = P(forged)
        forgery_score = probs[0, 1].item()
        
        return FraudSignal(
            source='signature_verifier',
            score=forgery_score,
            confidence=probs[0, pred_class].item(),
            details={
                'prediction': 'genuine' if pred_class == 0 else 'forged',
                'genuine_probability': probs[0, 0].item(),
                'forged_probability': probs[0, 1].item()
            }
        )


class TransactionAnalyzer:
    """Analyzes transaction data for anomalies."""
    
    def __init__(self, model_path: str = 'checkpoints/risk_model.pkl'):
        self.model = None
        self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load transaction risk model."""
        try:
            from data_integration.creditcard_fraud_loader import TransactionRiskScorer
            self.model = TransactionRiskScorer(path)
            print(f"✅ Loaded transaction risk model from {path}")
        except Exception as e:
            print(f"⚠️ Could not load transaction risk model: {e}")
    
    def analyze(self, features: np.ndarray) -> FraudSignal:
        """Analyze transaction features for fraud risk."""
        if self.model is None:
            return FraudSignal(
                source='transaction_analyzer',
                score=0.5,
                confidence=0.0,
                details={'error': 'Model not loaded', 'status': 'unavailable'}
            )
        
        result = self.model.predict_risk(features)
        
        return FraudSignal(
            source='transaction_analyzer',
            score=result['risk_score'][0],
            confidence=0.85,  # Ensemble model confidence
            details={
                'risk_level': result['risk_level'][0],
                'is_fraud': result['is_fraud'][0],
                'model_scores': result['model_scores']
            }
        )


class RiskAggregator:
    """
    Multi-Modal Banking Fraud Risk Aggregator
    
    Combines signals from multiple detection systems:
    - Image fraud detection
    - Signature verification
    - Transaction anomaly detection
    
    Final Risk Score = w1 * Image + w2 * Signature + w3 * Transaction
    """
    
    # Default weights (can be configured)
    DEFAULT_WEIGHTS = {
        'image_classifier': 0.5,
        'signature_verifier': 0.3,
        'transaction_analyzer': 0.2
    }
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        'LOW': 0.2,
        'MEDIUM': 0.4,
        'HIGH': 0.6,
        'CRITICAL': 0.8
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize risk aggregator.
        
        Args:
            weights: Custom weights for each signal source
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Initialize analyzers
        print("Initializing Multi-Modal Fraud Detection System...")
        print("-" * 50)
        
        self.image_analyzer = ImageFraudAnalyzer()
        self.signature_analyzer = SignatureAnalyzer()
        self.transaction_analyzer = TransactionAnalyzer()
        
        print("-" * 50)
        print("System ready.\n")
    
    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        if score < self.RISK_THRESHOLDS['LOW']:
            return 'LOW'
        elif score < self.RISK_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        elif score < self.RISK_THRESHOLDS['HIGH']:
            return 'HIGH'
        elif score < self.RISK_THRESHOLDS['CRITICAL']:
            return 'CRITICAL'
        else:
            return 'CRITICAL'
    
    def _get_recommendation(self, risk_level: str, signals: List[FraudSignal]) -> str:
        """Generate recommendation based on risk level and signals."""
        recommendations = {
            'LOW': "✅ APPROVE - Transaction appears legitimate. Standard processing recommended.",
            'MEDIUM': "⚠️ REVIEW - Moderate risk detected. Manual verification recommended before processing.",
            'HIGH': "🚨 FLAG - High risk indicators detected. Escalate to fraud investigation team.",
            'CRITICAL': "🛑 REJECT - Critical fraud indicators detected. Block transaction and initiate investigation."
        }
        
        recommendation = recommendations.get(risk_level, recommendations['MEDIUM'])
        
        # Add specific concerns
        concerns = []
        for signal in signals:
            if signal.score > 0.6:
                if signal.source == 'image_classifier':
                    concerns.append("cheque image tampering")
                elif signal.source == 'signature_verifier':
                    concerns.append("signature forgery")
                elif signal.source == 'transaction_analyzer':
                    concerns.append("unusual transaction pattern")
        
        if concerns:
            recommendation += f"\n   Concerns: {', '.join(concerns)}"
        
        return recommendation
    
    def _generate_explanation(self, signals: List[FraudSignal], final_score: float) -> str:
        """Generate human-readable explanation of the assessment."""
        explanation_parts = []
        
        explanation_parts.append(f"Risk Assessment Summary (Score: {final_score:.2%})")
        explanation_parts.append("-" * 40)
        
        for signal in signals:
            source_name = signal.source.replace('_', ' ').title()
            
            if signal.score < 0.3:
                status = "appears legitimate"
            elif signal.score < 0.6:
                status = "shows some concerns"
            else:
                status = "indicates potential fraud"
            
            explanation_parts.append(f"• {source_name}: {status} (confidence: {signal.confidence:.0%})")
            
            # Add specific details
            if 'predicted_class' in signal.details:
                explanation_parts.append(f"  - Detected as: {signal.details['predicted_class']}")
            if 'prediction' in signal.details:
                explanation_parts.append(f"  - Signature: {signal.details['prediction']}")
            if 'risk_level' in signal.details:
                explanation_parts.append(f"  - Transaction risk: {signal.details['risk_level']}")
        
        return '\n'.join(explanation_parts)
    
    def assess(self, 
               cheque_image: Optional[Image.Image] = None,
               signature_image: Optional[Image.Image] = None,
               transaction_features: Optional[np.ndarray] = None,
               extract_signature: bool = True) -> RiskAssessment:
        """
        Perform comprehensive fraud risk assessment.
        
        Args:
            cheque_image: Full cheque image (PIL Image)
            signature_image: Signature image (optional, extracted if not provided)
            transaction_features: Transaction features array (optional)
            extract_signature: Whether to extract signature from cheque image
            
        Returns:
            RiskAssessment with combined risk score and details
        """
        signals = []
        weighted_sum = 0
        total_weight = 0
        
        # 1. Image Fraud Analysis
        if cheque_image is not None:
            image_signal = self.image_analyzer.analyze(cheque_image)
            signals.append(image_signal)
            
            weight = self.weights.get('image_classifier', 0.5)
            weighted_sum += image_signal.score * weight * image_signal.confidence
            total_weight += weight * image_signal.confidence
        
        # 2. Signature Analysis
        if signature_image is not None:
            sig_signal = self.signature_analyzer.analyze(signature_image)
            signals.append(sig_signal)
            
            weight = self.weights.get('signature_verifier', 0.3)
            if sig_signal.confidence > 0:
                weighted_sum += sig_signal.score * weight * sig_signal.confidence
                total_weight += weight * sig_signal.confidence
                
        elif cheque_image is not None and extract_signature:
            # Extract signature from cheque
            signature = self.signature_analyzer.extract_signature_region(cheque_image)
            sig_signal = self.signature_analyzer.analyze(signature)
            signals.append(sig_signal)
            
            weight = self.weights.get('signature_verifier', 0.3)
            if sig_signal.confidence > 0:
                weighted_sum += sig_signal.score * weight * sig_signal.confidence
                total_weight += weight * sig_signal.confidence
        
        # 3. Transaction Analysis
        if transaction_features is not None:
            trans_signal = self.transaction_analyzer.analyze(transaction_features)
            signals.append(trans_signal)
            
            weight = self.weights.get('transaction_analyzer', 0.2)
            if trans_signal.confidence > 0:
                weighted_sum += trans_signal.score * weight * trans_signal.confidence
                total_weight += weight * trans_signal.confidence
        
        # Calculate final score
        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.5  # Unknown risk
        
        # Determine risk level and recommendation
        risk_level = self._get_risk_level(final_score)
        recommendation = self._get_recommendation(risk_level, signals)
        explanation = self._generate_explanation(signals, final_score)
        
        return RiskAssessment(
            final_score=final_score,
            risk_level=risk_level,
            recommendation=recommendation,
            signals=signals,
            explanation=explanation
        )
    
    def batch_assess(self, items: List[Dict]) -> List[RiskAssessment]:
        """
        Assess multiple items in batch.
        
        Args:
            items: List of dicts with 'cheque_image', 'signature_image', 'transaction_features'
            
        Returns:
            List of RiskAssessment objects
        """
        results = []
        for item in items:
            result = self.assess(
                cheque_image=item.get('cheque_image'),
                signature_image=item.get('signature_image'),
                transaction_features=item.get('transaction_features')
            )
            results.append(result)
        return results


def demo():
    """Run demonstration of risk aggregator."""
    print("=" * 70)
    print("MULTI-MODAL BANKING FRAUD RISK AGGREGATOR DEMO")
    print("=" * 70)
    print()
    
    # Initialize aggregator
    aggregator = RiskAggregator()
    
    # Test with sample images
    data_dir = Path('data/fraud_dataset')
    
    test_cases = []
    
    # Find sample images from each class
    for class_name in ['genuine', 'fraud', 'tampered', 'forged']:
        class_dir = data_dir / class_name
        if class_dir.exists():
            images = list(class_dir.glob('*.png'))
            if images:
                test_cases.append({
                    'name': f"Sample {class_name} cheque",
                    'image_path': str(images[0]),
                    'expected': class_name
                })
    
    if not test_cases:
        print("No test images found. Using blank placeholder for demo...")
        # Create a blank placeholder image (no real data available)
        test_img = Image.new('RGB', (224, 224), color='white')
        test_cases = [{'name': 'Placeholder test (add real data)', 'image': test_img, 'expected': 'unknown'}]
    
    # Run assessments
    print("\n" + "=" * 70)
    print("RUNNING ASSESSMENTS")
    print("=" * 70)
    
    for case in test_cases:
        print(f"\n{'─' * 60}")
        print(f"📋 {case['name']} (Expected: {case['expected']})")
        print('─' * 60)
        
        if 'image_path' in case:
            image = Image.open(case['image_path']).convert('RGB')
        else:
            image = case['image']
        
        # Generate random transaction features for demo
        transaction_features = np.random.randn(28)
        
        # Assess
        result = aggregator.assess(
            cheque_image=image,
            transaction_features=transaction_features
        )
        
        # Print results
        print(f"\n🎯 FINAL RISK SCORE: {result.final_score:.1%}")
        print(f"📊 RISK LEVEL: {result.risk_level}")
        print(f"\n{result.recommendation}")
        print(f"\n{result.explanation}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Fraud Risk Aggregator')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--analyze', type=str, help='Analyze specific cheque image')
    parser.add_argument('--output', type=str, default='results/risk_assessment.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.analyze:
        if not Path(args.analyze).exists():
            print(f"Error: Image not found: {args.analyze}")
            return
        
        aggregator = RiskAggregator()
        image = Image.open(args.analyze).convert('RGB')
        
        result = aggregator.assess(cheque_image=image)
        
        print("\n" + "=" * 50)
        print("RISK ASSESSMENT RESULT")
        print("=" * 50)
        print(f"Final Score: {result.final_score:.1%}")
        print(f"Risk Level: {result.risk_level}")
        print(f"\n{result.recommendation}")
        print(f"\n{result.explanation}")
        
        # Save result
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n✅ Results saved to: {args.output}")
    else:
        demo()


if __name__ == '__main__':
    main()
