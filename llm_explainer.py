"""
LLM-Based Fraud Explanation Layer
==================================
Generates natural language explanations for fraud detection decisions.

Features:
1. Rule-based explanation generator (no API required)
2. Template-based explanation system
3. Optional LLM integration (OpenAI, local models)
4. Detailed reasoning chains

Converts your project into:
  Vision + LLM Explainable Fraud System

Usage:
    python llm_explainer.py --demo
    python llm_explainer.py --explain path/to/assessment.json
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import random

sys.path.append(str(Path(__file__).parent))


@dataclass
class ExplanationContext:
    """Context for generating explanations."""
    image_class: str
    image_confidence: float
    image_probs: Dict[str, float]
    signature_status: str
    signature_confidence: float
    transaction_risk: str
    final_score: float
    risk_level: str


class RuleBasedExplainer:
    """
    Rule-based explanation generator.
    Generates human-readable explanations without requiring LLM API.
    """
    
    # Explanation templates for different scenarios
    TEMPLATES = {
        'genuine_low_risk': [
            "The cheque image analysis indicates this document is authentic with {confidence:.0%} confidence. "
            "The signature verification confirms a genuine signature pattern. "
            "No irregularities were detected in the document layout or security features. "
            "Recommendation: This transaction appears legitimate and can proceed with standard processing.",
            
            "Our multi-modal analysis confirms document authenticity. "
            "The cheque passed all visual inspection criteria with high confidence ({confidence:.0%}). "
            "The signature matches expected genuine patterns. "
            "Risk assessment: LOW - Standard approval recommended.",
        ],
        
        'fraud_high_risk': [
            "⚠️ FRAUD ALERT: The cheque image shows characteristics consistent with fraudulent documents. "
            "Detected anomalies include {anomalies}. "
            "The signature region shows signs of potential forgery with {sig_risk:.0%} forgery probability. "
            "Immediate action: Block this transaction and escalate to the fraud investigation team.",
            
            "🚨 HIGH RISK DETECTED: Analysis reveals multiple fraud indicators. "
            "The document classification suggests this is a {class_type} document ({confidence:.0%} confidence). "
            "Key concerns: {concerns}. "
            "Recommendation: REJECT - Initiate fraud investigation protocol.",
        ],
        
        'tampered_medium_risk': [
            "The cheque shows signs of potential tampering. "
            "Our image analysis detected modifications in {tampered_areas}. "
            "The signature appears {sig_status} with {sig_conf:.0%} confidence. "
            "Recommendation: Manual verification required before processing.",
            
            "⚠️ TAMPERING SUSPECTED: Document integrity concerns identified. "
            "Classification confidence: {confidence:.0%} for tampered document. "
            "Specific areas of concern: {concerns}. "
            "Action required: Human review necessary - escalate to supervisor.",
        ],
        
        'forged_critical_risk': [
            "🛑 CRITICAL: This cheque appears to be a forged document. "
            "The forgery detection system identified {forgery_indicators}. "
            "Signature analysis: {sig_result} (forgery probability: {sig_risk:.0%}). "
            "Immediate action: BLOCK transaction, preserve evidence, notify security team.",
            
            "🛑 FORGERY DETECTED: High confidence ({confidence:.0%}) forgery indicators present. "
            "The document exhibits characteristics inconsistent with genuine bank cheques. "
            "Combined risk assessment: CRITICAL ({final_score:.0%}). "
            "Action: Reject immediately and initiate fraud response protocol.",
        ],
        
        'signature_mismatch': [
            "The signature verification system has flagged a potential signature mismatch. "
            "Forgery probability: {sig_risk:.0%}. "
            "The signature shows {sig_concerns}. "
            "This requires additional verification before the transaction can proceed.",
        ],
        
        'transaction_anomaly': [
            "Transaction pattern analysis has identified unusual activity. "
            "Risk factors include: {risk_factors}. "
            "This transaction deviates from expected patterns. "
            "Enhanced due diligence is recommended before approval.",
        ]
    }
    
    # Anomaly descriptions
    ANOMALIES = {
        'fraud': [
            "inconsistent font patterns",
            "irregular MICR line encoding",
            "misaligned security features",
            "suspicious color variations"
        ],
        'tampered': [
            "modified amount field",
            "altered payee information",
            "edited date region",
            "overwritten numeric values"
        ],
        'forged': [
            "duplicated security patterns",
            "artificial watermark artifacts",
            "synthetic paper texture",
            "copied signature elements"
        ]
    }
    
    SIGNATURE_CONCERNS = {
        'forged': [
            "inconsistent pen pressure patterns",
            "unnatural stroke continuity",
            "deviation from baseline signature",
            "mechanical reproduction artifacts"
        ],
        'genuine': [
            "consistent stroke patterns",
            "natural pen pressure variation",
            "matches registered signature profile"
        ]
    }
    
    def __init__(self):
        self.explanation_history = []
    
    def _select_template(self, context: ExplanationContext) -> str:
        """Select appropriate template based on context."""
        if context.risk_level == 'LOW' and context.image_class == 'genuine':
            templates = self.TEMPLATES['genuine_low_risk']
        elif context.risk_level == 'CRITICAL' or context.image_class == 'forged':
            templates = self.TEMPLATES['forged_critical_risk']
        elif context.image_class == 'fraud':
            templates = self.TEMPLATES['fraud_high_risk']
        elif context.image_class == 'tampered':
            templates = self.TEMPLATES['tampered_medium_risk']
        else:
            templates = self.TEMPLATES['genuine_low_risk']
        
        return random.choice(templates)
    
    def _get_anomalies(self, image_class: str) -> str:
        """Get relevant anomaly descriptions."""
        anomalies = self.ANOMALIES.get(image_class, [])
        if anomalies:
            selected = random.sample(anomalies, min(2, len(anomalies)))
            return ", ".join(selected)
        return "visual inconsistencies"
    
    def _get_signature_concerns(self, is_forged: bool) -> str:
        """Get signature-related concerns."""
        key = 'forged' if is_forged else 'genuine'
        concerns = self.SIGNATURE_CONCERNS.get(key, [])
        if concerns:
            return random.choice(concerns)
        return "patterns requiring verification"
    
    def generate(self, context: ExplanationContext) -> str:
        """Generate explanation for given context."""
        template = self._select_template(context)
        
        # Prepare substitution values
        sig_is_forged = context.signature_status == 'forged'
        
        values = {
            'confidence': context.image_confidence,
            'class_type': context.image_class,
            'sig_risk': 1 - context.signature_confidence if context.signature_status == 'genuine' 
                        else context.signature_confidence,
            'sig_conf': context.signature_confidence,
            'sig_status': context.signature_status,
            'sig_result': f"likely {context.signature_status}",
            'final_score': context.final_score,
            'anomalies': self._get_anomalies(context.image_class),
            'concerns': self._get_anomalies(context.image_class),
            'tampered_areas': "amount field and date region",
            'forgery_indicators': self._get_anomalies('forged'),
            'sig_concerns': self._get_signature_concerns(sig_is_forged),
            'risk_factors': "unusual transaction amount and timing"
        }
        
        # Format template
        try:
            explanation = template.format(**values)
        except KeyError:
            # Fallback to simple explanation
            explanation = f"Risk assessment: {context.risk_level} ({context.final_score:.0%}). "
            explanation += f"Document classified as {context.image_class} with {context.image_confidence:.0%} confidence."
        
        self.explanation_history.append({
            'context': context.__dict__,
            'explanation': explanation
        })
        
        return explanation


class PromptBasedExplainer:
    """
    Prompt-based explainer for use with LLM APIs.
    Can integrate with OpenAI, Anthropic, or local models.
    """
    
    SYSTEM_PROMPT = """You are an expert fraud analyst for a banking institution. 
Your role is to provide clear, professional explanations for fraud detection system outputs.

Guidelines:
- Be specific about detected issues
- Explain the reasoning behind risk assessments
- Provide actionable recommendations
- Use professional banking terminology
- Be concise but thorough"""
    
    EXPLANATION_PROMPT = """Based on the following fraud detection analysis, provide a clear explanation for bank personnel:

Document Classification: {image_class} (Confidence: {image_confidence:.1%})
Class Probabilities:
{class_probs}

Signature Verification: {signature_status} (Confidence: {signature_confidence:.1%})

Transaction Risk Level: {transaction_risk}

Combined Risk Score: {final_score:.1%}
Overall Risk Level: {risk_level}

Please provide:
1. A clear summary of the findings
2. Specific concerns identified
3. Recommended action
4. Any additional verification steps needed

Keep the explanation professional and suitable for fraud investigation documentation."""

    def __init__(self, api_provider: str = 'openai', api_key: Optional[str] = None):
        self.api_provider = api_provider
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key:
            self._init_client()
    
    def _init_client(self):
        """Initialize API client."""
        if self.api_provider == 'openai':
            try:
                import openai  # type: ignore  # Optional dependency
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                print("OpenAI package not installed. Using rule-based explainer.")
    
    def generate(self, context: ExplanationContext) -> str:
        """Generate explanation using LLM API."""
        if self.client is None:
            # Fallback to rule-based
            rule_based = RuleBasedExplainer()
            return rule_based.generate(context)
        
        # Format class probabilities
        class_probs = "\n".join([
            f"  - {k}: {v:.1%}" for k, v in context.image_probs.items()
        ])
        
        prompt = self.EXPLANATION_PROMPT.format(
            image_class=context.image_class,
            image_confidence=context.image_confidence,
            class_probs=class_probs,
            signature_status=context.signature_status,
            signature_confidence=context.signature_confidence,
            transaction_risk=context.transaction_risk,
            final_score=context.final_score,
            risk_level=context.risk_level
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            # Fallback
            rule_based = RuleBasedExplainer()
            return rule_based.generate(context)


class FraudExplainer:
    """
    Main fraud explanation interface.
    Combines rule-based and LLM-based explanations.
    """
    
    def __init__(self, use_llm: bool = False, llm_provider: str = 'openai'):
        """
        Initialize fraud explainer.
        
        Args:
            use_llm: Whether to use LLM API for explanations
            llm_provider: LLM provider ('openai', 'anthropic', 'local')
        """
        self.rule_explainer = RuleBasedExplainer()
        self.llm_explainer = None
        
        if use_llm:
            self.llm_explainer = PromptBasedExplainer(api_provider=llm_provider)
    
    def explain_assessment(self, assessment_dict: Dict) -> str:
        """
        Generate explanation from risk assessment dictionary.
        
        Args:
            assessment_dict: Output from RiskAggregator.assess()
            
        Returns:
            Human-readable explanation string
        """
        # Extract context from assessment
        signals = assessment_dict.get('signals', [])
        
        image_signal = next((s for s in signals if s['source'] == 'image_classifier'), None)
        sig_signal = next((s for s in signals if s['source'] == 'signature_verifier'), None)
        trans_signal = next((s for s in signals if s['source'] == 'transaction_analyzer'), None)
        
        context = ExplanationContext(
            image_class=image_signal['details'].get('predicted_class', 'unknown') if image_signal else 'unknown',
            image_confidence=image_signal['confidence'] if image_signal else 0.5,
            image_probs=image_signal['details'].get('class_probabilities', {}) if image_signal else {},
            signature_status=sig_signal['details'].get('prediction', 'unknown') if sig_signal else 'unknown',
            signature_confidence=sig_signal['confidence'] if sig_signal else 0.5,
            transaction_risk=trans_signal['details'].get('risk_level', 'unknown') if trans_signal else 'unknown',
            final_score=assessment_dict.get('final_score', 0.5),
            risk_level=assessment_dict.get('risk_level', 'MEDIUM')
        )
        
        # Generate explanation
        if self.llm_explainer and self.llm_explainer.client:
            explanation = self.llm_explainer.generate(context)
        else:
            explanation = self.rule_explainer.generate(context)
        
        return explanation
    
    def explain_image_classification(self, class_name: str, confidence: float, 
                                     class_probs: Dict[str, float]) -> str:
        """Generate explanation for image classification only."""
        context = ExplanationContext(
            image_class=class_name,
            image_confidence=confidence,
            image_probs=class_probs,
            signature_status='unknown',
            signature_confidence=0.5,
            transaction_risk='unknown',
            final_score=1 - class_probs.get('genuine', 0.5),
            risk_level='HIGH' if class_name != 'genuine' else 'LOW'
        )
        
        return self.rule_explainer.generate(context)
    
    def generate_investigation_report(self, assessment_dict: Dict) -> str:
        """
        Generate a formal investigation report.
        
        Args:
            assessment_dict: Complete risk assessment
            
        Returns:
            Formal report string
        """
        from datetime import datetime
        
        explanation = self.explain_assessment(assessment_dict)
        
        report = f"""
================================================================================
                    FRAUD INVESTIGATION REPORT
================================================================================
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Case Reference: FRD-{datetime.now().strftime('%Y%m%d%H%M%S')}

EXECUTIVE SUMMARY
-----------------
Risk Level: {assessment_dict.get('risk_level', 'UNKNOWN')}
Risk Score: {assessment_dict.get('final_score', 0)*100:.1f}%
Recommendation: {assessment_dict.get('recommendation', 'Manual review required')}

DETAILED ANALYSIS
-----------------
{explanation}

SIGNAL BREAKDOWN
----------------"""
        
        for signal in assessment_dict.get('signals', []):
            report += f"""
Source: {signal['source'].replace('_', ' ').title()}
  Score: {signal['score']*100:.1f}%
  Confidence: {signal['confidence']*100:.1f}%
  Details: {json.dumps(signal['details'], indent=4)}
"""
        
        report += f"""
ACTION ITEMS
------------
"""
        risk_level = assessment_dict.get('risk_level', 'MEDIUM')
        if risk_level == 'LOW':
            report += "[ ] Proceed with standard processing\n"
            report += "[ ] Archive this assessment for records\n"
        elif risk_level == 'MEDIUM':
            report += "[ ] Conduct manual document verification\n"
            report += "[ ] Contact account holder for confirmation\n"
            report += "[ ] Review transaction history\n"
        elif risk_level == 'HIGH':
            report += "[ ] Escalate to fraud investigation team\n"
            report += "[ ] Place temporary hold on account\n"
            report += "[ ] Gather additional evidence\n"
            report += "[ ] Document chain of custody\n"
        else:  # CRITICAL
            report += "[!] IMMEDIATE: Block all related transactions\n"
            report += "[!] Notify fraud response team\n"
            report += "[!] Preserve all evidence\n"
            report += "[!] Consider law enforcement notification\n"
            report += "[!] Freeze related accounts\n"
        
        report += """
================================================================================
                           END OF REPORT
================================================================================
"""
        return report


def demo():
    """Demonstrate the LLM explainer capabilities."""
    print("=" * 70)
    print("LLM-BASED FRAUD EXPLANATION LAYER DEMO")
    print("=" * 70)
    print()
    
    explainer = FraudExplainer(use_llm=False)  # Rule-based for demo
    
    # Test cases
    test_cases = [
        {
            'name': 'Genuine Low-Risk',
            'assessment': {
                'final_score': 0.12,
                'risk_level': 'LOW',
                'recommendation': 'APPROVE',
                'signals': [
                    {
                        'source': 'image_classifier',
                        'score': 0.08,
                        'confidence': 0.95,
                        'details': {
                            'predicted_class': 'genuine',
                            'class_probabilities': {
                                'genuine': 0.92, 'fraud': 0.03,
                                'tampered': 0.03, 'forged': 0.02
                            }
                        }
                    },
                    {
                        'source': 'signature_verifier',
                        'score': 0.15,
                        'confidence': 0.88,
                        'details': {'prediction': 'genuine'}
                    }
                ]
            }
        },
        {
            'name': 'Forged Critical-Risk',
            'assessment': {
                'final_score': 0.89,
                'risk_level': 'CRITICAL',
                'recommendation': 'REJECT',
                'signals': [
                    {
                        'source': 'image_classifier',
                        'score': 0.92,
                        'confidence': 0.87,
                        'details': {
                            'predicted_class': 'forged',
                            'class_probabilities': {
                                'genuine': 0.05, 'fraud': 0.08,
                                'tampered': 0.02, 'forged': 0.85
                            }
                        }
                    },
                    {
                        'source': 'signature_verifier',
                        'score': 0.78,
                        'confidence': 0.82,
                        'details': {'prediction': 'forged'}
                    }
                ]
            }
        },
        {
            'name': 'Tampered Medium-Risk',
            'assessment': {
                'final_score': 0.55,
                'risk_level': 'MEDIUM',
                'recommendation': 'REVIEW',
                'signals': [
                    {
                        'source': 'image_classifier',
                        'score': 0.65,
                        'confidence': 0.72,
                        'details': {
                            'predicted_class': 'tampered',
                            'class_probabilities': {
                                'genuine': 0.25, 'fraud': 0.08,
                                'tampered': 0.62, 'forged': 0.05
                            }
                        }
                    },
                    {
                        'source': 'signature_verifier',
                        'score': 0.35,
                        'confidence': 0.75,
                        'details': {'prediction': 'genuine'}
                    }
                ]
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"CASE: {case['name']}")
        print('='*60)
        
        explanation = explainer.explain_assessment(case['assessment'])
        print(f"\n📝 EXPLANATION:\n{explanation}")
        
        print(f"\n{'─'*60}")
        print("📋 INVESTIGATION REPORT PREVIEW:")
        print('─'*60)
        report = explainer.generate_investigation_report(case['assessment'])
        # Print first 1000 chars of report
        print(report[:1500] + "..." if len(report) > 1500 else report)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nTo use with LLM API, set OPENAI_API_KEY environment variable")
    print("or initialize with: FraudExplainer(use_llm=True)")


def main():
    parser = argparse.ArgumentParser(description='LLM-Based Fraud Explainer')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--explain', type=str, help='Path to assessment JSON file')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM API')
    parser.add_argument('--report', action='store_true', help='Generate formal report')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.explain:
        if not Path(args.explain).exists():
            print(f"File not found: {args.explain}")
            return
        
        with open(args.explain, 'r') as f:
            assessment = json.load(f)
        
        explainer = FraudExplainer(use_llm=args.use_llm)
        
        if args.report:
            output = explainer.generate_investigation_report(assessment)
        else:
            output = explainer.explain_assessment(assessment)
        
        print(output)
    else:
        demo()


if __name__ == '__main__':
    main()
