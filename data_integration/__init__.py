"""
Data Integration Module
=======================
Provides loaders for external datasets to enhance fraud detection.

Available loaders:
- RVL-CDIP: Document images for ViT pretraining
- CEDAR: Signature verification
- Credit Card Fraud: Transaction risk scoring

Usage:
    from data_integration import UnifiedDataManager
    
    manager = UnifiedDataManager()
    manager.setup_sample_data()
    manager.train_all_models()
"""

from data_integration.unified_loader import UnifiedDataManager
from data_integration.cedar_signature_loader import CEDARSignatureDataset, SignatureVerificationViT
from data_integration.creditcard_fraud_loader import TransactionRiskScorer, MultiModalFraudDetector
from data_integration.rvl_cdip_loader import RVLCDIPDataset

__all__ = [
    'UnifiedDataManager',
    'RVLCDIPDataset',
    'CEDARSignatureDataset',
    'SignatureVerificationViT',
    'TransactionRiskScorer',
    'MultiModalFraudDetector'
]
