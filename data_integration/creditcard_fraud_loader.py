"""
Credit Card Fraud Dataset Integration
======================================
Integrates Kaggle Credit Card Fraud dataset for transaction
risk scoring and multi-modal fraud detection.

Dataset: 284,807 transactions (492 frauds, 0.17% imbalanced)

Usage:
    python data_integration/creditcard_fraud_loader.py --download-info
    python data_integration/creditcard_fraud_loader.py --create-sample
    python data_integration/creditcard_fraud_loader.py --train
    python data_integration/creditcard_fraud_loader.py --evaluate
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))


class TransactionRiskScorer:
    """
    Machine learning model for transaction fraud risk scoring.
    Uses ensemble of models for robust predictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def _prepare_data(self, X, y=None, fit_scaler=False):
        """Prepare and scale features."""
        from sklearn.preprocessing import StandardScaler
        
        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit(self, X, y, feature_names=None):
        """
        Train ensemble of models for fraud detection.
        
        Args:
            X: Feature matrix
            y: Labels (0=legitimate, 1=fraud)
            feature_names: Optional list of feature names
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, roc_auc_score
        
        print("=" * 60)
        print("Training Transaction Risk Scoring Models")
        print("=" * 60)
        
        self.feature_names = feature_names or [f'V{i}' for i in range(X.shape[1])]
        
        # Prepare data
        X_scaled = self._prepare_data(X, fit_scaler=True)
        
        # Handle class imbalance
        fraud_count = y.sum()
        legitimate_count = len(y) - fraud_count
        class_weight = {0: 1.0, 1: legitimate_count / fraud_count}
        
        print(f"Training samples: {len(y)}")
        print(f"Fraud cases: {fraud_count} ({100*fraud_count/len(y):.2f}%)")
        print(f"Class weight for fraud: {class_weight[1]:.2f}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train multiple models
        print("\nTraining ensemble models...")
        
        # 1. Random Forest
        print("  Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train)
        
        # 2. Gradient Boosting
        print("  Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gb'].fit(X_train, y_train)
        
        # 3. Logistic Regression
        print("  Training Logistic Regression...")
        self.models['lr'] = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        )
        self.models['lr'].fit(X_train, y_train)
        
        # Evaluate ensemble
        print("\n" + "=" * 40)
        print("Validation Results")
        print("=" * 40)
        
        # Ensemble prediction (average of probabilities)
        probs = np.zeros(len(y_val))
        for name, model in self.models.items():
            prob = model.predict_proba(X_val)[:, 1]
            probs += prob
            
            auc = roc_auc_score(y_val, prob)
            print(f"{name.upper()} AUC: {auc:.4f}")
        
        probs /= len(self.models)
        ensemble_preds = (probs > 0.5).astype(int)
        
        ensemble_auc = roc_auc_score(y_val, probs)
        print(f"\nENSEMBLE AUC: {ensemble_auc:.4f}")
        print("\nClassification Report (Ensemble):")
        print(classification_report(y_val, ensemble_preds, target_names=['Legitimate', 'Fraud']))
        
        self.is_fitted = True
        
        return {
            'ensemble_auc': ensemble_auc,
            'rf_auc': roc_auc_score(y_val, self.models['rf'].predict_proba(X_val)[:, 1]),
            'gb_auc': roc_auc_score(y_val, self.models['gb'].predict_proba(X_val)[:, 1]),
            'lr_auc': roc_auc_score(y_val, self.models['lr'].predict_proba(X_val)[:, 1])
        }
    
    def predict_risk(self, features: np.ndarray) -> Dict:
        """
        Predict fraud risk for a transaction.
        
        Args:
            features: Transaction features (1D or 2D array)
            
        Returns:
            Dictionary with risk score and details
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features = np.atleast_2d(features)
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call fit() first.")
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from each model
        predictions = {}
        probs_sum = np.zeros(len(features))
        
        for name, model in self.models.items():
            prob = model.predict_proba(features_scaled)[:, 1]
            predictions[name] = prob
            probs_sum += prob
        
        # Ensemble risk score
        risk_score = probs_sum / len(self.models)
        
        # Determine risk level
        risk_levels = []
        for score in risk_score:
            if score < 0.2:
                risk_levels.append('LOW')
            elif score < 0.5:
                risk_levels.append('MEDIUM')
            elif score < 0.8:
                risk_levels.append('HIGH')
            else:
                risk_levels.append('CRITICAL')
        
        return {
            'risk_score': risk_score.tolist(),
            'risk_level': risk_levels,
            'is_fraud': (risk_score > 0.5).tolist(),
            'model_scores': {k: v.tolist() for k, v in predictions.items()}
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest model."""
        if 'rf' not in self.models or self.feature_names is None:
            return {}
        
        importances = self.models['rf'].feature_importances_
        
        return dict(zip(self.feature_names, importances))
    
    def save(self, path: str):
        """Save models and scaler."""
        import pickle
        
        save_dict = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✅ Saved model to: {path}")
    
    def load(self, path: str):
        """Load models and scaler."""
        import pickle
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.models = save_dict['models']
        self.scaler = save_dict['scaler']
        self.feature_names = save_dict['feature_names']
        self.is_fitted = save_dict['is_fitted']
        
        print(f"✅ Loaded model from: {path}")


class MultiModalFraudDetector:
    """
    Combines image-based fraud detection with transaction risk scoring
    for comprehensive fraud detection.
    """
    
    def __init__(self, image_model_path: Optional[str] = None, transaction_model_path: Optional[str] = None):
        self.image_model = None
        self.transaction_scorer = None
        
        # Weights for combining scores
        self.image_weight = 0.6
        self.transaction_weight = 0.4
        
        # Load models if paths provided
        if image_model_path:
            self._load_image_model(image_model_path)
        if transaction_model_path:
            self.transaction_scorer = TransactionRiskScorer(transaction_model_path)
    
    def _load_image_model(self, path: str):
        """Load image classification model."""
        from models.hybrid_model import CNNBaseline
        
        checkpoint = torch.load(path, map_location='cpu')
        self.image_model = CNNBaseline(num_classes=4)
        self.image_model.load_state_dict(checkpoint['model_state_dict'])
        self.image_model.eval()
    
    def predict(self, image=None, transaction_features=None) -> Dict:
        """
        Predict fraud risk using available inputs.
        
        Args:
            image: PIL Image or tensor (optional)
            transaction_features: Transaction feature array (optional)
            
        Returns:
            Combined prediction dictionary
        """
        results: Dict[str, Any] = {
            'image_prediction': None,
            'transaction_risk': None,
            'combined_score': None,
            'final_verdict': None
        }
        
        scores = []
        weights = []
        
        # Image-based prediction
        if image is not None and self.image_model is not None:
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            if hasattr(image, 'mode'):  # PIL Image
                image_tensor = transform(image).unsqueeze(0)
            else:
                image_tensor = image
            
            with torch.no_grad():
                outputs = self.image_model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = outputs.argmax(dim=1).item()
            
            class_names = ['genuine', 'fraud', 'tampered', 'forged']
            fraud_prob = 1 - probs[0, 0].item()  # 1 - P(genuine)
            
            results['image_prediction'] = {
                'class': class_names[pred_class],
                'confidence': probs[0, pred_class].item(),
                'fraud_probability': fraud_prob,
                'all_probs': {name: probs[0, i].item() for i, name in enumerate(class_names)}
            }
            
            scores.append(fraud_prob)
            weights.append(self.image_weight)
        
        # Transaction-based risk
        if transaction_features is not None and self.transaction_scorer is not None:
            risk_result = self.transaction_scorer.predict_risk(transaction_features)
            
            results['transaction_risk'] = {
                'risk_score': risk_result['risk_score'][0],
                'risk_level': risk_result['risk_level'][0],
                'is_fraud': risk_result['is_fraud'][0]
            }
            
            scores.append(risk_result['risk_score'][0])
            weights.append(self.transaction_weight)
        
        # Combined score
        if scores:
            total_weight = sum(weights)
            results['combined_score'] = sum(s * w for s, w in zip(scores, weights)) / total_weight
            
            # Final verdict
            if results['combined_score'] < 0.3:
                results['final_verdict'] = 'LIKELY LEGITIMATE'
            elif results['combined_score'] < 0.5:
                results['final_verdict'] = 'REVIEW RECOMMENDED'
            elif results['combined_score'] < 0.7:
                results['final_verdict'] = 'SUSPICIOUS'
            else:
                results['final_verdict'] = 'LIKELY FRAUD'
        
        return results


def create_sample_transaction_data(output_path: str, n_samples: int = 10000):
    """
    DEPRECATED: This project now uses real Kaggle data only.
    See setup_datasets.py --creditcard for download instructions.
    """
    print("⚠️  Synthetic sample generation has been removed.")
    print("    Download the real Kaggle credit-card fraud dataset instead:")
    print("      python setup_datasets.py --creditcard")
    print("    Place creditcard.csv at data/transactions/creditcard.csv")
    return None, None


def train_risk_model(data_path: str, output_path: str = 'checkpoints/risk_model.pkl'):
    """
    Train transaction risk scoring model.
    
    Args:
        data_path: Path to transaction CSV
        output_path: Path to save trained model
    """
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        os.system(f"{sys.executable} -m pip install pandas")
        import pandas as pd
    
    print(f"Loading data from: {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        data = np.load(data_path)
        X, y = data['X'], data['y']
        feature_names = [f'V{i}' for i in range(1, X.shape[1] + 1)]
        
        scorer = TransactionRiskScorer()
        scorer.fit(X, y, feature_names)
        scorer.save(output_path)
        return scorer
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col.startswith('V') or col in ['Amount', 'Time']]
    X = df[feature_cols].values
    y = df['Class'].values.astype(int)
    
    # Train model
    scorer = TransactionRiskScorer()
    results = scorer.fit(X, y, feature_names=feature_cols)
    
    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scorer.save(output_path)
    
    # Print feature importance
    print("\nTop 10 Important Features:")
    importance = scorer.get_feature_importance()
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, imp in sorted_imp:
        print(f"  {name}: {imp:.4f}")
    
    return scorer


def download_info():
    """Print download instructions for Kaggle dataset."""
    print("=" * 60)
    print("Kaggle Credit Card Fraud Dataset Download Instructions")
    print("=" * 60)
    
    print("""
Dataset: Credit Card Fraud Detection
URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download Options:

1. Using Kaggle CLI:
   pip install kaggle
   kaggle datasets download mlg-ulb/creditcardfraud
   unzip creditcardfraud.zip -d data/transactions/

2. Manual Download:
   - Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Click "Download" button
   - Extract to data/transactions/creditcard.csv

Dataset Details:
   - 284,807 transactions (2 days)
   - 492 frauds (0.172%)
   - Features: Time, V1-V28 (PCA), Amount, Class
   - File size: ~144 MB

For testing without the real dataset, use:
   python data_integration/creditcard_fraud_loader.py --create-sample
""")


def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Dataset Integration')
    parser.add_argument('--download-info', action='store_true', help='Show download instructions')
    parser.add_argument('--train', action='store_true', help='Train risk scoring model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on test data')
    parser.add_argument('--data-path', type=str, default='data/transactions/creditcard.csv', help='Data path')
    parser.add_argument('--model-path', type=str, default='checkpoints/risk_model.pkl', help='Model path')
    
    args = parser.parse_args()
    
    if args.download_info:
        download_info()
    elif args.train:
        if not Path(args.data_path).exists():
            print(f"❌ Data not found: {args.data_path}")
            print("   Download from Kaggle first:")
            download_info()
            return
        train_risk_model(args.data_path, args.model_path)
    elif args.evaluate:
        if not Path(args.model_path).exists():
            print(f"Model not found: {args.model_path}")
            print("Train a model first with --train")
            return
        
        # Load model and run quick test
        scorer = TransactionRiskScorer(args.model_path)
        
        # Generate test transaction
        test_features = np.random.randn(1, 28)
        result = scorer.predict_risk(test_features)
        
        print("\nTest Prediction:")
        print(f"  Risk Score: {result['risk_score'][0]:.4f}")
        print(f"  Risk Level: {result['risk_level'][0]}")
        print(f"  Is Fraud: {result['is_fraud'][0]}")
    else:
        download_info()


if __name__ == '__main__':
    main()
