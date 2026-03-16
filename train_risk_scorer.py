"""
Transaction Risk Scorer Training
================================
Trains a machine learning model to score transaction fraud risk.

Uses the credit card fraud dataset (50K transactions).
Supports multiple algorithms: XGBoost, RandomForest, LightGBM.

Usage:
    python train_risk_scorer.py
    python train_risk_scorer.py --model xgboost
    python train_risk_scorer.py --model ensemble
"""

import sys
import os
import json
import time
import warnings
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    _reconfigure = 'reconfigure'
    if hasattr(sys.stdout, _reconfigure):
        getattr(sys.stdout, _reconfigure)(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Optional: XGBoost and LightGBM
try:
    import xgboost as xgb  # pyright: ignore[reportMissingImports]
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb  # pyright: ignore[reportMissingImports]
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# Configuration
DATA_PATH = Path("data/transactions/creditcard.csv")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42


def load_data():
    """Load and preprocess credit card data."""
    print("Loading data...")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset: {len(df):,} transactions")
    print(f"  Features: {df.shape[1] - 1}")
    print(f"  Fraud rate: {df['Class'].mean() * 100:.2f}%")
    
    # Features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y


def create_features(X):
    """Create additional features for better fraud detection."""
    X_eng = X.copy()
    
    # Log transform Amount (handle zeros)
    if 'Amount' in X.columns:
        X_eng['Amount_Log'] = np.log1p(X['Amount'])
        X_eng['Amount_Sqrt'] = np.sqrt(X['Amount'])
    
    # V-feature interactions (top suspicious combinations)
    if 'V1' in X.columns and 'V2' in X.columns:
        X_eng['V1_V2'] = X['V1'] * X['V2']
        X_eng['V1_V3'] = X['V1'] * X['V3']
        X_eng['V3_V4'] = X['V3'] * X['V4']
    
    # V-feature statistics
    v_cols = [c for c in X.columns if c.startswith('V')]
    if v_cols:
        X_eng['V_mean'] = X[v_cols].mean(axis=1)
        X_eng['V_std'] = X[v_cols].std(axis=1)
        X_eng['V_max'] = X[v_cols].max(axis=1)
        X_eng['V_min'] = X[v_cols].min(axis=1)
    
    return X_eng


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier."""
    print("\n[1/3] Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
    auc = roc_auc_score(y_val, y_prob)
    
    print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f} | F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f} | Time:      {train_time:.1f}s")
    
    return model, {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting classifier."""
    print("\n[2/3] Training Gradient Boosting...")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
    auc = roc_auc_score(y_val, y_prob)
    
    print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f} | F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f} | Time:      {train_time:.1f}s")
    
    return model, {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier (if available)."""
    if not HAS_XGB:
        print("\n[3/3] XGBoost not installed, skipping...")
        return None, None
    
    print("\n[3/3] Training XGBoost...")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
    auc = roc_auc_score(y_val, y_prob)
    
    print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f} | F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f} | Time:      {train_time:.1f}s")
    
    return model, {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}


def train_risk_scorer():
    """Main training function."""
    
    print("=" * 60)
    print("  TRANSACTION RISK SCORER TRAINING")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    
    # Feature engineering
    print("\nFeature engineering...")
    X = create_features(X)
    print(f"  Total features: {X.shape[1]}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train):,} ({y_train.mean()*100:.2f}% fraud)")
    print(f"  Val:   {len(X_val):,} ({y_val.mean()*100:.2f}% fraud)")
    print(f"  Test:  {len(X_test):,} ({y_test.mean()*100:.2f}% fraud)")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)
    
    models = {}
    metrics = {}
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
    models['RandomForest'] = rf_model
    metrics['RandomForest'] = rf_metrics
    
    # Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting(X_train_scaled, y_train, X_val_scaled, y_val)
    models['GradientBoosting'] = gb_model
    metrics['GradientBoosting'] = gb_metrics
    
    # XGBoost (if available)
    xgb_model, xgb_metrics = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
    if xgb_model:
        models['XGBoost'] = xgb_model
        metrics['XGBoost'] = xgb_metrics
    
    # Find best model
    best_name = max(metrics, key=lambda k: metrics[k]['auc'])
    best_model = models[best_name]
    
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)
    print(f"\nBest model: {best_name}")
    
    # Evaluate best model on test set
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest Results ({best_name}):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]:,}")
    print(f"  False Positives: {cm[0][1]:,}")
    print(f"  False Negatives: {cm[1][0]:,}")
    print(f"  True Positives:  {cm[1][1]:,}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        indices = np.argsort(importance)[::-1][:10]
        print(f"\nTop 10 Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {X.columns[idx]}: {importance[idx]:.4f}")
    
    # Save model and scaler
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    joblib.dump(best_model, CHECKPOINT_DIR / "risk_scorer_best.joblib")
    joblib.dump(scaler, CHECKPOINT_DIR / "risk_scorer_scaler.joblib")
    
    # Save all results
    results = {
        "best_model": best_name,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_auc": auc,
        "confusion_matrix": cm.tolist(),
        "model_metrics": {k: v for k, v in metrics.items()},
        "num_features": X.shape[1],
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    with open(RESULTS_DIR / "risk_scorer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to: {CHECKPOINT_DIR / 'risk_scorer_best.joblib'}")
    print(f"Results saved to: {RESULTS_DIR / 'risk_scorer_results.json'}")
    
    print("\n" + "=" * 60)
    print(f"  {best_name} - AUC: {auc:.4f} | F1: {f1:.4f}")
    print("=" * 60)
    
    return results


class RiskScorer:
    """Risk scorer for inference."""
    
    def __init__(self, model_path=None, scaler_path=None):
        model_path = model_path or CHECKPOINT_DIR / "risk_scorer_best.joblib"
        scaler_path = scaler_path or CHECKPOINT_DIR / "risk_scorer_scaler.joblib"
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict(self, X):
        """Predict fraud probability."""
        X = create_features(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def score(self, transaction):
        """Score a single transaction."""
        if isinstance(transaction, dict):
            transaction = pd.DataFrame([transaction])
        return self.predict(transaction)[0]


if __name__ == "__main__":
    train_risk_scorer()
