"""
MLOps Monitoring Module
========================
Production-grade monitoring for fraud detection models.

Features:
1. Prediction Logging
2. Confidence Monitoring
3. Data Drift Detection
4. Model Performance Tracking
5. Alert Generation
6. Misclassification Analysis

Usage:
    python mlops_monitoring.py --demo
    python mlops_monitoring.py --dashboard
    python mlops_monitoring.py --check-drift
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import hashlib

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))


@dataclass
class PredictionLog:
    """Single prediction log entry."""
    timestamp: str
    model_name: str
    input_hash: str
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    inference_time_ms: float
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    flagged: bool = False
    

@dataclass
class PerformanceMetrics:
    """Performance metrics over a time window."""
    window_start: str
    window_end: str
    total_predictions: int
    accuracy: float
    avg_confidence: float
    low_confidence_count: int  # predictions below threshold
    class_distribution: Dict[str, int]
    avg_inference_time_ms: float
    drift_detected: bool = False
    alerts: List[str] = field(default_factory=list)


class PredictionLogger:
    """Logs all predictions for monitoring."""
    
    def __init__(self, log_dir: str = 'logs/predictions', 
                 buffer_size: int = 10000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer = deque(maxlen=buffer_size)
        self.current_file = None
        self.daily_logs = {}
        
        # Load existing logs
        self._load_recent_logs()
    
    def _load_recent_logs(self):
        """Load recent log files."""
        log_files = sorted(self.log_dir.glob('predictions_*.jsonl'))
        for log_file in log_files[-7:]:  # Last 7 days
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        self.buffer.append(PredictionLog(**entry))
                    except:
                        pass
    
    def _get_log_file(self) -> Path:
        """Get current day's log file."""
        date_str = datetime.now().strftime('%Y%m%d')
        return self.log_dir / f'predictions_{date_str}.jsonl'
    
    def _compute_input_hash(self, input_data) -> str:
        """Compute hash of input for deduplication."""
        if isinstance(input_data, torch.Tensor):
            data_bytes = input_data.numpy().tobytes()
        elif isinstance(input_data, np.ndarray):
            data_bytes = input_data.tobytes()
        else:
            data_bytes = str(input_data).encode()
        
        return hashlib.md5(data_bytes).hexdigest()[:12]
    
    def log(self, model_name: str, input_data, predicted_class: str,
            confidence: float, class_probabilities: Dict[str, float],
            inference_time_ms: float, ground_truth: Optional[str] = None) -> PredictionLog:
        """Log a prediction."""
        entry = PredictionLog(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            input_hash=self._compute_input_hash(input_data),
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            inference_time_ms=inference_time_ms,
            ground_truth=ground_truth,
            is_correct=predicted_class == ground_truth if ground_truth else None,
            flagged=confidence < 0.6  # Flag low confidence predictions
        )
        
        self.buffer.append(entry)
        
        # Write to file
        log_file = self._get_log_file()
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry.__dict__) + '\n')
        
        return entry
    
    def get_recent_logs(self, n: int = 100) -> List[PredictionLog]:
        """Get n most recent logs."""
        return list(self.buffer)[-n:]
    
    def get_logs_in_window(self, hours: int = 24) -> List[PredictionLog]:
        """Get logs from last n hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()
        
        return [log for log in self.buffer if log.timestamp >= cutoff_str]


class ConfidenceMonitor:
    """Monitors prediction confidence levels."""
    
    def __init__(self, low_threshold: float = 0.6, 
                 critical_threshold: float = 0.4,
                 alert_rate_threshold: float = 0.1):
        self.low_threshold = low_threshold
        self.critical_threshold = critical_threshold
        self.alert_rate_threshold = alert_rate_threshold
        
        self.confidence_history = deque(maxlen=1000)
        self.alerts = []
    
    def update(self, confidence: float, predicted_class: str):
        """Update with new prediction."""
        self.confidence_history.append({
            'confidence': confidence,
            'class': predicted_class,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check for alerts
        self._check_alerts()
    
    def _check_alerts(self):
        """Check if alerts should be triggered."""
        if len(self.confidence_history) < 10:
            return
        
        recent = list(self.confidence_history)[-100:]
        
        # Calculate low confidence rate
        low_conf_count = sum(1 for c in recent if c['confidence'] < self.low_threshold)
        low_conf_rate = low_conf_count / len(recent)
        
        if low_conf_rate > self.alert_rate_threshold:
            self.alerts.append({
                'type': 'HIGH_LOW_CONFIDENCE_RATE',
                'message': f'Low confidence rate ({low_conf_rate:.1%}) exceeds threshold',
                'timestamp': datetime.now().isoformat(),
                'severity': 'WARNING'
            })
        
        # Check for sudden confidence drop
        if len(recent) >= 20:
            old_avg = np.mean([c['confidence'] for c in recent[:10]])
            new_avg = np.mean([c['confidence'] for c in recent[-10:]])
            
            if new_avg < old_avg - 0.15:
                self.alerts.append({
                    'type': 'CONFIDENCE_DROP',
                    'message': f'Sudden confidence drop detected ({old_avg:.2f} → {new_avg:.2f})',
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'CRITICAL'
                })
    
    def get_statistics(self) -> Dict:
        """Get confidence statistics."""
        if not self.confidence_history:
            return {'error': 'No data'}
        
        confidences = [c['confidence'] for c in self.confidence_history]
        
        return {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'low_confidence_rate': sum(1 for c in confidences if c < self.low_threshold) / len(confidences),
            'critical_rate': sum(1 for c in confidences if c < self.critical_threshold) / len(confidences),
            'total_predictions': len(confidences)
        }


class DriftDetector:
    """Detects data and concept drift."""
    
    def __init__(self, reference_window: int = 500, 
                 test_window: int = 100,
                 threshold: float = 0.1):
        self.reference_window = reference_window
        self.test_window = test_window
        self.threshold = threshold
        
        self.class_distribution_history = deque(maxlen=reference_window)
        self.drift_alerts = []
    
    def update(self, predicted_class: str, class_probabilities: Dict[str, float]):
        """Update with new prediction."""
        self.class_distribution_history.append({
            'class': predicted_class,
            'probs': class_probabilities,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check for drift periodically
        if len(self.class_distribution_history) >= self.reference_window:
            self._check_drift()
    
    def _check_drift(self):
        """Check for distribution drift using PSI."""
        history = list(self.class_distribution_history)
        
        if len(history) < self.reference_window:
            return
        
        # Split into reference and test
        reference = history[:self.reference_window - self.test_window]
        test = history[-self.test_window:]
        
        # Calculate class distributions
        ref_dist = self._calculate_distribution(reference)
        test_dist = self._calculate_distribution(test)
        
        # Calculate Population Stability Index (PSI)
        psi = self._calculate_psi(ref_dist, test_dist)
        
        if psi > self.threshold:
            self.drift_alerts.append({
                'type': 'DISTRIBUTION_DRIFT',
                'psi': psi,
                'reference_dist': ref_dist,
                'test_dist': test_dist,
                'timestamp': datetime.now().isoformat(),
                'severity': 'CRITICAL' if psi > 0.25 else 'WARNING'
            })
    
    def _calculate_distribution(self, predictions: List[Dict]) -> Dict[str, float]:
        """Calculate class distribution."""
        classes = [p['class'] for p in predictions]
        unique_classes = set(classes)
        total = len(classes)
        
        return {c: classes.count(c) / total for c in unique_classes}
    
    def _calculate_psi(self, reference: Dict[str, float], 
                       test: Dict[str, float]) -> float:
        """Calculate Population Stability Index."""
        psi = 0
        all_classes = set(reference.keys()) | set(test.keys())
        
        for cls in all_classes:
            ref_pct = reference.get(cls, 0.0001)
            test_pct = test.get(cls, 0.0001)
            
            ref_pct = max(ref_pct, 0.0001)
            test_pct = max(test_pct, 0.0001)
            
            psi += (test_pct - ref_pct) * np.log(test_pct / ref_pct)
        
        return abs(psi)
    
    def get_drift_status(self) -> Dict:
        """Get current drift status."""
        return {
            'drift_detected': len(self.drift_alerts) > 0,
            'recent_alerts': self.drift_alerts[-5:] if self.drift_alerts else [],
            'samples_collected': len(self.class_distribution_history)
        }


class MisclassificationTracker:
    """Tracks misclassifications for analysis."""
    
    def __init__(self, max_tracked: int = 500):
        self.misclassifications = deque(maxlen=max_tracked)
        self.confusion_data = {}
    
    def track(self, predicted_class: str, ground_truth: str,
              confidence: float, input_hash: str):
        """Track a misclassification."""
        if predicted_class == ground_truth:
            return
        
        self.misclassifications.append({
            'predicted': predicted_class,
            'actual': ground_truth,
            'confidence': confidence,
            'input_hash': input_hash,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update confusion data
        key = f"{ground_truth}→{predicted_class}"
        self.confusion_data[key] = self.confusion_data.get(key, 0) + 1
    
    def get_analysis(self) -> Dict:
        """Get misclassification analysis."""
        if not self.misclassifications:
            return {'error': 'No misclassifications tracked'}
        
        misclass_list = list(self.misclassifications)
        
        # Most common errors
        sorted_errors = sorted(
            self.confusion_data.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Confidence of misclassifications
        confidences = [m['confidence'] for m in misclass_list]
        
        return {
            'total_misclassifications': len(misclass_list),
            'most_common_errors': sorted_errors[:5],
            'avg_misclass_confidence': np.mean(confidences),
            'high_confidence_errors': sum(1 for c in confidences if c > 0.8),
            'recent_errors': misclass_list[-10:]
        }


class ModelMonitor:
    """
    Main MLOps monitoring interface.
    Combines all monitoring capabilities.
    """
    
    def __init__(self, model_name: str = 'fraud_detector',
                 log_dir: str = 'logs'):
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.prediction_logger = PredictionLogger(str(self.log_dir / 'predictions'))
        self.confidence_monitor = ConfidenceMonitor()
        self.drift_detector = DriftDetector()
        self.misclass_tracker = MisclassificationTracker()
        
        # Performance history
        self.hourly_metrics = deque(maxlen=168)  # 7 days
        
        print(f"✅ ModelMonitor initialized for '{model_name}'")
    
    def log_prediction(self, input_data, predicted_class: str,
                       confidence: float, class_probabilities: Dict[str, float],
                       inference_time_ms: float, ground_truth: Optional[str] = None):
        """Log a prediction and update all monitors."""
        # Log prediction
        log_entry = self.prediction_logger.log(
            model_name=self.model_name,
            input_data=input_data,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            inference_time_ms=inference_time_ms,
            ground_truth=ground_truth
        )
        
        # Update confidence monitor
        self.confidence_monitor.update(confidence, predicted_class)
        
        # Update drift detector
        self.drift_detector.update(predicted_class, class_probabilities)
        
        # Track misclassifications
        if ground_truth and predicted_class != ground_truth:
            self.misclass_tracker.track(
                predicted_class, ground_truth, confidence, log_entry.input_hash
            )
        
        return log_entry
    
    def get_health_status(self) -> Dict:
        """Get overall model health status."""
        conf_stats = self.confidence_monitor.get_statistics()
        drift_status = self.drift_detector.get_drift_status()
        
        # Determine overall health
        issues = []
        
        if conf_stats.get('low_confidence_rate', 0) > 0.2:
            issues.append('High rate of low-confidence predictions')
        
        if conf_stats.get('critical_rate', 0) > 0.1:
            issues.append('Critical confidence rate too high')
        
        if drift_status.get('drift_detected'):
            issues.append('Data drift detected')
        
        if len(self.confidence_monitor.alerts) > 0:
            issues.append(f'{len(self.confidence_monitor.alerts)} confidence alerts')
        
        health = 'HEALTHY' if not issues else ('DEGRADED' if len(issues) < 2 else 'CRITICAL')
        
        return {
            'status': health,
            'issues': issues,
            'confidence_stats': conf_stats,
            'drift_status': drift_status,
            'alerts': self._get_all_alerts(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_all_alerts(self) -> List[Dict]:
        """Get all active alerts."""
        alerts = []
        alerts.extend(self.confidence_monitor.alerts[-10:])
        alerts.extend(self.drift_detector.drift_alerts[-10:])
        return sorted(alerts, key=lambda x: x.get('timestamp', ''), reverse=True)[:15]
    
    def generate_report(self, hours: int = 24) -> str:
        """Generate monitoring report."""
        logs = self.prediction_logger.get_logs_in_window(hours)
        health = self.get_health_status()
        misclass = self.misclass_tracker.get_analysis()
        
        report = f"""
================================================================================
                    MODEL MONITORING REPORT
================================================================================
Model: {self.model_name}
Report Period: Last {hours} hours
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

HEALTH STATUS: {health['status']}
{'─' * 60}
"""
        
        if health['issues']:
            report += "Issues Detected:\n"
            for issue in health['issues']:
                report += f"  ⚠️ {issue}\n"
        else:
            report += "  ✅ No issues detected\n"
        
        report += f"""
PREDICTION STATISTICS
{'─' * 60}
Total Predictions: {len(logs)}
"""
        
        if logs:
            confidences = [log.confidence for log in logs]
            report += f"""Average Confidence: {np.mean(confidences):.2%}
Min Confidence: {np.min(confidences):.2%}
Max Confidence: {np.max(confidences):.2%}

Class Distribution:
"""
            class_counts = {}
            for log in logs:
                class_counts[log.predicted_class] = class_counts.get(log.predicted_class, 0) + 1
            
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                pct = count / len(logs) * 100
                report += f"  {cls}: {count} ({pct:.1f}%)\n"
        
        # Misclassification analysis
        if 'total_misclassifications' in misclass:
            report += f"""
MISCLASSIFICATION ANALYSIS
{'─' * 60}
Total Misclassifications: {misclass['total_misclassifications']}
Avg Misclass Confidence: {misclass.get('avg_misclass_confidence', 0):.2%}
High-Confidence Errors: {misclass.get('high_confidence_errors', 0)}

Most Common Error Patterns:
"""
            for pattern, count in misclass.get('most_common_errors', []):
                report += f"  {pattern}: {count} occurrences\n"
        
        # Alerts
        alerts = health.get('alerts', [])
        if alerts:
            report += f"""
ALERTS ({len(alerts)})
{'─' * 60}
"""
            for alert in alerts[:5]:
                report += f"  [{alert.get('severity', 'INFO')}] {alert.get('message', 'Unknown')}\n"
                report += f"    Time: {alert.get('timestamp', 'Unknown')}\n"
        
        report += f"""
================================================================================
                           END OF REPORT
================================================================================
"""
        return report
    
    def save_metrics(self, output_path: Optional[str] = None):
        """Save current metrics to file."""
        resolved_path: Path = (
            Path(output_path) if output_path is not None
            else self.log_dir / 'metrics' / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'health': self.get_health_status(),
            'misclassifications': self.misclass_tracker.get_analysis()
        }
        
        with open(resolved_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"✅ Metrics saved to: {resolved_path}")


def demo():
    """Demonstrate MLOps monitoring capabilities."""
    print("=" * 70)
    print("MLOPS MONITORING DEMO")
    print("=" * 70)
    print()
    
    # Initialize monitor
    monitor = ModelMonitor(model_name='fraud_detector_cnn')
    
    # Simulate predictions
    print("Simulating 100 predictions...")
    
    classes = ['genuine', 'fraud', 'tampered', 'forged']
    
    for i in range(100):
        # Simulate prediction
        true_class = np.random.choice(classes, p=[0.7, 0.1, 0.1, 0.1])
        
        # Simulate model output
        if np.random.random() > 0.15:  # 85% correct
            pred_class = true_class
            confidence = np.random.uniform(0.75, 0.99)
        else:
            pred_class = np.random.choice([c for c in classes if c != true_class])
            confidence = np.random.uniform(0.4, 0.8)
        
        # Generate class probabilities
        probs = {c: 0.05 for c in classes}
        probs[pred_class] = confidence
        remaining = 1 - confidence
        for c in classes:
            if c != pred_class:
                probs[c] = remaining / 3
        
        # Log prediction
        input_data = np.random.randn(3, 224, 224)
        monitor.log_prediction(
            input_data=input_data,
            predicted_class=pred_class,
            confidence=confidence,
            class_probabilities=probs,
            inference_time_ms=np.random.uniform(5, 30),
            ground_truth=true_class
        )
    
    print("✅ Simulated 100 predictions\n")
    
    # Get health status
    print("=" * 60)
    print("HEALTH STATUS")
    print("=" * 60)
    
    health = monitor.get_health_status()
    print(f"Status: {health['status']}")
    print(f"Issues: {health['issues'] if health['issues'] else 'None'}")
    
    # Generate report
    print("\n" + "=" * 60)
    print("MONITORING REPORT")
    print("=" * 60)
    
    report = monitor.generate_report(hours=1)
    print(report)
    
    # Save metrics
    monitor.save_metrics()


def main():
    parser = argparse.ArgumentParser(description='MLOps Monitoring')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--report', type=int, default=24, help='Generate report for last N hours')
    parser.add_argument('--check-health', action='store_true', help='Check model health')
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.check_health:
        monitor = ModelMonitor()
        health = monitor.get_health_status()
        print(json.dumps(health, indent=2, default=str))
    else:
        demo()


if __name__ == '__main__':
    main()
