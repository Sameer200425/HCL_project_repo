"""
MLOps Monitoring Routes.
Provides endpoints for model health, metrics, alerts, and reporting.
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime, timezone

import sys
sys.path.insert(0, '.')

from mlops_monitoring import ModelMonitor

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# Global model monitor instance
_model_monitor: Optional[ModelMonitor] = None

_runtime_metrics: Dict[str, Any] = {
    "request_count": 0,
    "error_count": 0,
    "avg_latency_ms": 0.0,
    "max_latency_ms": 0.0,
    "last_updated": None,
}


def record_request_metric(latency_ms: float, status_code: int) -> None:
    """Update runtime API metrics from request middleware."""
    count = int(_runtime_metrics["request_count"]) + 1
    prev_avg = float(_runtime_metrics["avg_latency_ms"])
    _runtime_metrics["request_count"] = count
    _runtime_metrics["avg_latency_ms"] = ((prev_avg * (count - 1)) + latency_ms) / count
    _runtime_metrics["max_latency_ms"] = max(float(_runtime_metrics["max_latency_ms"]), latency_ms)
    if status_code >= 500:
        _runtime_metrics["error_count"] = int(_runtime_metrics["error_count"]) + 1
    _runtime_metrics["last_updated"] = datetime.now(timezone.utc).isoformat()


def get_runtime_metrics() -> Dict[str, Any]:
    """Expose a snapshot of runtime API metrics for health dashboards."""
    request_count = int(_runtime_metrics["request_count"])
    error_count = int(_runtime_metrics["error_count"])
    error_rate = (error_count / request_count) if request_count else 0.0
    return {
        "request_count": request_count,
        "error_count": error_count,
        "error_rate": round(error_rate, 4),
        "avg_latency_ms": round(float(_runtime_metrics["avg_latency_ms"]), 2),
        "max_latency_ms": round(float(_runtime_metrics["max_latency_ms"]), 2),
        "last_updated": _runtime_metrics["last_updated"],
    }


def get_model_monitor() -> ModelMonitor:
    """Get or create the model monitor instance."""
    global _model_monitor
    if _model_monitor is None:
        _model_monitor = ModelMonitor(model_name='fraud_detector', log_dir='logs')
    return _model_monitor


# Response models
class HealthResponse(BaseModel):
    status: str
    issues: List[str]
    confidence_stats: Dict[str, Any]
    drift_status: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    timestamp: str


class MetricsResponse(BaseModel):
    total_predictions: int
    mean_confidence: Optional[float]
    std_confidence: Optional[float]
    min_confidence: Optional[float]
    max_confidence: Optional[float]
    low_confidence_rate: Optional[float]
    critical_rate: Optional[float]


class AlertsResponse(BaseModel):
    alerts: List[Dict[str, Any]]
    total_count: int


class PredictionLogResponse(BaseModel):
    logs: List[Dict[str, Any]]
    total_count: int


class ReportResponse(BaseModel):
    report: str
    generated_at: str


@router.get("/health", response_model=HealthResponse)
async def get_health_status():
    """
    Get overall model health status.
    
    Returns health status including:
    - Overall status (HEALTHY, DEGRADED, CRITICAL)
    - Active issues
    - Confidence statistics
    - Drift detection status
    - Recent alerts
    """
    monitor = get_model_monitor()
    health = monitor.get_health_status()
    return HealthResponse(**health)


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get model performance metrics.
    
    Returns confidence statistics and prediction counts.
    """
    monitor = get_model_monitor()
    stats = monitor.confidence_monitor.get_statistics()
    
    if 'error' in stats:
        return MetricsResponse(
            total_predictions=0,
            mean_confidence=None,
            std_confidence=None,
            min_confidence=None,
            max_confidence=None,
            low_confidence_rate=None,
            critical_rate=None
        )
    
    return MetricsResponse(
        total_predictions=stats.get('total_predictions', 0),
        mean_confidence=stats.get('mean'),
        std_confidence=stats.get('std'),
        min_confidence=stats.get('min'),
        max_confidence=stats.get('max'),
        low_confidence_rate=stats.get('low_confidence_rate'),
        critical_rate=stats.get('critical_rate')
    )


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts():
    """
    Get all active alerts.
    
    Returns confidence alerts and drift alerts.
    """
    monitor = get_model_monitor()
    alerts = monitor._get_all_alerts()
    
    return AlertsResponse(
        alerts=alerts,
        total_count=len(alerts)
    )


@router.get("/logs")
async def get_prediction_logs(
    hours: int = Query(default=24, ge=1, le=168, description="Hours of logs to retrieve"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of logs")
) -> PredictionLogResponse:
    """
    Get recent prediction logs.
    
    Args:
        hours: Number of hours of logs to retrieve (1-168)
        limit: Maximum number of logs to return (1-1000)
    """
    monitor = get_model_monitor()
    logs = monitor.prediction_logger.get_logs_in_window(hours)
    
    # Convert to dicts and limit
    log_dicts = []
    for log in logs[-limit:]:
        log_dicts.append({
            'timestamp': log.timestamp,
            'model_name': log.model_name,
            'predicted_class': log.predicted_class,
            'confidence': log.confidence,
            'class_probabilities': log.class_probabilities,
            'inference_time_ms': log.inference_time_ms,
            'ground_truth': log.ground_truth,
            'is_correct': log.is_correct,
            'flagged': log.flagged
        })
    
    return PredictionLogResponse(
        logs=log_dicts,
        total_count=len(logs)
    )


@router.get("/report")
async def generate_report(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to include in report")
) -> ReportResponse:
    """
    Generate a monitoring report.
    
    Args:
        hours: Number of hours to include in the report
    """
    from datetime import datetime
    
    monitor = get_model_monitor()
    report = monitor.generate_report(hours)
    
    return ReportResponse(
        report=report,
        generated_at=datetime.now().isoformat()
    )


@router.get("/drift")
async def get_drift_status() -> Dict[str, Any]:
    """
    Get data drift detection status.
    
    Returns:
    - Whether drift has been detected
    - Recent drift alerts
    - Number of samples collected
    """
    monitor = get_model_monitor()
    return monitor.drift_detector.get_drift_status()


@router.get("/misclassifications")
async def get_misclassification_analysis() -> Dict[str, Any]:
    """
    Get misclassification analysis.
    
    Returns analysis of incorrect predictions including:
    - Total misclassifications tracked
    - Most common error patterns
    - Average confidence of misclassified predictions
    """
    monitor = get_model_monitor()
    return monitor.misclass_tracker.get_analysis()


@router.post("/clear-alerts")
async def clear_alerts() -> Dict[str, str]:
    """
    Clear all alerts.
    
    Use this after acknowledging alerts.
    """
    monitor = get_model_monitor()
    monitor.confidence_monitor.alerts.clear()
    monitor.drift_detector.drift_alerts.clear()
    
    return {"message": "All alerts cleared"}


@router.post("/save-metrics")
async def save_metrics() -> Dict[str, str]:
    """
    Save current metrics to a file.
    """
    monitor = get_model_monitor()
    monitor.save_metrics()
    
    return {"message": "Metrics saved successfully"}


@router.get("/runtime")
async def runtime_metrics() -> Dict[str, Any]:
    """Get runtime API metrics and basic operational alerts."""
    metrics = get_runtime_metrics()
    alerts: List[str] = []
    if metrics["avg_latency_ms"] > 1500:
        alerts.append("Average API latency is above 1500ms")
    if metrics["error_rate"] > 0.05:
        alerts.append("API 5xx error rate is above 5%")
    return {
        "status": "critical" if alerts else "healthy",
        "alerts": alerts,
        "metrics": metrics,
    }
