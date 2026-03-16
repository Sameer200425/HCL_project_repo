"""
Prediction Routes with Database Integration
============================================
API endpoints for fraud detection predictions.
"""

import io
import hashlib
from datetime import datetime
from typing import List, Optional, cast
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from PIL import Image
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from .database import get_db
from .models import User, Prediction, ModelMetrics
from .auth import get_optional_user, get_current_active_user

# Image quality
import numpy as np
import cv2  # Should be available if installed
# Removed strict image quality threshold to allow all standard documents
_quality_checker = None

router = APIRouter(prefix="/api/predict", tags=["Predictions"])
limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# Pydantic Models
# =============================================================================

class PredictionResponse(BaseModel):
    id: int
    filename: Optional[str] = None
    model_name: str
    predicted_class: str
    confidence: float
    probabilities: dict = {}
    risk_level: str
    inference_time_ms: float = 0.0
    explanation: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "examples": [
                {
                    "id": 42,
                    "filename": "cheque_scan_001.png",
                    "model_name": "vit",
                    "predicted_class": "genuine",
                    "confidence": 0.9734,
                    "probabilities": {
                        "genuine": 0.9734,
                        "fraud": 0.0121,
                        "tampered": 0.0089,
                        "forged": 0.0056
                    },
                    "risk_level": "low",
                    "inference_time_ms": 48.3,
                    "explanation": "The cheque image analysis indicates this document is authentic with 97% confidence.",
                    "created_at": "2024-06-15T10:30:00"
                }
            ]
        }


def _pred_resp(p: Prediction, explanation: Optional[str] = None) -> PredictionResponse:
    """Convert a Prediction ORM object to PredictionResponse, bypassing Column type issues."""
    resp = PredictionResponse.model_validate(p)
    if explanation:
        resp.explanation = explanation
    return resp


def generate_explanation(class_name: str, confidence: float, probabilities: dict) -> Optional[str]:
    """Generate natural language explanation for prediction."""
    if _fraud_explainer is None:
        return None
    try:
        return _fraud_explainer.explain_image_classification(
            class_name=class_name,
            confidence=confidence,
            class_probs=probabilities
        )
    except Exception:
        return None


class PredictionHistory(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    page: int
    page_size: int


class PredictionStats(BaseModel):
    total_predictions: int
    fraud_detected: int = 0
    genuine_documents: int = 0
    by_class: dict
    by_risk_level: dict
    by_model: dict = {}
    avg_confidence: float
    avg_inference_time_ms: float

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "total_predictions": 150,
                    "fraud_detected": 23,
                    "genuine_documents": 127,
                    "by_class": {
                        "genuine": 127,
                        "fraud": 12,
                        "tampered": 8,
                        "forged": 3
                    },
                    "by_risk_level": {
                        "low": 120,
                        "medium": 18,
                        "high": 9,
                        "critical": 3
                    },
                    "by_model": {"vit": 80, "cnn": 50, "hybrid": 20},
                    "avg_confidence": 0.9142,
                    "avg_inference_time_ms": 52.7
                }
            ]
        }


# =============================================================================
# Helper Functions
# =============================================================================

def compute_image_hash(content: bytes) -> str:
    """Compute SHA-256 hash of image content."""
    return hashlib.sha256(content).hexdigest()


def save_prediction_to_db(
    db: Session,
    user_id: Optional[int],
    filename: Optional[str],
    image_hash: Optional[str],
    model_name: str,
    result: dict
) -> Prediction:
    """Save prediction to database."""
    prediction = Prediction(
        user_id=user_id,
        filename=filename,
        image_hash=image_hash,
        model_name=model_name,
        predicted_class=result["class_name"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        risk_level=result["risk_level"],
        inference_time_ms=result["inference_time_ms"]
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


# =============================================================================
# Prediction Endpoints
# =============================================================================

# Import ModelManager from deployment
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enum import Enum

# Import fraud explainer for natural language explanations
try:
    from llm_explainer import FraudExplainer
    _fraud_explainer = FraudExplainer(use_llm=False)
except ImportError:
    _fraud_explainer = None

# Import MLOps monitoring
try:
    from .routes_monitoring import get_model_monitor
    _mlops_enabled = True
except ImportError:
    _mlops_enabled = False
    get_model_monitor = None

class ModelType(str, Enum):
    CNN = "cnn"
    VIT = "vit"
    VIT_SSL = "vit_ssl"
    HYBRID = "hybrid"


# Global model manager (will be set by main app)
_model_manager = None

def set_model_manager(manager):
    global _model_manager
    _model_manager = manager

def get_model_manager():
    if _model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    return _model_manager


@router.post("/single", response_model=PredictionResponse)
@limiter.limit("60/minute")
async def predict_single(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    model: ModelType = Query(default=ModelType.CNN, description="Model to use"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Predict fraud class for a single document image.
    Results are saved to database if authenticated.
    """
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image_hash = compute_image_hash(contents)
        
        # 1. Quality Check
        if _quality_checker is not None:
            # Convert raw bytes to numpy array for OpenCV
            nparr = np.frombuffer(contents, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_np is not None:
                passed, issues = _quality_checker.analyze(img_np)
                if not passed:
                    # Return error with specific issues
                    raise HTTPException(
                        status_code=400, 
                        detail={
                            "error": "Image Quality Check Failed",
                            "issues": issues,
                            "suggestion": "Please retake the photo in better lighting."
                        }
                    )
        
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Run prediction
        manager = get_model_manager()
        result = manager.predict(image, model)
        
        # Log prediction to MLOps monitor
        if _mlops_enabled and get_model_monitor:
            try:
                monitor = get_model_monitor()
                monitor.log_prediction(
                    input_data=image_hash,
                    predicted_class=result["class_name"],
                    confidence=result["confidence"],
                    class_probabilities=result["probabilities"],
                    inference_time_ms=result["inference_time_ms"]
                )
            except Exception:
                pass  # Don't fail prediction if monitoring fails
        
        # Generate natural language explanation
        explanation = generate_explanation(
            class_name=result["class_name"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
        
        # Save to database
        user_id = cast(int, current_user.id) if current_user else None
        prediction = save_prediction_to_db(
            db=db,
            user_id=user_id,
            filename=file.filename,
            image_hash=image_hash,
            model_name=model.value,
            result=result
        )
        
        return _pred_resp(prediction, explanation=explanation)
        
    except HTTPException:
        raise


@router.post("/batch", response_model=List[PredictionResponse])
@limiter.limit("20/minute")
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="Image files to analyze"),
    model: ModelType = Query(default=ModelType.CNN, description="Model to use"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Predict fraud class for multiple images (max 32)."""
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 images per batch")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        images = []
        file_data = []
        
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not an image"
                )
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            images.append(image)
            file_data.append({
                "filename": file.filename,
                "hash": compute_image_hash(contents)
            })
        
        # Run batch prediction
        manager = get_model_manager()
        results = manager.predict_batch(images, model)
        
        # Log predictions to MLOps monitor
        if _mlops_enabled and get_model_monitor:
            try:
                monitor = get_model_monitor()
                for i, result in enumerate(results):
                    monitor.log_prediction(
                        input_data=file_data[i]["hash"],
                        predicted_class=result["class_name"],
                        confidence=result["confidence"],
                        class_probabilities=result["probabilities"],
                        inference_time_ms=result["inference_time_ms"]
                    )
            except Exception:
                pass  # Don't fail prediction if monitoring fails
        
        # Save all predictions
        predictions = []
        user_id = cast(int, current_user.id) if current_user else None
        
        for i, result in enumerate(results):
            # Generate explanation for each prediction
            explanation = generate_explanation(
                class_name=result["class_name"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            )
            prediction = save_prediction_to_db(
                db=db,
                user_id=user_id,
                filename=file_data[i]["filename"],
                image_hash=file_data[i]["hash"],
                model_name=model.value,
                result=result
            )
            predictions.append(_pred_resp(prediction, explanation=explanation))
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# =============================================================================
# History & Analytics Endpoints
# =============================================================================

@router.get("/history", response_model=PredictionHistory)
async def get_prediction_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    model: Optional[str] = None,
    risk_level: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get prediction history for current user."""
    query = db.query(Prediction).filter(Prediction.user_id == current_user.id)
    
    if model:
        query = query.filter(Prediction.model_name == model)
    if risk_level:
        query = query.filter(Prediction.risk_level == risk_level)
    
    total = query.count()
    predictions = query.order_by(Prediction.created_at.desc())\
        .offset((page - 1) * page_size)\
        .limit(page_size)\
        .all()
    
    return PredictionHistory(
        predictions=[_pred_resp(p) for p in predictions],
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/stats", response_model=PredictionStats)
async def get_prediction_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get prediction statistics for current user."""
    query = db.query(Prediction).filter(Prediction.user_id == current_user.id)
    
    total = query.count()
    
    if total == 0:
        return PredictionStats(
            total_predictions=0,
            fraud_detected=0,
            genuine_documents=0,
            by_class={},
            by_risk_level={},
            by_model={},
            avg_confidence=0,
            avg_inference_time_ms=0
        )
    
    # By class
    class_counts = db.query(
        Prediction.predicted_class,
        func.count(Prediction.id)
    ).filter(
        Prediction.user_id == current_user.id
    ).group_by(Prediction.predicted_class).all()
    
    by_class = {c: count for c, count in class_counts}
    
    # Compute fraud_detected (fraud + tampered + forged) and genuine_documents
    fraud_detected = sum(count for cls, count in class_counts if cls in ('fraud', 'tampered', 'forged'))
    genuine_documents = by_class.get('genuine', 0)
    
    # By risk level
    risk_counts = db.query(
        Prediction.risk_level,
        func.count(Prediction.id)
    ).filter(
        Prediction.user_id == current_user.id
    ).group_by(Prediction.risk_level).all()
    
    by_risk = {r: count for r, count in risk_counts}
    
    # By model
    model_counts = db.query(
        Prediction.model_name,
        func.count(Prediction.id)
    ).filter(
        Prediction.user_id == current_user.id
    ).group_by(Prediction.model_name).all()
    
    by_model = {m: count for m, count in model_counts}
    
    # Averages
    avg_conf = db.query(func.avg(Prediction.confidence))\
        .filter(Prediction.user_id == current_user.id).scalar() or 0
    avg_time = db.query(func.avg(Prediction.inference_time_ms))\
        .filter(Prediction.user_id == current_user.id).scalar() or 0
    
    return PredictionStats(
        total_predictions=total,
        fraud_detected=fraud_detected,
        genuine_documents=genuine_documents,
        by_class=by_class,
        by_risk_level=by_risk,
        by_model=by_model,
        avg_confidence=round(float(avg_conf), 4),
        avg_inference_time_ms=round(float(avg_time), 2)
    )


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific prediction by ID."""
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return _pred_resp(prediction)


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a prediction from history."""
    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id,
        Prediction.user_id == current_user.id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    db.delete(prediction)
    db.commit()
    return {"message": "Prediction deleted"}
