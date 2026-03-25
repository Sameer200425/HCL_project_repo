"""
Upload Routes — Real-time Document Upload & Detection
======================================================
Handles real document uploads:
  • Saves the image to data/uploads/ for audit trail
  • Optionally saves to data/raw_images/{class}/ for re-training
  • Returns fraud prediction immediately

Endpoints:
  POST /api/upload/detect        — upload a document and get instant prediction
  POST /api/upload/add-to-dataset — upload + label + add to training set
  GET  /api/upload/pending       — list uploaded files awaiting labelling
"""

import io
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

router = APIRouter(prefix="/api/upload", tags=["Upload & Real-Time Detection"])

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw_images"
VALID_CLASSES = ["genuine", "fraud", "tampered", "forged"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _ensure_dirs():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for cls in VALID_CLASSES:
        (RAW_IMAGES_DIR / cls).mkdir(parents=True, exist_ok=True)


_ensure_dirs()


# =============================================================================
# Pydantic Models
# =============================================================================

class DetectionResponse(BaseModel):
    filename: str
    saved_path: str
    predicted_class: str
    confidence: float
    probabilities: dict
    risk_level: str
    inference_time_ms: float


class AddToDatasetResponse(BaseModel):
    filename: str
    saved_path: str
    label: str
    message: str


class PendingFile(BaseModel):
    filename: str
    uploaded_at: str
    size_kb: float


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/detect", response_model=DetectionResponse)
async def upload_and_detect(
    file: UploadFile = File(..., description="Bank document / cheque / statement image"),
    model: str = Query(default="cnn", description="Model to use: cnn | vit | hybrid"),
    save: bool = Query(default=True, description="Save the upload to data/uploads/ for audit"),
):
    """
    Upload a real document and get instant fraud prediction.
    The image is saved to data/uploads/ (unless save=false).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")

    # Save to uploads dir
    saved_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{file.filename}"
    saved_path = UPLOADS_DIR / saved_name

    if save:
        try:
            with open(saved_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    from .routes_predict import get_model_manager, ModelType

    try:
        model_enum = ModelType(model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model}. Error: {str(e)}")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    try:
        manager = get_model_manager()
        result = manager.predict(image, model_enum)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return DetectionResponse(
        filename=file.filename or "unknown",
        saved_path=str(saved_path.relative_to(PROJECT_ROOT)) if save else "not saved",
        predicted_class=result["class_name"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        risk_level=result["risk_level"],
        inference_time_ms=result["inference_time_ms"],
    )


@router.post("/add-to-dataset", response_model=AddToDatasetResponse)
async def add_to_training_dataset(
    file: UploadFile = File(..., description="Image to add to the training set"),
    label: str = Query(..., description="True class: genuine | fraud | tampered | forged"),
):
    """
    Upload a labelled document directly into data/raw_images/{label}/.
    After adding enough images, re-run training to improve accuracy.
    """
    label = label.lower().strip()
    if label not in VALID_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label '{label}'. Must be one of {VALID_CLASSES}",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{file.filename}"
    dest = RAW_IMAGES_DIR / label / fname

    with open(dest, "wb") as f:
        f.write(contents)

    return AddToDatasetResponse(
        filename=file.filename or "unknown",
        saved_path=str(dest.relative_to(PROJECT_ROOT)),
        label=label,
        message=f"Image saved to {label}/ training folder. Re-run 'python setup_datasets.py --prepare' then 'python run_pipeline.py' to retrain.",
    )


@router.get("/pending", response_model=List[PendingFile])
async def list_pending_uploads():
    """
    List images in data/uploads/ that haven't been labelled yet.
    These can be reviewed and moved to the training set.
    """
    files: List[PendingFile] = []
    if UPLOADS_DIR.exists():
        for f in sorted(UPLOADS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix.lower() in IMAGE_EXTS:
                files.append(PendingFile(
                    filename=f.name,
                    uploaded_at=datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    size_kb=round(f.stat().st_size / 1024, 1),
                ))
    return files


@router.post("/label-pending")
async def label_pending_upload(
    filename: str = Query(..., description="Name of file in data/uploads/"),
    label: str = Query(..., description="True class: genuine | fraud | tampered | forged"),
):
    """
    Move a pending upload into data/raw_images/{label}/ for training.
    """
    label = label.lower().strip()
    if label not in VALID_CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid label. Use one of {VALID_CLASSES}")

    src = UPLOADS_DIR / filename
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    dest = RAW_IMAGES_DIR / label / filename
    shutil.move(str(src), str(dest))

    return {"message": f"Moved {filename} → raw_images/{label}/", "path": str(dest.relative_to(PROJECT_ROOT))}
