"""
FastAPI Server for Bank Fraud Detection
========================================
Production-grade REST API with:
- Async request handling
- Batch inference
- Model versioning
- Health checks
- OpenAPI documentation
- Request validation
- Structured responses

Launch:
    uvicorn deployment.fastapi_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import io
import time
import base64
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union, cast, Sequence
from enum import Enum
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# =============================================================================
# Pydantic Models
# =============================================================================

class ModelType(str, Enum):
    CNN = "cnn"
    VIT = "vit"
    VIT_SSL = "vit_ssl"
    HYBRID = "hybrid"


class PredictionResult(BaseModel):
    """Single prediction result."""
    class_name: str = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Per-class probabilities")
    risk_level: str = Field(..., description="Risk classification (LOW/MEDIUM/HIGH/CRITICAL)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class SinglePredictionResponse(BaseModel):
    """Response for single image prediction."""
    success: bool
    prediction: PredictionResult
    model_name: str
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    success: bool
    predictions: List[PredictionResult]
    total_images: int
    total_inference_time_ms: float
    model_name: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    available_models: List[str]
    gpu_available: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    name: str
    version: str
    classes: List[str]
    input_size: List[int]
    parameters: int
    loaded: bool


# =============================================================================
# Model Manager
# =============================================================================

CLASS_NAMES = ['genuine', 'fraud', 'tampered', 'forged']

RISK_THRESHOLDS = {
    'genuine': {'LOW': 0.8, 'MEDIUM': 0.6, 'HIGH': 0.4},
    'fraud': {'CRITICAL': 0.7, 'HIGH': 0.5, 'MEDIUM': 0.3},
    'tampered': {'CRITICAL': 0.7, 'HIGH': 0.5, 'MEDIUM': 0.3},
    'forged': {'CRITICAL': 0.7, 'HIGH': 0.5, 'MEDIUM': 0.3},
}


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.model_info: Dict[str, Dict] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.project_root = Path(__file__).resolve().parent.parent
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_cnn(self) -> nn.Module:
        """Load CNN model."""
        from torchvision import models as tv_models
        
        backbone = tv_models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        setattr(backbone, 'fc', nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
        ))
        
        class CNNModel(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            def forward(self, x):
                return self.backbone(x)
        
        model = CNNModel(backbone)
        ckpt_path = self.project_root / "checkpoints" / "cnn_best.pth"
        
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
        
        return model.eval().to(self.device)
    
    def load_vit(self) -> nn.Module:
        """Load ViT model."""
        from models.vit_model import VisionTransformer
        
        model = VisionTransformer(
            image_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=4,
            embed_dim=128,
            num_heads=4,
            num_layers=4,
            mlp_dim=256,
            dropout=0.0
        )
        
        ckpt_path = self.project_root / "checkpoints" / "vit_best.pth"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
        
        return model.eval().to(self.device)
    
    def load_hybrid(self) -> nn.Module:
        """Load Hybrid model."""
        from models.hybrid_model import HybridCNNViT
        
        # Match training config: embed_dim=128, num_heads=4, num_layers=2
        model = HybridCNNViT(
            num_classes=4,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )
        ckpt_path = self.project_root / "checkpoints" / "hybrid_best.pth"
        
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
        
        return model.eval().to(self.device)
    
    def load_vit_ssl(self) -> nn.Module:
        """Load ViT model with SSL pretraining."""
        from models.vit_model import VisionTransformer
        
        model = VisionTransformer(
            image_size=224,
            patch_size=16,
            in_channels=3,
            num_classes=4,
            embed_dim=128,
            num_heads=4,
            num_layers=4,
            mlp_dim=256,
            dropout=0.0
        )
        
        ckpt_path = self.project_root / "checkpoints" / "vit_ssl_best.pth"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
        
        return model.eval().to(self.device)
    
    def get_model(self, model_type: ModelType) -> nn.Module:
        """Get or load a model."""
        if model_type.value not in self.models:
            if model_type == ModelType.CNN:
                self.models['cnn'] = self.load_cnn()
                self.model_info['cnn'] = {
                    'name': 'CNN (ResNet50)',
                    'version': '1.0.0',
                    'parameters': sum(p.numel() for p in self.models['cnn'].parameters())
                }
            elif model_type == ModelType.VIT:
                self.models['vit'] = self.load_vit()
                self.model_info['vit'] = {
                    'name': 'Vision Transformer',
                    'version': '1.0.0',
                    'parameters': sum(p.numel() for p in self.models['vit'].parameters())
                }
            elif model_type == ModelType.HYBRID:
                self.models['hybrid'] = self.load_hybrid()
                self.model_info['hybrid'] = {
                    'name': 'Hybrid CNN+ViT',
                    'version': '1.0.0',
                    'parameters': sum(p.numel() for p in self.models['hybrid'].parameters())
                }
            elif model_type == ModelType.VIT_SSL:
                self.models['vit_ssl'] = self.load_vit_ssl()
                self.model_info['vit_ssl'] = {
                    'name': 'ViT + SSL (MAE)',
                    'version': '1.0.0',
                    'parameters': sum(p.numel() for p in self.models['vit_ssl'].parameters())
                }
        
        return self.models[model_type.value]
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = cast(torch.Tensor, self.transform(image))
        return tensor.unsqueeze(0).to(self.device)
    
    def classify_risk(self, class_name: str, confidence: float) -> str:
        """Classify risk level based on prediction."""
        if class_name == 'genuine':
            if confidence >= 0.8:
                return 'LOW'
            elif confidence >= 0.6:
                return 'MEDIUM'
            else:
                return 'HIGH'
        else:  # fraud, tampered, forged
            if confidence >= 0.7:
                return 'CRITICAL'
            elif confidence >= 0.5:
                return 'HIGH'
            elif confidence >= 0.3:
                return 'MEDIUM'
            else:
                return 'LOW'
    
    @torch.no_grad()
    def predict(self, image: Image.Image, model_type: ModelType) -> Dict:
        """Run inference on a single image."""
        model = self.get_model(model_type)
        input_tensor = self.preprocess_image(image)
        
        start_time = time.perf_counter()
        output = model(input_tensor)
        inference_time = (time.perf_counter() - start_time) * 1000
        
        probs = F.softmax(output, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx].item())
        
        return {
            'class_name': pred_class,
            'confidence': round(confidence, 4),
            'probabilities': {
                name: round(p.item(), 4) 
                for name, p in zip(CLASS_NAMES, probs)
            },
            'risk_level': self.classify_risk(pred_class, confidence),
            'inference_time_ms': round(inference_time, 2)
        }
    
    @torch.no_grad()
    def predict_batch(self, images: Sequence[Image.Image], model_type: ModelType) -> List[Dict]:
        """Run inference on a batch of images."""
        model = self.get_model(model_type)
        
        # Preprocess all images
        tensors = [self.preprocess_image(img) for img in images]
        batch = torch.cat(tensors, dim=0)
        
        start_time = time.perf_counter()
        outputs = model(batch)
        total_time = (time.perf_counter() - start_time) * 1000
        
        probs_batch = F.softmax(outputs, dim=1)
        
        results = []
        for i, probs in enumerate(probs_batch):
            pred_idx = int(probs.argmax().item())
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx].item())
            
            results.append({
                'class_name': pred_class,
                'confidence': round(confidence, 4),
                'probabilities': {
                    name: round(p.item(), 4) 
                    for name, p in zip(CLASS_NAMES, probs)
                },
                'risk_level': self.classify_risk(pred_class, confidence),
                'inference_time_ms': round(total_time / len(images), 2)
            })
        
        return results


# =============================================================================
# FastAPI Application
# =============================================================================

# Global model manager
model_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Get the model manager, raising error if not initialized."""
    if model_manager is None:
        raise RuntimeError("Model manager not initialized")
    return model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global model_manager
    print("Starting up FastAPI server...")
    model_manager = ModelManager()
    # Pre-load default model
    model_manager.get_model(ModelType.CNN)
    print(f"Default model loaded on {model_manager.device}")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Bank Fraud Detection API",
    description="""
    ## Vision Transformer based Financial Document Fraud Detection
    
    This API provides endpoints for detecting fraudulent financial documents using
    state-of-the-art deep learning models including CNN, Vision Transformer (ViT),
    and Hybrid CNN+ViT architectures.
    
    ### Features
    - Single and batch image prediction
    - Multiple model support (CNN, ViT, Hybrid)
    - Risk level classification
    - Detailed confidence scores
    
    ### Classes
    - **genuine**: Authentic document
    - **fraud**: Fraudulent document
    - **tampered**: Modified/altered document
    - **forged**: Completely fake document
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root - welcome message."""
    return {
        "message": "Bank Fraud Detection API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API and model health status."""
    mgr = get_manager()
    return {
        "status": "healthy",
        "model_loaded": len(mgr.models) > 0,
        "available_models": list(mgr.models.keys()),
        "gpu_available": torch.cuda.is_available(),
        "version": "2.0.0"
    }


@app.get("/models", response_model=List[ModelInfoResponse], tags=["Models"])
async def list_models():
    """List available models and their information."""
    mgr = get_manager()
    models = []
    for model_type in ModelType:
        loaded = model_type.value in mgr.models
        info = mgr.model_info.get(model_type.value, {})
        models.append({
            "name": info.get('name', model_type.value),
            "version": info.get('version', '1.0.0'),
            "classes": CLASS_NAMES,
            "input_size": [3, 224, 224],
            "parameters": info.get('parameters', 0),
            "loaded": loaded
        })
    return models


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Prediction"])
async def predict_single(
    file: UploadFile = File(..., description="Image file to analyze"),
    model: ModelType = Query(default=ModelType.CNN, description="Model to use")
):
    """
    Predict fraud class for a single document image.
    
    - **file**: Image file (JPEG, PNG)
    - **model**: Model to use (cnn, vit, hybrid)
    
    Returns prediction with confidence scores and risk level.
    """
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Run prediction
        mgr = get_manager()
        result = mgr.predict(image, model)
        
        return {
            "success": True,
            "prediction": result,
            "model_name": model.value,
            "model_version": mgr.model_info.get(model.value, {}).get('version', '1.0.0'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Image files to analyze"),
    model: ModelType = Query(default=ModelType.CNN, description="Model to use")
):
    """
    Predict fraud class for multiple document images.
    
    - **files**: List of image files (max 32)
    - **model**: Model to use (cnn, vit, hybrid)
    
    Returns predictions for all images with aggregate statistics.
    """
    if len(files) > 32:
        raise HTTPException(status_code=400, detail="Maximum 32 images per batch")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        images = []
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            images.append(image)
        
        start_time = time.perf_counter()
        results = get_manager().predict_batch(images, model)
        total_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "predictions": results,
            "total_images": len(images),
            "total_inference_time_ms": round(total_time, 2),
            "model_name": model.value,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict/base64", response_model=SinglePredictionResponse, tags=["Prediction"])
async def predict_base64(
    image_data: str,
    model: ModelType = Query(default=ModelType.CNN, description="Model to use")
):
    """
    Predict fraud class from base64-encoded image.
    
    - **image_data**: Base64-encoded image string
    - **model**: Model to use (cnn, vit, hybrid)
    """
    try:
        # Decode base64
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        mgr = get_manager()
        result = mgr.predict(image, model)
        
        return {
            "success": True,
            "prediction": result,
            "model_name": model.value,
            "model_version": mgr.model_info.get(model.value, {}).get('version', '1.0.0'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/load-model", tags=["Models"])
async def load_model(model: ModelType = Query(..., description="Model to load")):
    """Pre-load a model into memory."""
    try:
        get_manager().get_model(model)
        return {
            "success": True,
            "model": model.value,
            "message": f"Model {model.value} loaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "deployment.fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
