"""
Fast Inference Module using ONNX Runtime
========================================
Optimized inference manager that transparently uses ONNX models for speed
when available, falling back to PyTorch for compatibility.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import base ModelManager
from deployment.fastapi_server import ModelManager, ModelType, CLASS_NAMES

logger = logging.getLogger(__name__)


class FastModelManager(ModelManager):
    """
    Enhanced ModelManager that uses ONNX Runtime for faster inference.
    Falls back to normal PyTorch execution if ONNX models are missing.
    """
    
    def __init__(self):
        super().__init__()
        self.onnx_sessions: Dict[str, Any] = {}
        
        if ONNX_AVAILABLE:
            logger.info(f"✅ Fast Inference: ONNX Runtime available (device={ort.get_device()})")
        else:
            logger.warning("⚠️ Fast Inference: ONNX Runtime NOT found. Using PyTorch only.")

    def _get_onnx_path(self, model_type: str) -> Path:
        """Get the path to the ONNX model file."""
        return self.project_root / "checkpoints" / f"{model_type}_best.onnx"

    def load_onnx_session(self, model_type: str) -> bool:
        """Try to load ONNX session for a model type."""
        if not ONNX_AVAILABLE:
            return False
            
        onnx_path = self._get_onnx_path(model_type)
        if not onnx_path.exists():
            return False
            
        try:
            # Configure providers (prefer CUDA execution provider if available)
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                 providers.insert(0, 'CUDAExecutionProvider')
                 
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.onnx_sessions[model_type] = session
            
            # Store metadata for consistency
            self.model_info[model_type] = {
                'name': f"{model_type.upper()} (ONNX Optimized)",
                'version': '1.0.0-onnx',
                'parameters': "N/A (Optimized Graph)"
            }
            logger.info(f"🚀 Loaded ONNX model for {model_type}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model for {model_type}: {e}")
            return False

    def get_model(self, model_type: ModelType) -> Any:
        """
        Get model or ONNX session. 
        Prioritizes ONNX for supported types (vit, hybrid).
        """
        m_type = model_type.value
        
        # 1. Check if we already have an ONNX session
        if m_type in self.onnx_sessions:
            return self.onnx_sessions[m_type]
            
        # 2. Try to load ONNX session if we haven't tried yet
        #    Only for ViT and Hybrid which we converted
        if m_type in ['vit', 'hybrid']:
            if self.load_onnx_session(m_type):
                return self.onnx_sessions[m_type]
        
        # 3. Fallback to standard PyTorch model
        return super().get_model(model_type)

    def preprocess_image_numpy(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image specifically for ONNX Runtime (returns numpy array).
        Matches the PyTorch transforms: Resize(224) -> ToTensor -> Normalize
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        
        # ToTensor (0-255 -> 0-1, HWC -> CHW)
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_np = (img_np - mean) / std
        
        # Add batch dimension
        return np.expand_dims(img_np, axis=0)

    def predict(self, image: Image.Image, model_type: ModelType) -> Dict:
        """Run inference using ONNX if available, else PyTorch."""
        m_type = model_type.value
        
        # --- PYTORCH PATH ---
        if m_type not in self.onnx_sessions:
            # Try to load it (it might load ONNX if available)
            self.get_model(model_type)
            # If still not in onnx_sessions, use parent (PyTorch)
            if m_type not in self.onnx_sessions:
                return super().predict(image, model_type)
        
        # --- ONNX PATH ---
        session = self.onnx_sessions[m_type]
        input_name = session.get_inputs()[0].name
        
        # Preprocess
        input_data = self.preprocess_image_numpy(image)
        
        # Run Inference
        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: input_data})
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Post-process
        logits = outputs[0][0]  # shape (4,)
        probs = self._softmax(logits)
        
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        
        return {
            'class_name': pred_class,
            'confidence': round(confidence, 4),
            'probabilities': {
                name: round(float(p), 4) 
                for name, p in zip(CLASS_NAMES, probs)
            },
            'risk_level': self.classify_risk(pred_class, confidence),
            'inference_time_ms': round(inference_time, 2)
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for a vector x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

