"""
Main FastAPI Application
========================
Production-grade REST API with:
- JWT Authentication
- Database integration (SQLAlchemy)
- User management
- Prediction history
- Analytics

Launch:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .database import init_db, get_db
from .auth import AUTH_DISABLED, ENVIRONMENT, IS_PRODUCTION
from .routes_auth import router as auth_router
from .routes_predict import router as predict_router, set_model_manager
from .routes_analytics import router as analytics_router
from .routes_upload import router as upload_router
from .routes_monitoring import router as monitoring_router
from .routes_monitoring import record_request_metric
from .rate_limit import register_rate_limit

# Import model manager from fast inference module (with fallback)
try:
    from backend.fast_inference import FastModelManager as ModelManager
    from deployment.fastapi_server import ModelType
    print("✅ Using FastModelManager (ONNX support enabled)")
except ImportError as e:
    print(f"⚠️ FastModelManager could not be imported: {e}")
    print("⚠️ Falling back to standard PyTorch ModelManager")
    from deployment.fastapi_server import ModelManager, ModelType


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    print("🚀 Starting Bank Fraud Detection API...")
    secret_key_present = bool(os.getenv("SECRET_KEY"))
    print(
        f"🔐 Auth config: env={ENVIRONMENT}, "
        f"auth_disabled={AUTH_DISABLED}, secret_key_set={secret_key_present}"
    )
    if IS_PRODUCTION:
        print("🛡️ Production mode active: strict auth/secret validation enforced")
    else:
        print("🧪 Development mode active")
    
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Load model manager
    try:
        manager = ModelManager()
        manager.get_model(ModelType.CNN)  # Pre-load default model
        set_model_manager(manager)
        print(f"✅ Models loaded on {manager.device}")
    except Exception as exc:
        print(f"⚠️ Model preload failed, continuing in degraded mode: {exc}")
    
    yield
    
    print("👋 Shutting down...")


# =============================================================================
# FastAPI Application
# =============================================================================


def _parse_cors_origins() -> list[str]:
    """Resolve CORS origins from environment with safe defaults."""
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        origins = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        origins = [
            "http://localhost:3000",  # Next.js dev
            "http://127.0.0.1:3000",
            "http://localhost:5173",  # Vite dev
            "http://127.0.0.1:5173",
        ]

    if IS_PRODUCTION and "*" in origins:
        raise RuntimeError(
            "CORS_ORIGINS includes '*' in production, which is not allowed. "
            "Set explicit trusted origins."
        )

    return origins

app = FastAPI(
    title="Bank Fraud Detection API",
    description="""
## 🏦 Bank Document Fraud Detection System

AI-powered fraud detection for financial documents using **Vision Transformers (ViT)**
with **Explainable AI** and **Self-Supervised Pretraining**.

### 🆕 Unique Features (HCL Tech)
- 🎲 **Uncertainty Quantification** — Monte Carlo Dropout confidence scoring
- 📈 **Fraud Trend Analytics** — Temporal anomaly detection & branch hotspots
- 🤖 **Active Learning** — Entropy-based human review queue
- 📋 **Regulatory Compliance** — RBI + BASEL III + FATF automated checklist

### Core Features
- 🔐 **JWT Authentication** - Secure user accounts
- 📊 **Multiple Models** - CNN, ViT, Hybrid (CNN+ViT) architectures
- 📈 **Prediction History** - Track and analyze past predictions
- 🎯 **Risk Assessment** - LOW, MEDIUM, HIGH, CRITICAL levels
- 🔍 **XAI** - GradCAM, SHAP, Attention Rollout explanations

### Document Classes
| Class | Description |
|-------|-------------|
| `genuine` | Authentic document |
| `fraud` | Fraudulent document |
| `tampered` | Modified/altered document |
| `forged` | Completely fake document |

### Authentication
Use Bearer token or X-API-Key header for authenticated endpoints.
    """,
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

if IS_PRODUCTION:
    app.add_middleware(HTTPSRedirectMiddleware)
    trusted_hosts_raw = os.getenv("TRUSTED_HOSTS", "").strip()
    trusted_hosts = [item.strip() for item in trusted_hosts_raw.split(",") if item.strip()]
    if trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# Rate limiter — uses client IP by default
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )

# CORS middleware - environment aware
CORS_ORIGINS = _parse_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers_and_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    record_request_metric(latency_ms=latency_ms, status_code=response.status_code)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Response-Time-ms"] = f"{latency_ms:.2f}"
    if IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Register rate limiting
register_rate_limit(app)


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(analytics_router)
app.include_router(upload_router)
app.include_router(monitoring_router)

try:
    from backend import routes_graph
    app.include_router(routes_graph.router)
except ImportError as e:
    print(f"Warning: Could not import routes_graph. Error: {e}")


# =============================================================================
# Base Endpoints
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root - welcome message."""
    return {
        "message": "🏦 Bank Fraud Detection API",
        "version": "3.0.0",
        "docs": "/docs",
        "auth": {
            "register": "/api/auth/register",
            "login": "/api/auth/login"
        },
        "predict": {
            "single": "/api/predict/single",
            "batch": "/api/predict/batch"
        },
        "upload": {
            "detect": "/api/upload/detect",
            "add_to_dataset": "/api/upload/add-to-dataset",
            "pending": "/api/upload/pending",
            "label_pending": "/api/upload/label-pending"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Check API and model health."""
    from .routes_predict import get_model_manager
    
    try:
        manager = get_model_manager()
        return {
            "status": "healthy",
            "database": "connected",
            "models_loaded": list(manager.models.keys()),
            "gpu_available": torch.cuda.is_available(),
            "version": "3.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/api/models", tags=["Models"])
async def list_models():
    """List available ML models."""
    from .routes_predict import get_model_manager
    
    manager = get_model_manager()
    models = []
    
    for model_type in ModelType:
        loaded = model_type.value in manager.models
        info = manager.model_info.get(model_type.value, {})
        models.append({
            "name": info.get('name', model_type.value),
            "id": model_type.value,
            "version": info.get('version', '1.0.0'),
            "classes": ["genuine", "fraud", "tampered", "forged"],
            "input_size": [3, 224, 224],
            "parameters": info.get('parameters', 0),
            "loaded": loaded
        })
    
    return models


# =============================================================================
# Run with Uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
