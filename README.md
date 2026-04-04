# Bank Fraud Detection with Vision Transformers (ViT)

**Domain:** Banking & Finance — AI/ML + Full-Stack Development  
**Frontend:** Next.js 14 + React 18 + TypeScript + Tailwind CSS  
**Backend:** FastAPI + SQLAlchemy + JWT Authentication  
**ML Models:** Vision Transformer (ViT) | Hybrid CNN+ViT | ResNet50 CNN

---

## 🎯 Overview

An end-to-end banking document fraud detection system using Vision Transformers with a modern web interface. The system analyzes financial documents (cheques, KYC forms, statements) and classifies them as `genuine`, `fraud`, `tampered`, or `forged`.

### Key Highlights
- **4 Deep Learning Models** — CNN, ViT, ViT+SSL, Hybrid CNN-ViT
- **Smart Scanner** — Real-time camera-based document scanning with quality detection
- **Explainable AI** — Attention maps, Grad-CAM, SHAP analysis
- **Production Ready** — JWT auth, rate limiting, Docker support

---

## 🖼️ Screenshots

| Dashboard | Smart Scanner | Prediction Results |
|-----------|---------------|-------------------|
| Analytics & Stats | Camera with quality feedback | Model predictions with confidence |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- CUDA 11.8+ (optional, for GPU training)

### 1. Clone & Setup Backend

```bash
cd bank_vit_project

# Create Python virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Frontend

```bash
cd frontend
npm install
```

### 2.1 Initialize Database Migrations (Alembic)

```bash
# from project root
alembic upgrade head

# create a new migration after model changes
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

### 3. Run the Application

**Terminal 1 — Backend API:**
```bash
cd bank_vit_project
.venv\Scripts\activate
uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

**Terminal 2 — Frontend:**
```bash
cd bank_vit_project/frontend
npm run dev
```

### 4. Access the Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8001
- **API Docs:** http://localhost:8001/docs
- **Runtime Metrics:** http://localhost:8001/api/monitoring/runtime

### 5. Quick Pipeline Validation (No Long Training)

Use the pipeline runner with lightweight switches when you want to validate specific steps quickly:

```bash
# Run only model comparison (Task 3)
python run_all_tasks.py --only-task3

# Run only PDF report generation (Task 4)
python run_all_tasks.py --only-task4

# Skip training-heavy tasks (Tasks 1 and 2), run Tasks 3 and 4
python run_all_tasks.py --skip-training
```

Notes:
- `--only-task3` and `--only-task4` are mutually exclusive.
- Module warmup is opt-in (it can be slow on some machines). To enable repo-wide import warmup for audits, set `MODULE_USAGE_WARMUP=1` before running pipeline scripts.

### Default Login / Public Access
```
Email: admin@example.com
Password: admin123
```

Authentication is disabled in the current demo build, so the dashboard and prediction tools are open to everyone. Set `DISABLE_AUTH=false` (backend) and `NEXT_PUBLIC_AUTH_DISABLED=false` (frontend) if you need to restore login workflows.

### Environment-Safe Auth Configuration

The backend auth module enforces different behavior for development vs production.

Set these environment variables explicitly:

```bash
# Required in production-safe setups
ENVIRONMENT=development   # or production
SECRET_KEY=your-long-random-secret-32-plus-chars
DISABLE_AUTH=false
LOGIN_MAX_FAILS=5
LOGIN_LOCK_WINDOW_MINUTES=15
LOGIN_LOCK_MINUTES=15
```

Behavior summary:
- Development (`ENVIRONMENT=development`):
	- If `SECRET_KEY` is missing, an ephemeral key is generated and a warning is logged.
	- Tokens become invalid after restart when using ephemeral key.
- Production (`ENVIRONMENT=production`):
	- `SECRET_KEY` is mandatory.
	- `SECRET_KEY` must be at least 32 characters.
	- `DISABLE_AUTH=true` is blocked and raises an error.
	- HTTP security headers and HTTPS redirect middleware are enabled.

### Runtime Readiness Checklist (Current Status)

The following runtime checks have been executed in the local environment (as of 18 Mar 2026):

- **Backend & API tests** — `pytest tests/test_backend.py tests/test_api.py tests/test_integration_smoke.py` → **74 passed**.
- **Deployment tests** — `pytest tests/test_deployment.py` → **16 passed**, ONNX export verified after installing `onnxscript`.
- **Quick pipeline check (Task 4)** — `python run_all_tasks.py --only-task4` → PDF audit reports successfully generated for all 4 classes using the current `vit_best.pth` checkpoint.
- **Auth & CORS hardening** — backend enforces env-safe `SECRET_KEY` rules and blocks wildcard CORS in production.
- **Neo4j graph engine** — `GraphEngine.test_connection()` currently reports **no live Neo4j at bolt://localhost:7687**, so graph features run in degraded/mock mode on this machine.

To reach full production parity, you should also:

- Run Neo4j locally or in Docker, apply `neo4j_seed_data.py`, and confirm that `GraphEngine.test_connection()` returns `True`.
- Exercise the full Docker compose stack (`docker-compose up`) in the target infrastructure.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │  Login   │ │Dashboard │ │ Predict  │ │    History       │   │
│  │ Register │ │Analytics │ │ Upload   │ │  Past Results    │   │
│  └──────────┘ └──────────┘ │ Scanner  │ └──────────────────┘   │
│                            └──────────┘                          │
├─────────────────────────────────────────────────────────────────┤
│                        BACKEND (FastAPI)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │  Auth    │ │ Predict  │ │ Upload   │ │   Analytics      │   │
│  │  JWT     │ │  Models  │ │  Files   │ │   Metrics        │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     ML MODELS (PyTorch)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │   CNN    │ │   ViT    │ │ ViT+SSL  │ │  Hybrid CNN+ViT  │   │
│  │ ResNet50 │ │  Base/16 │ │   MAE    │ │  ResNet + Trans  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
bank_vit_project/
├── frontend/                     # Next.js Frontend
│   ├── src/
│   │   ├── app/                  # App Router pages
│   │   │   ├── login/            # Login page
│   │   │   ├── register/         # Registration page
│   │   │   ├── dashboard/        # Main dashboard
│   │   │   ├── predict/          # Prediction & Smart Scanner
│   │   │   ├── history/          # Prediction history
│   │   │   └── settings/         # User settings
│   │   ├── components/           # React components
│   │   │   ├── ui/               # Shadcn UI components
│   │   │   └── SmartScanner.tsx  # Camera scanner component
│   │   └── lib/                  # Utilities & stores
│   ├── package.json
│   └── next.config.js
│
├── backend/                      # FastAPI Backend
│   ├── main.py                   # App entry point
│   ├── auth.py                   # JWT authentication
│   ├── database.py               # SQLAlchemy setup
│   ├── models.py                 # Database models
│   ├── routes_auth.py            # Auth endpoints
│   ├── routes_predict.py         # Prediction endpoints
│   ├── routes_analytics.py       # Analytics endpoints
│   └── routes_upload.py          # File upload endpoints
│
├── models/                       # ML Model Definitions
│   ├── vit_model.py              # Vision Transformer
│   ├── hybrid_model.py           # Hybrid CNN+ViT
│   └── knowledge_distillation.py # Model distillation
│
├── deployment/                   # Deployment Tools
│   ├── fastapi_server.py         # Model serving
│   ├── onnx_export.py            # ONNX conversion
│   └── model_registry.py         # Model versioning
│
├── explainability/               # XAI Tools
│   ├── attention_visualization.py
│   ├── gradcam.py
│   ├── shap_analysis.py
│   └── report_generator.py
│
├── ssl_pretraining/              # Self-Supervised Learning
│   ├── mae_model.py              # Masked Autoencoder
│   └── contrastive_model.py      # SimCLR-style
│
├── checkpoints/                  # Trained Model Weights
│   ├── cnn_best.pth
│   ├── vit_best.pth
│   ├── vit_ssl_best.pth
│   ├── hybrid_best.pth
│   └── mae_pretrained.pth
│
├── config.yaml                   # Configuration
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Docker orchestration
└── Dockerfile                    # Container build
```

---

## 🤖 ML Models

### Available Models

| Model | Description | Accuracy | Params |
|-------|-------------|----------|--------|
| **CNN** | ResNet50 backbone with custom head | ~92% | 25M |
| **ViT** | Vision Transformer Base/16 | ~94% | 86M |
| **ViT+SSL** | ViT with MAE self-supervised pretraining | ~96% | 86M |
| **Hybrid** | ResNet50 features + Transformer encoder | ~95% | 45M |

### Model Architecture

**Vision Transformer (ViT-Base/16):**
| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 |
| Patch Size | 16×16 |
| Embed Dim | 128 |
| Attention Heads | 4 |
| Encoder Layers | 4 |
| MLP Dim | 256 |

**Hybrid CNN+ViT:**
| Parameter | Value |
|-----------|-------|
| CNN Backbone | ResNet50 |
| Embed Dim | 128 |
| Attention Heads | 4 |
| Transformer Layers | 2 |

---

## 🎥 Smart Scanner

Real-time document scanning with quality assessment:

- **Blur Detection** — Laplacian variance analysis
- **Glare Detection** — Overexposure detection
- **Edge Detection** — Document boundary finding
- **Auto-Capture** — Automatically captures when quality passes
- **Flip Camera** — Switch between front/back camera

Access at: `http://localhost:3000/predict` → Smart Scanner tab

---

## 🔐 Authentication

JWT-based authentication with:
- User registration with email validation
- Secure password hashing (bcrypt)
- Access token & refresh token
- Rate limiting (200 requests/minute)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Create new account |
| `/api/auth/login` | POST | Get access token |
| `/api/auth/me` | GET | Get current user |
| `/api/predict/single` | POST | Single image prediction |
| `/api/predict/batch` | POST | Batch prediction |
| `/api/analytics/stats` | GET | Get prediction statistics |
| `/api/analytics/history` | GET | Get prediction history |

---

## 📊 Features

### Dashboard
- Total scans counter
- Fraud detection rate
- Model usage breakdown
- Risk level distribution
- Recent predictions timeline

### Prediction
- **File Upload** — Drag & drop or browse
- **Smart Scanner** — Camera-based scanning
- **Model Selection** — Choose between CNN, ViT, ViT+SSL, Hybrid
- **Results** — Class prediction, confidence scores, risk assessment

### History
- View all past predictions
- Filter by date, model, result
- Re-analyze documents
- Export reports

---

## 🧠 Explainability (XAI)

### Attention Rollout
Visualizes which image regions the ViT model attends to across all transformer layers.

### Grad-CAM
Gradient-weighted Class Activation Mapping highlights discriminative regions.

### SHAP Analysis
Shapley values show per-pixel feature importance.

### PDF Reports
Generate audit-ready reports with:
- Document image
- Prediction & confidence
- Attention heatmaps
- Risk assessment
- Decision traceability

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Services:
# - api: FastAPI backend (port 8000)
# - frontend: Next.js app (port 3000)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📚 References

1. Dosovitskiy et al. (2020) — *"An Image is Worth 16x16 Words"*
2. He et al. (2022) — *"Masked Autoencoders Are Scalable Vision Learners"*
3. Chen et al. (2020) — *"SimCLR: A Simple Framework for Contrastive Learning"*
4. Selvaraju et al. (2017) — *"Grad-CAM: Visual Explanations from Deep Networks"*

---

## 📄 License

Academic and research purposes — Banking & Finance AI/ML Capstone Project.

---

**Tech Stack:** Python 3.11 | PyTorch | FastAPI | Next.js 14 | TypeScript | Tailwind CSS  
**Year:** 2026
