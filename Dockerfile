# Bank Fraud Detection - Docker Container
# ========================================
# Build:
#   docker build -t bank-fraud-detection:latest .
#
# Run API:
#   docker run -p 8000:8000 bank-fraud-detection:latest
#
# Run with GPU:
#   docker run --gpus all -p 8000:8000 bank-fraud-detection:latest

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints results logs model_registry deployment/onnx_models

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run FastAPI server
CMD ["uvicorn", "deployment.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
