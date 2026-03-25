# Rate limiting setup for FastAPI using slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

# Create a Limiter instance (e.g., 5 requests per minute per IP)
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])

def register_rate_limit(app: FastAPI):
    app.state.limiter = limiter
    # Do not add explicit exception handler here; handled in main.py or by slowapi
    @app.middleware("http")
    async def add_rate_limit_headers(request: Request, call_next):
        response = await call_next(request)
        # Optionally add custom headers here
        return response
