"""
Authentication Routes
====================
API endpoints for user authentication.
"""

import os
from datetime import datetime
from datetime import timedelta
from typing import List, Literal, cast
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from fastapi import Request
from sqlalchemy import func
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from backend.rate_limit import limiter

from .database import get_db
from .models import User, APIKey, AuditLog
from .auth import (
    Token, UserCreate, UserLogin, UserResponse, UserUpdate,
    create_user, authenticate_user, get_current_active_user,
    create_access_token, create_refresh_token, decode_token,
    get_user_by_id, create_api_key, ACCESS_TOKEN_EXPIRE_MINUTES,
    AUTH_COOKIE_NAME, REFRESH_COOKIE_NAME,
    create_password_reset_token, decode_password_reset_token,
    get_password_hash, validate_password_strength,
    IS_PRODUCTION,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

_raw_samesite = os.getenv("AUTH_COOKIE_SAMESITE", "lax").strip().lower()
COOKIE_SAMESITE: Literal['lax', 'strict', 'none'] = (
    cast(Literal['lax', 'strict', 'none'], _raw_samesite)
    if _raw_samesite in {'lax', 'strict', 'none'}
    else 'lax'
)
COOKIE_SECURE = IS_PRODUCTION or os.getenv("AUTH_COOKIE_SECURE", "false").lower() in {"1", "true", "yes", "on"}
REFRESH_COOKIE_MAX_AGE = 60 * 60 * 24 * 7
LOGIN_LOCK_WINDOW_MINUTES = int(os.getenv("LOGIN_LOCK_WINDOW_MINUTES", "15"))
LOGIN_MAX_FAILS = int(os.getenv("LOGIN_MAX_FAILS", "5"))
LOGIN_LOCK_MINUTES = int(os.getenv("LOGIN_LOCK_MINUTES", "15"))
_failed_login_attempts: dict[str, list[datetime]] = {}
_locked_until: dict[str, datetime] = {}


def _key_from_identifier(identifier: str) -> str:
    return identifier.strip().lower()


def _is_locked(identifier: str) -> bool:
    key = _key_from_identifier(identifier)
    lock_until = _locked_until.get(key)
    if not lock_until:
        return False
    if datetime.utcnow() >= lock_until:
        _locked_until.pop(key, None)
        _failed_login_attempts.pop(key, None)
        return False
    return True


def _record_login_failure(identifier: str) -> None:
    key = _key_from_identifier(identifier)
    now = datetime.utcnow()
    window_start = now - timedelta(minutes=LOGIN_LOCK_WINDOW_MINUTES)
    attempts = [ts for ts in _failed_login_attempts.get(key, []) if ts >= window_start]
    attempts.append(now)
    _failed_login_attempts[key] = attempts
    if len(attempts) >= LOGIN_MAX_FAILS:
        _locked_until[key] = now + timedelta(minutes=LOGIN_LOCK_MINUTES)


def _record_login_success(identifier: str) -> None:
    key = _key_from_identifier(identifier)
    _failed_login_attempts.pop(key, None)
    _locked_until.pop(key, None)


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    secure_cookie = COOKIE_SECURE or COOKIE_SAMESITE == "none"
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=secure_cookie,
        samesite=COOKIE_SAMESITE,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=secure_cookie,
        samesite=COOKIE_SAMESITE,
        max_age=REFRESH_COOKIE_MAX_AGE,
        path="/",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
    response.delete_cookie(key=REFRESH_COOKIE_NAME, path="/")


@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    db_user = create_user(db, user)
    
    # Log registration
    log = AuditLog(
        user_id=db_user.id,
        action="user_registered",
        resource_type="user",
        resource_id=db_user.id
    )
    db.add(log)
    db.commit()
    
    return db_user


@router.post("/login", response_model=Token)
@limiter.limit("10/minute")
async def login(request: Request, response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token."""
    if _is_locked(form_data.username):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again later.",
        )

    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        _record_login_failure(form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _record_login_success(form_data.username)
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.id, "email": user.email}
    )
    _set_auth_cookies(response, access_token, refresh_token)
    
    # Log login
    log = AuditLog(
        user_id=user.id,
        action="user_login",
        resource_type="user",
        resource_id=user.id
    )
    db.add(log)
    db.commit()
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/login/json", response_model=Token)
@limiter.limit("10/minute")
async def login_json(request: Request, response: Response, credentials: UserLogin, db: Session = Depends(get_db)):
    """Login with JSON body (alternative to form)."""
    if _is_locked(credentials.email):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again later.",
        )

    user = authenticate_user(db, credentials.email, credentials.password)
    if not user:
        _record_login_failure(credentials.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    _record_login_success(credentials.email)
    
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.id, "email": user.email}
    )
    _set_auth_cookies(response, access_token, refresh_token)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str, response: Response, db: Session = Depends(get_db)):
    """Refresh access token using refresh token."""
    token_data = decode_token(refresh_token)
    if not token_data or not token_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = get_user_by_id(db, token_data.user_id)
    if not user or not cast(bool, user.is_active):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    new_refresh_token = create_refresh_token(
        data={"sub": user.id, "email": user.email}
    )
    _set_auth_cookies(response, access_token, new_refresh_token)
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout")
async def logout(response: Response):
    """Clear auth cookies and end the current session."""
    _clear_auth_cookies(response)
    return {"message": "Logged out"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_active_user)):
    """Get current user profile."""
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_me(
    update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user profile."""
    if update.full_name is not None:
        setattr(current_user, 'full_name', update.full_name)
    if update.email is not None:
        setattr(current_user, 'email', update.email)
    
    setattr(current_user, 'updated_at', datetime.utcnow())
    db.commit()
    db.refresh(current_user)
    return current_user


# =============================================================================
# API Key Management
# =============================================================================

from pydantic import BaseModel


class ForgotPasswordRequest(BaseModel):
    email: str


class ForgotPasswordResponse(BaseModel):
    message: str
    reset_token: str | None = None


class ResetPasswordRequest(BaseModel):
    token: str
    password: str


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
@limiter.limit("5/15minute")
async def forgot_password(request: Request, payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Issue a password reset token (dev mode returns token for testing)."""
    normalized = payload.email.strip().lower()
    user = db.query(User).filter(func.lower(User.email) == normalized).first()
    if not user:
        return ForgotPasswordResponse(message="If that email exists, reset instructions have been sent.")

    reset_token = create_password_reset_token(cast(int, user.id), cast(str, user.email))
    if IS_PRODUCTION:
        return ForgotPasswordResponse(message="If that email exists, reset instructions have been sent.")

    return ForgotPasswordResponse(
        message="Reset token generated for development/testing.",
        reset_token=reset_token,
    )


@router.post("/reset-password")
@limiter.limit("10/hour")
async def reset_password(request: Request, payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Reset password using short-lived reset token."""
    token_data = decode_password_reset_token(payload.token)
    if not token_data or not token_data.user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token")

    if not validate_password_strength(payload.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters and include a letter and a number.",
        )

    user = get_user_by_id(db, token_data.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    setattr(user, 'hashed_password', get_password_hash(payload.password))
    setattr(user, 'updated_at', datetime.utcnow())
    db.commit()
    return {"message": "Password reset successful"}

class APIKeyCreate(BaseModel):
    name: str
    expires_days: int | None = None

class APIKeyResponse(BaseModel):
    id: int
    name: str
    key: str | None = None  # Only shown on creation
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    
    class Config:
        from_attributes = True


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_new_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new API key."""
    api_key = create_api_key(
        db, 
        user_id=cast(int, current_user.id), 
        name=key_data.name,
        expires_days=key_data.expires_days
    )
    return APIKeyResponse.model_validate(api_key)


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all API keys for current user."""
    keys = db.query(APIKey).filter(APIKey.user_id == current_user.id).all()
    return [
        APIKeyResponse.model_validate(k).model_copy(update={'key': None})
        for k in keys
    ]


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete an API key."""
    api_key = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == current_user.id
    ).first()
    
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    db.delete(api_key)
    db.commit()
    return {"message": "API key deleted"}
