"""
Authentication Module
=====================
JWT-based authentication with password hashing.
"""

import os
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, cast
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from .database import get_db
from .models import User, APIKey

logger = logging.getLogger(__name__)

# Environment mode
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT in {"prod", "production"}

# Feature flag to disable authentication entirely
AUTH_DISABLED = os.getenv("DISABLE_AUTH", "false").lower() in {"1", "true", "yes", "on"}
PUBLIC_USER_EMAIL = os.getenv("PUBLIC_USER_EMAIL", "public@example.local")
PUBLIC_USER_USERNAME = os.getenv("PUBLIC_USER_USERNAME", "public_user")
PUBLIC_USER_FULL_NAME = os.getenv("PUBLIC_USER_FULL_NAME", "Public Analyst")

if IS_PRODUCTION and AUTH_DISABLED:
    raise RuntimeError(
        "DISABLE_AUTH is enabled in production. This is unsafe and not allowed. "
        "Unset DISABLE_AUTH or set it to false."
    )

# Configuration
_env_secret = os.getenv("SECRET_KEY")
if not _env_secret:
    if IS_PRODUCTION:
        raise RuntimeError(
            "SECRET_KEY is required in production. "
            "Set SECRET_KEY to a strong random value (>=32 chars)."
        )
    SECRET_KEY = secrets.token_urlsafe(32)
    logger.warning(
        "SECRET_KEY not set; generated an ephemeral development key. "
        "All JWT tokens will be invalidated on restart. "
        "Set SECRET_KEY in environment before deployment."
    )
else:
    SECRET_KEY = _env_secret
    if IS_PRODUCTION and len(SECRET_KEY) < 32:
        raise RuntimeError(
            "SECRET_KEY is too short for production. "
            "Use at least 32 characters."
        )
    if not IS_PRODUCTION and len(SECRET_KEY) < 32:
        logger.warning(
            "SECRET_KEY is shorter than recommended for production (32+ chars)."
        )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing - use sha256_crypt as fallback for compatibility
pwd_context = CryptContext(schemes=["sha256_crypt", "bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# =============================================================================
# Pydantic Schemas
# =============================================================================

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiJ9...",
                    "refresh_token": "eyJhbGciOiJIUzI1NiJ9...",
                    "token_type": "bearer",
                    "expires_in": 1440
                }
            ]
        }


class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "email": "analyst@bank.com",
                    "username": "fraud_analyst",
                    "password": "SecureP@ss123",
                    "full_name": "Jane Analyst"
                }
            ]
        }


class UserLogin(BaseModel):
    email: str
    password: str

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "email": "analyst@bank.com",
                    "password": "SecureP@ss123"
                }
            ]
        }


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


# =============================================================================
# Password Functions
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# =============================================================================
# Token Functions
# =============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        # Use options to skip subject validation since we use integer IDs
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM],
            options={"verify_sub": False}
        )
        user_id = payload.get("sub")
        if user_id is not None:
            user_id = int(user_id)  # Ensure it's an int
        email: str = payload.get("email")
        if user_id is None:
            return None
        return TokenData(user_id=user_id, email=email)
    except JWTError:
        return None


# =============================================================================
# User Functions
# =============================================================================

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def _ensure_public_user(db: Session) -> User:
    """Create or return a shared public user for auth-disabled mode."""
    user = get_user_by_email(db, PUBLIC_USER_EMAIL)
    if user:
        return user

    auth_disabled_password = f"auth_disabled::{PUBLIC_USER_EMAIL}"
    public_user = User(
        email=PUBLIC_USER_EMAIL,
        username=PUBLIC_USER_USERNAME,
        full_name=PUBLIC_USER_FULL_NAME,
        hashed_password=get_password_hash(auth_disabled_password),
        role="admin",
        is_active=True,
    )
    db.add(public_user)
    db.commit()
    db.refresh(public_user)
    return public_user


def create_user(db: Session, user: UserCreate) -> User:
    """Create a new user."""
    # Check if email exists
    if get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username exists
    if get_user_by_username(db, user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=get_password_hash(user.password),
        full_name=user.full_name,
        role="user"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user with email and password."""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, cast(str, user.hashed_password)):
        return None
    return user


# =============================================================================
# Dependencies
# =============================================================================

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user from JWT token or API key (or shared public user)."""

    if AUTH_DISABLED:
        return _ensure_public_user(db)
    
    # Try API key first
    if api_key:
        api_key_obj = db.query(APIKey).filter(
            APIKey.key == api_key,
            APIKey.is_active == True
        ).first()
        if api_key_obj:
            # Update last used
            setattr(api_key_obj, 'last_used_at', datetime.utcnow())
            db.commit()
            return get_user_by_id(db, cast(int, api_key_obj.user_id))
    
    # Try JWT token
    if token:
        token_data = decode_token(token)
        if token_data and token_data.user_id:
            return get_user_by_id(db, token_data.user_id)
    
    return None


async def get_current_active_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """Get current active user (required auth)."""
    if AUTH_DISABLED:
        return cast(User, current_user)
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not cast(bool, current_user.is_active):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current user with admin privileges."""
    if AUTH_DISABLED:
        return current_user
    if cast(str, current_user.role) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


async def get_optional_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    if AUTH_DISABLED:
        return current_user
    return current_user


# =============================================================================
# API Key Functions
# =============================================================================

def generate_api_key() -> str:
    """Generate a new API key."""
    return secrets.token_urlsafe(32)


def create_api_key(db: Session, user_id: int, name: str, expires_days: Optional[int] = None) -> APIKey:
    """Create a new API key for a user."""
    key = generate_api_key()
    expires_at = None
    if expires_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_days)
    
    api_key = APIKey(
        user_id=user_id,
        key=key,
        name=name,
        expires_at=expires_at
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    return api_key
