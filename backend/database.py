"""
Database Configuration
======================
SQLAlchemy setup with SQLite (easily switch to PostgreSQL for production).
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database URL - SQLite for development, PostgreSQL for production
def _resolve_database_url() -> str:
    env = os.getenv("ENVIRONMENT", "development").lower()
    db_url = os.getenv("DATABASE_URL", "").strip()

    if not db_url:
        database_dir = Path(__file__).parent.parent / "data"
        database_dir.mkdir(exist_ok=True)
        db_url = f"sqlite:///{database_dir}/fraud_detection.db"

    if env == "production" and db_url.startswith("sqlite"):
        raise ValueError("DATABASE_URL must be set to a production database in production mode.")

    return db_url


DATABASE_URL = _resolve_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")
