"""
Database Configuration
======================
SQLAlchemy setup with SQLite (easily switch to PostgreSQL for production).
"""

import os
import subprocess
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
    """Initialize database tables with configurable migration strategy."""
    init_strategy = os.getenv("DB_INIT_STRATEGY", "create_all").strip().lower()

    if init_strategy == "migrate":
        try:
            subprocess.run(["alembic", "upgrade", "head"], check=True)
            print("Database migrated successfully (alembic upgrade head).")
            return
        except Exception as exc:
            print(f"Migration failed, falling back to create_all: {exc}")

    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully (create_all).")
