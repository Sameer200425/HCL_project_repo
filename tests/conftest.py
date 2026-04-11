"""Pytest session-wide test environment defaults.

Keeps tests isolated from local runtime artifacts and flaky auth limits.
"""

import os


os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unit-tests-only")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DB_INIT_STRATEGY", "create_all")
os.environ.setdefault("AUTH_LOGIN_RATE_LIMIT", "1000/minute")
