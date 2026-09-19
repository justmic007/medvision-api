"""Database engine and session setup (Phase 6).

Reads DATABASE_URL from the environment (.env locally, or the container's
env in docker-compose). Provides a SQLAlchemy 2.0 engine, a session factory,
and a FastAPI dependency (get_db) that yields a session per request.

This is the app database (Postgres) for users/patients/cases/audit — separate
from MLflow's tracking store (which stays file-based in ./mlruns).
"""
from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Load .env if present (no-op in containers where env is already set).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://medvision:medvision_dev@localhost:5432/medvision",
)

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base all ORM models inherit from."""


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yield a session, always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
