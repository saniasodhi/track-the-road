"""SQLite + SQLAlchemy setup.

The database is a single file on disk (backend/data/tracksense.db). There is no
server to start and nothing to configure - if the file is missing it is created
on first run.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# backend/app/db.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TRACKSENSE_DATA_DIR", BACKEND_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLES_DIR = Path(os.getenv("TRACKSENSE_SAMPLES_DIR", DATA_DIR / "samples"))
REAL_SAMPLES_DIR = DATA_DIR / "samples_real"
CONFIG_DIR = BACKEND_DIR / "config"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.getenv("TRACKSENSE_DB_PATH", DATA_DIR / "tracksense.db"))
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    # FastAPI serves requests from a thread pool, so the connection has to be
    # usable from more than the thread that created it.
    connect_args={"check_same_thread": False},
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: one database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they do not exist yet."""
    from . import models  # noqa: F401  (import registers the tables on Base)

    Base.metadata.create_all(bind=engine)


def db_status() -> dict:
    """Real health information about the database, for GET /api/health."""
    info = {"path": str(DB_PATH), "exists": DB_PATH.exists(), "writable": False, "error": None}
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        probe = DATA_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        info["writable"] = True
    except Exception as exc:  # pragma: no cover - defensive
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info
