import sys
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from enem_pipeline.config import settings

# Inject Windows native truststore certificates into SSL if on Windows
if sys.platform == "win32":
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

# Adjust connect_args for SQLite
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


@contextmanager
def get_db() -> Generator:
    """Context manager for SQLAlchemy database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Initializes database schema and tables."""
    from enem_pipeline.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
