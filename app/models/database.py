"""Database models for the application."""

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings
from app.models.thread import Thread
from app.models.feedback import Feedback
from app.models.eval_result import EvalResult
from app.models.retrieval_metric import RetrievalMetric
from app.models.pipeline_session import PipelineSession

__all__ = ["Thread", "Feedback", "EvalResult", "RetrievalMetric", "PipelineSession"]


def get_engine():
    """Create a database engine (for offline admin / scripts)."""
    url = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    return create_engine(url, pool_pre_ping=True)


def init_db():
    """Create all tables (idempotent)."""
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
