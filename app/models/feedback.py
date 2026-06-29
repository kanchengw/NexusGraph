"""Feedback model for data flywheel."""

from datetime import datetime, UTC
from sqlmodel import Field, SQLModel


class Feedback(SQLModel, table=True):
    """User feedback on a GraphRAG response."""

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    query: str
    response: str
    rating: int = Field(ge=0, le=5)
    correction: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
