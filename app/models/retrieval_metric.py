"""RetrievalMetric model for per-query online retrieval metrics."""
from datetime import datetime, UTC
from sqlmodel import Field, SQLModel


class RetrievalMetric(SQLModel, table=True):
    """Per-query 3-path retrieval metric record."""

    id: int | None = Field(default=None, primary_key=True)
    trace_id: str = Field(default="", index=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    query: str = Field(default="")
    answer: str = Field(default="")
    vector_count: int = Field(default=0)
    bm25_count: int = Field(default=0)
    graph_count: int = Field(default=0)
    unique_chunks: int = Field(default=0)
    vector_only: int = Field(default=0)
    bm25_only: int = Field(default=0)
    graph_only: int = Field(default=0)
    overlap_vector_bm25: int = Field(default=0)
    response_time_ms: int = Field(default=0)
