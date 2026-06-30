"""LangfuseSnapshot model for per-path LLM metrics in Grafana."""

from datetime import datetime, UTC
from sqlmodel import Field, SQLModel


class LangfuseSnapshot(SQLModel, table=True):
    """Periodic snapshot of Langfuse LLM metrics for Grafana dashboards."""

    id: int | None = Field(default=None, primary_key=True)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
    vector_search_avg_ms: float = Field(default=0.0)
    vector_search_count: int = Field(default=0)
    bm25_search_avg_ms: float = Field(default=0.0)
    bm25_search_count: int = Field(default=0)
    graph_expand_avg_ms: float = Field(default=0.0)
    graph_expand_count: int = Field(default=0)
    llm_inference_avg_ms: float = Field(default=0.0)
    llm_span_count: int = Field(default=0)
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)
