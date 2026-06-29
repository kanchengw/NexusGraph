"""EvalResult model for tracking evaluation history."""

from datetime import datetime, UTC
from sqlmodel import Field, SQLModel


class EvalResult(SQLModel, table=True):
    """Offline evaluation result record."""

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    trace_id: str = Field(default="", index=True)
    faithfulness: float = Field(default=0.0)
    relevance: float = Field(default=0.0)
    context_precision: float = Field(default=0.0)
    top_k: int = Field(default=5)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    judge_model: str = Field(default="qwen-plus")
    num_samples: int = Field(default=0)
    split: str = Field(default="test")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
