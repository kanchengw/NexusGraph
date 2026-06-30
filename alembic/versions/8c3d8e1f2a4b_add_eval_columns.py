"""Add answer_correctness, context_recall to evalresult + langfuse_snapshot

Revision ID: 8c3d8e1f2a4b
Revises: b25d38b0cd7c
Create Date: 2026-06-30 22:30:00.000000
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "8c3d8e1f2a4b"
down_revision: Union[str, None] = "b25d38b0cd7c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("evalresult", sa.Column("answer_correctness", sa.Float(), nullable=False, server_default="0.0"))
    op.add_column("evalresult", sa.Column("context_recall", sa.Float(), nullable=False, server_default="0.0"))

    op.create_table(
        "langfuse_snapshot",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(), nullable=False),
        sa.Column("vector_search_avg_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("vector_search_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("bm25_search_avg_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("bm25_search_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("graph_expand_avg_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("graph_expand_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_inference_avg_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("llm_span_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_cost", sa.Float(), nullable=False, server_default="0.0"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_langfuse_snapshot_fetched_at"), "langfuse_snapshot", ["fetched_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_langfuse_snapshot_fetched_at"), table_name="langfuse_snapshot")
    op.drop_table("langfuse_snapshot")
    op.drop_column("evalresult", "context_recall")
    op.drop_column("evalresult", "answer_correctness")
