"""PipelineSession model for tracking flywheel iteration lifecycle."""

from datetime import datetime, UTC
from sqlmodel import Field, SQLModel


class PipelineSession(SQLModel, table=True):
    """Tracks one complete flywheel iteration (Eval -> Optimize -> Apply -> Verify -> Rollback)."""

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)               # "fw_20260701_001"
    status: str = Field(default="created")         # created / eval_done / optimizing / waiting_approval / backing_up / applying / verifying / waiting_verdict / passed / discarded / rolled_back / failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ── Eval ──
    eval_full_result: str | None = Field(default=None)       # JSON: per-sample + aggregate
    eval_metrics_summary: str | None = Field(default=None)    # JSON: aggregate only

    # ── Optimize ──
    opt_suggestion_raw: str | None = Field(default=None)      # JSON: LLM raw suggestions
    opt_suggestion_final: str | None = Field(default=None)    # JSON: human-modified suggestions
    opt_human_notes: str | None = Field(default=None)
    opt_approved_at: datetime | None = Field(default=None)

    # ── Apply ──
    apply_backup_path: str | None = Field(default=None)
    apply_changes: str | None = Field(default=None)           # JSON: applied param changes
    apply_applied_at: datetime | None = Field(default=None)

    # ── Verify ──
    verify_result: str | None = Field(default=None)           # JSON: verification report
    verify_verdict: str | None = Field(default=None)          # passed / degraded / failed

    # ── Rollback ──
    rollback_reason: str | None = Field(default=None)
    rollback_report: str | None = Field(default=None)         # JSON: before/after comparison
    rollback_at: datetime | None = Field(default=None)
