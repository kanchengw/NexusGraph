"""Generate run_id and insert/update PipelineSession records."""
from __future__ import annotations
import json
from datetime import datetime, UTC
from typing import Any
from sqlmodel import Session, select, desc
from app.core.config import settings
from app.models.database import get_engine
from app.models.pipeline_session import PipelineSession


def _get_db_session():
    return Session(get_engine())


def _next_run_id() -> str:
    """Generate run_id like fw_20260701_001."""
    prefix = datetime.now(UTC).strftime("fw_%Y%m%d_")
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(PipelineSession).where(PipelineSession.run_id.like(f"{prefix}%")).order_by(desc(PipelineSession.id)).limit(1)
        last = session.exec(stmt).first()
        if last and last.run_id:
            seq = int(last.run_id.split("_")[-1]) + 1
        else:
            seq = 1
    return f"{prefix}{seq:03d}"


def _to_json(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _save(s: PipelineSession) -> None:
    with _get_db_session() as session:
        session.add(s)
        session.commit()
        session.refresh(s)


def create_session(run_id: str | None = None) -> PipelineSession:
    s = PipelineSession(
        run_id=run_id or _next_run_id(),
        status="created",
    )
    _save(s)
    return s


def update_session(session_id: int, **kwargs) -> PipelineSession:
    with _get_db_session() as db:
        s = db.get(PipelineSession, session_id)
        if not s:
            raise ValueError(f"PipelineSession {session_id} not found")
        for k, v in kwargs.items():
            if v is not None:
                setattr(s, k, v)
        s.updated_at = datetime.now(UTC)
        db.add(s)
        db.commit()
        db.refresh(s)
    return s
