"""Offline Admin Router — delegates to existing optimizer/judge/index modules."""

from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, UTC
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select, desc

from app.core.logging import logger
from app.models.database import get_engine
from app.models.pipeline_session import PipelineSession
from app.core.offline_admin.events import subscribe, unsubscribe, emit as event_emit, sse_format

# ── paths (consistent with existing modules) ──
SUGGESTION_PATH = "evals/reports/optimization_suggestion.json"
VERIFICATION_PATH = "evals/reports/verification_report.json"

router = APIRouter(prefix="/api/admin", tags=["admin"])

_flywheel_task: asyncio.Task | None = None


# ── Helpers ──

def _read_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── SSE Events ──

@router.get("/events")
async def event_stream(request: Request):
    """SSE endpoint: push real-time flywheel progress to connected clients."""
    q = subscribe()
    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=10)
                    yield sse_format(event)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            unsubscribe(q)
    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Sessions ──

@router.get("/sessions")
def list_sessions(limit: int = 20, offset: int = 0):
    """List flywheel pipeline sessions."""
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(PipelineSession).order_by(desc(PipelineSession.created_at)).offset(offset).limit(limit)
        results = session.exec(stmt).all()
        return {"sessions": [s.model_dump(mode="json") for s in results]}


@router.get("/sessions/{session_id}")
def get_session(session_id: int):
    """Get single pipeline session detail."""
    engine = get_engine()
    with Session(engine) as session:
        s = session.get(PipelineSession, session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        return s.model_dump(mode="json")


# ── Flywheel ──

@router.get("/flywheel/status")
async def flywheel_status():
    """Check current flywheel state and conversation count."""
    global _flywheel_task
    sug = _read_json(SUGGESTION_PATH)
    ver = _read_json(VERIFICATION_PATH)
    if _flywheel_task and not _flywheel_task.done():
        status = "running"
    elif sug and sug.get("approved"):
        status = "approved_pending_apply"
    elif sug and not sug.get("approved"):
        status = "waiting_hil_a"
    elif ver and ver.get("status") == "reviewed":
        status = "hil_b_verdict_recorded"
    elif ver and ver.get("status") != "reviewed":
        status = "waiting_hil_b"
    else:
        status = "idle"
    return {
        "status": status,
        "suggestion_pending": sug is not None,
        "verification_pending": ver is not None,
    }


@router.post("/flywheel/trigger")
async def trigger_flywheel(payload: dict = {}):
    """Manually trigger one flywheel cycle with real-time SSE progress.
    
    Body:
        force (bool): Skip conversation count threshold check
        once (bool): Run one cycle then exit (default true)
    """
    global _flywheel_task
    if _flywheel_task and not _flywheel_task.done():
        raise HTTPException(status_code=409, detail="Flywheel already running")
    try:
        from app.core.data_flywheel import run_flywheel

        async def progress_callback(evt: str, data: dict[str, Any]):
            await event_emit({"type": evt, "data": data})

        force = payload.get("force", True)  # Default force=True for UI trigger
        once = payload.get("once", True)
        _flywheel_task = asyncio.create_task(
            run_flywheel(continuous=False, once=once, force=force, progress_callback=progress_callback)
        )
        return {"status": "triggered"}
    except Exception as e:
        logger.exception("flywheel_trigger_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Eval Results ──

@router.get("/eval-results")
def list_eval_results(limit: int = 20, offset: int = 0):
    """List evaluation results from DB."""
    from app.models.eval_result import EvalResult
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(EvalResult).order_by(desc(EvalResult.created_at)).offset(offset).limit(limit)
        results = session.exec(stmt).all()
        return {"results": [r.model_dump(mode="json") for r in results]}


@router.get("/eval-results/{eval_id}")
def get_eval_result(eval_id: int):
    """Get single evaluation result detail."""
    from app.models.eval_result import EvalResult
    engine = get_engine()
    with Session(engine) as session:
        r = session.get(EvalResult, eval_id)
        if not r:
            raise HTTPException(status_code=404, detail="Eval result not found")
        return r.model_dump(mode="json")


# ── Optimizations (HIL A) ──

@router.get("/optimizations")
def list_optimizations():
    """List optimization suggestions."""
    sug = _read_json(SUGGESTION_PATH)
    return {"optimizations": [sug] if sug else []}


@router.post("/optimizations/approve")
def approve_optimization():
    """HIL A: Approve current optimization suggestion."""
    sug = _read_json(SUGGESTION_PATH)
    if not sug:
        raise HTTPException(status_code=404, detail="No pending optimization")
    from app.core.optimizer_agent.optimizer import approve
    approve()
    return {"status": "approved"}


@router.post("/optimizations/reject")
def reject_optimization():
    """HIL A: Reject and discard current optimization."""
    sug = _read_json(SUGGESTION_PATH)
    if not sug:
        raise HTTPException(status_code=404, detail="No pending optimization")
    sug["approved"] = False
    sug["rejected"] = True
    sug["rejected_at"] = datetime.now(UTC).isoformat()
    _write_json(SUGGESTION_PATH, sug)
    logger.info("optimization_rejected_via_ui")
    return {"status": "rejected"}


@router.post("/optimizations/modify")
def modify_optimization(payload: dict):
    """HIL A: Modify and approve suggestion."""
    sug = _read_json(SUGGESTION_PATH)
    if not sug:
        raise HTTPException(status_code=404, detail="No pending optimization")
    if "suggestions" in payload:
        sug["suggestions"] = payload["suggestions"]
    if "notes" in payload:
        sug["human_notes"] = payload["notes"]
    sug["approved"] = True
    sug["human_modified"] = True
    sug["approved_at"] = datetime.now(UTC).isoformat()
    _write_json(SUGGESTION_PATH, sug)
    logger.info("optimization_modified_and_approved_via_ui")
    return {"status": "approved_with_modifications"}


# ── Verify (HIL B) ──

@router.post("/verify/run")
async def run_verification(before_path: str = "evals/results/graphrag_eval_before.json", after_path: str = "evals/results/graphrag_eval_after.json"):
    """Trigger verification using existing verification module."""
    try:
        from app.core.optimizer_agent.verification import generate_verification_report
        await event_emit({"type": "step", "data": {"step": "verifying", "message": "Generating verification report..."}})
        report = generate_verification_report(before_path=before_path, after_path=after_path)
        await event_emit({"type": "step", "data": {"step": "verify_done", "message": "Verification report ready"}})
        return {"status": "verification_complete", "report_path": VERIFICATION_PATH}
    except Exception as e:
        logger.exception("verify_run_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/verify/result")
def get_verification_result():
    """Get current verification report."""
    report = _read_json(VERIFICATION_PATH)
    if not report:
        raise HTTPException(status_code=404, detail="No verification report found")
    return report


@router.post("/verify/verdict")
def record_verdict(payload: dict):
    """HIL B: Record human verdict on verification."""
    verdict = payload.get("verdict")
    if verdict not in ("passed", "degraded", "failed"):
        raise HTTPException(status_code=400, detail="verdict must be passed/degraded/failed")
    from app.core.optimizer_agent.verification import check_verdict
    ok = check_verdict(verdict, VERIFICATION_PATH)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to record verdict")
    return {"status": "verdict_recorded", "verdict": verdict}


# ── Rollback ──

@router.post("/rollback")
async def execute_rollback(payload: dict = {}):
    """Trigger rollback via existing backup module."""
    try:
        from app.core.optimizer_agent.backup import restore_backup
        backup_path = payload.get("backup_path", "evals/backups")
        await event_emit({"type": "step", "data": {"step": "rolling_back", "message": f"Restoring from {backup_path}..."}})
        await restore_backup(backup_path)
        await event_emit({"type": "step", "data": {"step": "rollback_done", "message": "Rollback complete"}})
        logger.info("rollback_executed_via_ui", backup=backup_path)
        return {"status": "rollback_complete"}
    except Exception as e:
        logger.exception("rollback_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rollback-history")
def rollback_history():
    """Read rollback history from pipeline sessions."""
    engine = get_engine()
    with Session(engine) as session:
        stmt = select(PipelineSession).where(PipelineSession.status == "rolled_back").order_by(desc(PipelineSession.rollback_at))
        results = session.exec(stmt).all()
        return {"rollbacks": [s.model_dump(mode="json") for s in results]}
