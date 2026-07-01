"""Data Flywheel - automated Judge -> Optimizer -> Index -> Verify workflow.

Continuous loop that:
  1. JUDGE: Polls PG for conversation count >= threshold, runs evaluation
  2. OPTIMIZER: Reads new eval results, runs LLM analysis, waits for HIL approval
  3. INDEX: On approval, creates Neo4j backup, rebuilds KB with new parameters
  4. VERIFY: Runs evaluation again, compares before/after metrics
     5a. PASSED: Cleanup backup, cycle complete
     5b. DEGRADED: Restore Neo4j from backup, revert params, inject into optimization memory

Usage:
    python -c "from app.core.data_flywheel import run_flywheel; import asyncio; asyncio.run(run_flywheel())"
"""
from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

from app.core.judge_agent.evaluator import run_evaluation
from app.core.optimizer_agent.analyzer import generate_all_reports, print_report_summary
from app.core.optimizer_agent.optimizer import run_llm_analysis, save_suggestion, print_report, apply_suggestion
from app.core.optimizer_agent.backup import create_backup, restore_backup, cleanup_backup
from app.core.optimizer_agent.verification import generate_verification_report, read_verdict, check_verdict
from app.core.graphrag.indexer import run_indexing
from app.core.logging import logger
from app.core.config import settings
from app.core.offline_admin.pipeline_writer import create_session, update_session, _to_json

POLL_INTERVAL = 60
CONVERSATION_THRESHOLD = 200
SUGGESTION_PATH = "evals/reports/optimization_suggestion.json"
EVAL_BEFORE_PATH = "evals/results/graphrag_eval_before.json"
EVAL_AFTER_PATH = "evals/results/graphrag_eval_after.json"
EVAL_NUM_SAMPLES = 100
VERDICT_PATH = "evals/reports/verification_report.json"


async def check_conversation_count() -> int:
    try:
        from sqlmodel import Session, create_engine, text
        db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        engine = create_engine(db_url, pool_pre_ping=True)
        with Session(engine) as session:
            last_eval = session.execute(text("SELECT MAX(created_at) FROM evalresult")).scalar()
            if last_eval:
                result = session.execute(text("SELECT COUNT(*) FROM retrievalmetric WHERE timestamp > :ts"), {"ts": last_eval})
            else:
                result = session.execute(text("SELECT COUNT(*) FROM retrievalmetric"))
            return result.scalar() or 0
    except Exception as e:
        logger.warning("conversation_count_failed", error=str(e))
        return 0


def check_approval() -> bool:
    if not os.path.exists(SUGGESTION_PATH):
        return False
    try:
        with open(SUGGESTION_PATH, "r", encoding="utf-8") as f:
            sug = json.load(f)
        return sug.get("approved", False)
    except (json.JSONDecodeError, FileNotFoundError):
        return False


async def run_flywheel(
    continuous: bool = True,
    once: bool = False,
    force: bool = False,
    progress_callback: Callable | None = None,
):
    """Run the data flywheel with optional real-time progress reporting.
    
    Args:
        continuous: Keep polling for new conversations
        once: Run one cycle then exit
        force: Force trigger regardless of threshold
        progress_callback: Async callable(event_type: str, data: dict)
    """
    logger.info("flywheel_started", continuous=continuous)

    async def emit(evt: str, data: dict[str, Any]):
        if progress_callback:
            await progress_callback(evt, data)

    while True:
        try:
            conv_count = await check_conversation_count()
            logger.info("flywheel_step_judge", conv_count=conv_count, threshold=CONVERSATION_THRESHOLD)

            if force or conv_count >= CONVERSATION_THRESHOLD:
                await emit("step", {"step": "eval", "message": "Running evaluation before optimization..."})

                # --- STEP 1: JUDGE (before index) ---
                pipeline = create_session()
                logger.info("flywheel_triggering_eval", conv_count=conv_count, run_id=pipeline.run_id)
                eval_before = await run_evaluation(split="test", num_samples=EVAL_NUM_SAMPLES, output_path=EVAL_BEFORE_PATH)
                pipeline = update_session(pipeline.id,
                    status="eval_done",
                    eval_full_result=_to_json(eval_before),
                    eval_metrics_summary=_to_json(eval_before.get("metrics", {})),
                )
                metric = eval_before.get("metrics", {}).get("faithfulness", 0)
                logger.info("flywheel_eval_before_complete", faithfulness=metric)
                await emit("step", {"step": "eval_done", "message": f"Evaluation complete — faithfulness: {metric:.3f}", "faithfulness": metric})

                # --- STEP 2: OPTIMIZER ---
                await emit("step", {"step": "optimizing", "message": "Running LLM analysis on eval results..."})
                logger.info("flywheel_step_optimizer")
                report = await generate_all_reports(days=7)
                suggestion = run_llm_analysis(report)
                save_suggestion(suggestion)
                print_report(suggestion)
                logger.info("flywheel_waiting_approval", path=SUGGESTION_PATH)
                await emit("step", {"step": "waiting_hil_a", "message": "⏸ Waiting for HIL A — human approval of optimization suggestions", "suggestion_count": len(suggestion.get("suggestions", []))})

                approval_count = 0
                while not check_approval():
                    approval_count += 1
                    if approval_count % 10 == 0:
                        logger.info("flywheel_still_waiting_approval", suggestion_path=SUGGESTION_PATH)
                    await asyncio.sleep(30)

                # --- STEP 3: BACKUP + INDEX ---
                await emit("step", {"step": "backing_up", "message": "Creating Neo4j backup before applying changes..."})
                logger.info("flywheel_step_index")
                suggestion = json.load(open(SUGGESTION_PATH, "r", encoding="utf-8"))
                pipeline = update_session(pipeline.id,
                    status="optimizing",
                    opt_suggestion_raw=_to_json(suggestion),
                )
                changes = {}
                for s in suggestion.get("suggestions", []):
                    p = s.get("parameter")
                    v = s.get("proposed_value")
                    if p and v is not None:
                        changes[p] = v

                if changes:
                    pipeline = update_session(pipeline.id,
                        status="backing_up",
                        apply_changes=_to_json(changes),
                    )
                    logger.info("flywheel_backup_before_index", params=changes)
                    backup_path = await create_backup(tag="pre_index")
                    logger.info("flywheel_backup_created", path=backup_path)
                    await emit("step", {"step": "backup_done", "message": f"Backup created: {backup_path}", "backup_path": backup_path})
                    pipeline = update_session(pipeline.id, apply_backup_path=backup_path)

                    # Index with progress
                    await emit("step", {"step": "indexing", "message": "Rebuilding knowledge base..."})
                    # Indexer reports (doc_index, total_docs) — wrap into generic callback
                    async def _index_progress(doc: int, total: int):
                        pct = round(doc / total * 100, 1)
                        await emit("progress", {"step": "indexing", "message": f"Indexing: {doc}/{total} docs ({pct}%)", "progress": doc, "total": total})
                    await run_indexing(split="train", reset=True, skip_entities=False, progress_callback=_index_progress)
                    logger.info("flywheel_reindex_complete")
                    await emit("step", {"step": "index_done", "message": "Knowledge base rebuilt"})
                    pipeline = update_session(pipeline.id, status="applying")

                    # --- STEP 4: VERIFY (after index) ---
                    await emit("step", {"step": "verifying", "message": "Running verification evaluation..."})
                    logger.info("flywheel_step_verify")
                    eval_after = await run_evaluation(split="test", num_samples=EVAL_NUM_SAMPLES, output_path=EVAL_AFTER_PATH)
                    metric_after = eval_after.get("metrics", {}).get("faithfulness", 0)
                    logger.info("flywheel_eval_after_complete", faithfulness=metric_after)

                    generate_verification_report(
                        before_path=EVAL_BEFORE_PATH,
                        after_path=EVAL_AFTER_PATH,
                        bad_case_count=0,
                        good_case_count=0,
                    )
                    pipeline = update_session(pipeline.id, status="verifying", verify_result=_to_json(
                        json.load(open(VERDICT_PATH, "r", encoding="utf-8")) if os.path.exists(VERDICT_PATH) else {}
                    ))
                    logger.info("flywheel_waiting_hil_b", path=VERDICT_PATH)
                    await emit("step", {"step": "waiting_hil_b", "message": f"⏸ Waiting for HIL B — verdict: faithfulness before={eval_before.get('metrics',{}).get('faithfulness',0):.3f} after={metric_after:.3f}", "before": eval_before.get("metrics",{}).get("faithfulness",0), "after": metric_after})

                    while True:
                        verdict = read_verdict(VERDICT_PATH)
                        if verdict == "passed":
                            logger.info("flywheel_hil_b_passed")
                            await emit("step", {"step": "cleaning_up", "message": "Verification passed — cleaning up backup..."})
                            await cleanup_backup(backup_path)
                            logger.info("flywheel_backup_cleaned", path=backup_path)
                            break
                        elif verdict == "degraded":
                            logger.warning("flywheel_hil_b_degraded")
                            pipeline = update_session(pipeline.id, status="rolled_back", verify_verdict="degraded",
                                rollback_reason="Verification degraded: faithfulness " + str(round(metric, 3)) + " -> " + str(round(metric_after, 3)),
                            )
                            await emit("step", {"step": "rolling_back", "message": "Verification degraded — restoring Neo4j from backup..."})
                            await restore_backup(backup_path)
                            logger.info("flywheel_rollback_complete", backup=backup_path)
                            break
                        await asyncio.sleep(10)

                    if os.path.exists(VERDICT_PATH):
                        os.remove(VERDICT_PATH)

                pipeline = update_session(pipeline.id, status="done")
                os.remove(SUGGESTION_PATH) if os.path.exists(SUGGESTION_PATH) else None
                await emit("step", {"step": "done", "message": "✅ Flywheel cycle complete"})
                logger.info("flywheel_cycle_complete")

            if once:
                break
            await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            try:
                pipeline = update_session(pipeline.id, status="failed")
            except Exception:
                pass
            logger.exception("flywheel_cycle_failed", error=str(e))
            await emit("step", {"step": "failed", "message": f"❌ Flywheel failed: {e}"})
            if once:
                raise
            await asyncio.sleep(POLL_INTERVAL)
