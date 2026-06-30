"""Data Flywheel - automated Judge -> Optimizer -> Index workflow.

Continuous loop that:
  1. JUDGE: Polls PG for conversation count (retrievalmetric rows since last eval) >= threshold, runs evaluation
  2. OPTIMIZER: Reads new eval results, runs LLM analysis, waits for HIL approval
  3. INDEX: On approval, rebuilds KB with new parameters

Usage:
    python -c "from app.core.data_flywheel import run_flywheel; import asyncio; asyncio.run(run_flywheel())"
"""
from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from app.core.judge_agent.evaluator import run_evaluation
from app.core.optimizer_agent.analyzer import generate_all_reports, print_report_summary
from app.core.optimizer_agent.optimizer import run_llm_analysis, save_suggestion, print_report, apply_suggestion
from app.core.index_agent.indexer import run_indexing
from app.core.logging import logger
from app.core.config import settings

POLL_INTERVAL = 60  # seconds between PG polls
CONVERSATION_THRESHOLD = 200  # number of queries (retrievalmetric rows) to trigger eval, adjustable
SUGGESTION_PATH = 'evals/reports/optimization_suggestion.json'


async def check_conversation_count() -> int:
    """Count retrievalmetric rows since the last eval run."""
    try:
        from sqlmodel import Session, create_engine, text
        db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        engine = create_engine(db_url, pool_pre_ping=True)
        with Session(engine) as session:
            # Get last eval timestamp
            last_eval = session.execute(text("SELECT MAX(created_at) FROM evalresult")).scalar()
            if last_eval:
                result = session.execute(text("SELECT COUNT(*) FROM retrievalmetric WHERE timestamp > :ts"), {"ts": last_eval})
            else:
                result = session.execute(text("SELECT COUNT(*) FROM retrievalmetric"))
            return result.scalar() or 0
    except Exception as e:
        logger.warning("conversation_count_failed", error=str(e))
        return 0


async def check_eval_results() -> list[dict[str, Any]]:
    """Read latest eval results from PG."""
    try:
        from sqlmodel import Session, create_engine, text
        db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        engine = create_engine(db_url, pool_pre_ping=True)
        with Session(engine) as session:
            result = session.execute(
                text("SELECT run_id, faithfulness, relevance, created_at FROM evalresult ORDER BY created_at DESC LIMIT 5")
            )
            return [dict(r._mapping) for r in result]
    except Exception as e:
        logger.warning("eval_results_fetch_failed", error=str(e))
        return []


def check_approval() -> bool:
    """Check if suggestion has been approved."""
    if not os.path.exists(SUGGESTION_PATH):
        return False
    try:
        with open(SUGGESTION_PATH, "r", encoding="utf-8") as f:
            sug = json.load(f)
        return sug.get('approved', False)
    except (json.JSONDecodeError, FileNotFoundError):
        return False


async def run_flywheel(continuous: bool = True, once: bool = False, force: bool = False):
    """Run the data flywheel: Judge -> Optimizer -> Index.

    Args:
        continuous: If True, loop forever polling PG.
        once: If True, run one full cycle and return.
    """
    logger.info('flywheel_started', continuous=continuous)

    while True:
        try:
            # --- STEP 1: JUDGE ---
            conv_count = await check_conversation_count()
            logger.info('flywheel_step_judge', conv_count=conv_count, threshold=CONVERSATION_THRESHOLD)

            if force or conv_count >= CONVERSATION_THRESHOLD:
                logger.info('flywheel_triggering_eval', conv_count=conv_count)
                eval_result = await run_evaluation(split='test', num_samples=5)
                logger.info('flywheel_eval_complete', faithfulness=eval_result.get('metrics', {}).get('faithfulness'))

                # --- STEP 2: OPTIMIZER ---
                logger.info('flywheel_step_optimizer')
                report = generate_all_reports(days=7)
                suggestion = run_llm_analysis(report)
                save_suggestion(suggestion)
                print_report(suggestion)

                logger.info('flywheel_waiting_approval', path=SUGGESTION_PATH)

                # Wait for HIL approval
                approval_count = 0
                while not check_approval():
                    approval_count += 1
                    if approval_count % 10 == 0:
                        logger.info('flywheel_still_waiting_approval',
                                   suggestion_path=SUGGESTION_PATH,
                                   cmd=f"python -c 'from app.core.optimizer_agent.graph import approve; approve()'")
                    await asyncio.sleep(30)

                # --- STEP 3: INDEX (triggered by Optimizer approval) ---
                logger.info('flywheel_step_index')
                suggestion = json.load(open(SUGGESTION_PATH))
                changes = {}
                for s in suggestion.get('suggestions', []):
                    p = s.get('parameter')
                    v = s.get('proposed_value')
                    if p and v is not None:
                        changes[p] = v

                if changes:
                    logger.info('flywheel_reindexing', params=changes)
                    # Re-index with new params
                    await run_indexing(split='train', reset=True, skip_entities=False)
                    logger.info('flywheel_reindex_complete')

                # Clean up
                os.remove(SUGGESTION_PATH) if os.path.exists(SUGGESTION_PATH) else None
                logger.info('flywheel_cycle_complete')

            if once:
                break

            await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            logger.exception('flywheel_cycle_failed', error=str(e))
            if once:
                raise
            await asyncio.sleep(POLL_INTERVAL)