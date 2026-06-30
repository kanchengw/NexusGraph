"""Offline Agent CLI - delegates to structured agent modules."""
from __future__ import annotations
import argparse, asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.logging import logger


def cmd_eval(num_samples: int = 50, split: str = "test"):
    """Run evaluation via Judge Agent."""
    logger.info("cli_eval_started", split=split, num_samples=num_samples)
    from app.core.judge_agent.evaluator import run_evaluation
    result = asyncio.run(run_evaluation(split=split, num_samples=num_samples))
    logger.info("cli_eval_complete")
    return result


def cmd_analyze(days: int = 7):
    """Generate analysis report via Optimizer Agent."""
    logger.info("cli_analyze_started", days=days)
    from app.core.optimizer_agent.analyzer import generate_all_reports, print_report_summary
    report = asyncio.run(generate_all_reports(days=days))
    print_report_summary(report)
    logger.info("cli_analyze_complete")
    return report


def cmd_optimize(apply: bool = False):
    """Run LLM optimization via Optimizer Agent."""
    logger.info("cli_optimize_started", apply=apply)
    from app.core.optimizer_agent.optimizer import run_optimization, apply_suggestion
    if apply:
        result = apply_suggestion()
    else:
        result = run_optimization()
    logger.info("cli_optimize_complete")
    return result


def cmd_index(reset: bool = False, max_docs: int = 200, skip_entities: bool = False):
    """Run KB indexing via Index Agent."""
    logger.info("cli_index_started", reset=reset, max_docs=max_docs)
    from app.core.index_agent.indexer import run_indexing
    asyncio.run(run_indexing(split="train", reset=reset, skip_entities=skip_entities))
    logger.info("cli_index_complete")


def cmd_flywheel(once: bool = False, force: bool = False):
    """Run the data flywheel (Judge -> Optimizer -> Index)."""
    logger.info("cli_flywheel_started", once=once)
    from app.core.data_flywheel import run_flywheel
    asyncio.run(run_flywheel(continuous=not once, once=once, force=force))
    logger.info("cli_flywheel_complete")


def main():
    parser = argparse.ArgumentParser(description="NexusGraph Offline Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run RAGBench offline evaluation")
    p_eval.add_argument("--num-samples", type=int, default=50)
    p_eval.add_argument("--split", default="test")

    p_analyze = sub.add_parser("analyze", help="Generate retrieval analysis report")
    p_analyze.add_argument("--days", type=int, default=7)

    p_optimize = sub.add_parser("optimize", help="Run LLM optimization suggestions")
    p_optimize.add_argument("--apply", action="store_true")

    p_index = sub.add_parser("index", help="Build/rebuild knowledge base")
    p_index.add_argument("--reset", action="store_true")
    p_index.add_argument("--max-docs", type=int, default=200)
    p_index.add_argument("--skip-entities", action="store_true")

    p_fly = sub.add_parser("flywheel", help="Automated data flywheel")
    p_fly.add_argument("--once", action="store_true", help="Run one cycle then exit")
    p_fly.add_argument("--force", action="store_true", help="Force trigger regardless of threshold")

    args = parser.parse_args()

    if args.command == "eval":
        cmd_eval(num_samples=args.num_samples, split=args.split)
    elif args.command == "analyze":
        cmd_analyze(days=args.days)
    elif args.command == "optimize":
        cmd_optimize(apply=args.apply)
    elif args.command == "index":
        cmd_index(reset=args.reset, max_docs=args.max_docs, skip_entities=args.skip_entities)
    elif args.command == "flywheel":
        cmd_flywheel(once=args.once, force=args.force)


if __name__ == "__main__":
    main()
