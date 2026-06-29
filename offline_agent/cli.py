"""Offline Agent CLI - unified entrypoint for eval, analysis, and optimization."""
from __future__ import annotations
import argparse, asyncio, os, sys, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.logging import logger
from app.core.config import settings

def cmd_eval(num_samples: int = 50, split: str = "test"):
    """Run RAGBench offline evaluation."""
    logger.info("offline_agent_eval_started", split=split, num_samples=num_samples)
    import asyncio
    from evals.evaluate_graphrag import run_evaluation
    result = asyncio.run(run_evaluation(split=split, num_samples=num_samples))
    logger.info("offline_agent_eval_complete")
    return result

def cmd_analyze(days: int = 7):
    """Generate retrieval metrics analysis report."""
    logger.info("offline_agent_analyze_started", days=days)
    from scripts.analyze_retrieval import generate_all_reports, print_report_summary
    report = generate_all_reports(days=days)
    print_report_summary(report)
    logger.info("offline_agent_analyze_complete", report_path="evals/reports/analysis_report.json")
    return report

def cmd_optimize(apply: bool = False):
    """Run LLM-driven optimization suggestions."""
    logger.info("offline_agent_optimize_started", apply=apply)
    from scripts.optimize_rag import run_optimization, apply_suggestion
    if apply:
        result = apply_suggestion()
    else:
        result = run_optimization()
    logger.info("offline_agent_optimize_complete")
    return result

def cmd_pipeline(apply: bool = False):
    """Full offline pipeline: eval -> analyze -> suggest."""
    logger.info("offline_agent_pipeline_started")
    
    # Step 1: Evaluate
    logger.info("pipeline_step_1_eval")
    import asyncio
    from evals.evaluate_graphrag import run_evaluation
    asyncio.run(run_evaluation(split="test", num_samples=50))
    
    # Step 2: Analyze
    logger.info("pipeline_step_2_analyze")
    from scripts.analyze_retrieval import generate_all_reports
    generate_all_reports(days=7)
    
    # Step 3: Optimize
    logger.info("pipeline_step_3_optimize")
    from scripts.optimize_rag import run_optimization, apply_suggestion
    if apply:
        apply_suggestion()
    else:
        run_optimization()
    
    logger.info("offline_agent_pipeline_complete")
    return {"status": "completed"}

def main():
    parser = argparse.ArgumentParser(description="Offline Agent: evaluation, analysis, optimization")
    sub = parser.add_subparsers(dest="command", required=True)
    
    p_eval = sub.add_parser("eval", help="Run RAGBench offline evaluation")
    p_eval.add_argument("--num-samples", type=int, default=50)
    p_eval.add_argument("--split", default="test")
    
    p_analyze = sub.add_parser("analyze", help="Generate retrieval analysis report")
    p_analyze.add_argument("--days", type=int, default=7)
    
    p_optimize = sub.add_parser("optimize", help="Run LLM optimization suggestions")
    p_optimize.add_argument("--apply", action="store_true", help="Apply approved suggestions")
    
    p_pipe = sub.add_parser("pipeline", help="Full pipeline: eval -> analyze -> optimize")
    p_pipe.add_argument("--apply", action="store_true", help="Apply optimizations at the end")
    
    args = parser.parse_args()
    
    if args.command == "eval":
        cmd_eval(num_samples=args.num_samples, split=args.split)
    elif args.command == "analyze":
        cmd_analyze(days=args.days)
    elif args.command == "optimize":
        cmd_optimize(apply=args.apply)
    elif args.command == "pipeline":
        cmd_pipeline(apply=args.apply)

if __name__ == "__main__":
    main()
