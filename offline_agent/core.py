import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.logging import logger

async def run_evaluation(generate_report: bool = True) -> dict:
    """Run RAGBench offline evaluation via evals/evaluate_graphrag."""
    logger.info("offline_agent_eval_started", generate_report=generate_report)
    from evals.evaluate_graphrag import run_evaluation as _eval
    result = await _eval()
    logger.info("offline_agent_eval_complete")
    return {"status": "completed", "result": str(result)[:200]}

def run_analysis() -> dict:
    """Generate retrieval metrics analysis report via scripts/analyze_retrieval."""
    logger.info("offline_agent_analyze_started")
    from scripts.analyze_retrieval import main as _analyze
    _analyze()
    logger.info("offline_agent_analyze_complete")
    return {"status": "completed", "report": "evals/reports/analysis_report.json"}

def run_optimization(apply: bool = False) -> dict:
    """Run LLM-driven optimization suggestions via scripts/optimize_rag."""
    logger.info("offline_agent_optimize_started", apply=apply)
    from scripts.optimize_rag import main as _optimize
    _optimize(apply=apply)
    logger.info("offline_agent_optimize_complete")
    return {"status": "completed", "report": "evals/reports/optimization_suggestion.json"}
