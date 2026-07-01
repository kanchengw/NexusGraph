"""Data-driven RAG optimization with LLM analysis.
Reads analysis report from analyze_retrieval.py, sends to LLM for
optimization suggestions, produces human-readable suggestion report.
Human-in-the-loop: report must be approved before applying changes.
"""

from __future__ import annotations
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings
from app.core.logging import logger

REPORT_PATH = "evals/reports/analysis_report.json"
SUGGESTION_PATH = "evals/reports/optimization_suggestion.json"

_ANALYST_LLM = ChatOpenAI(
    model="qwen-plus",
    api_key=SecretStr(settings.EVALUATION_API_KEY or settings.LLM_API_KEY),
    base_url=settings.EVALUATION_BASE_URL or settings.LLM_BASE_URL,
    temperature=0.1,
    max_tokens=2048,
)

ANALYSIS_PROMPT = """You are a RAG system optimization analyst. Your job is to analyze retrieval metrics data and suggest concrete, data-backed parameter changes.

## Input Data

{report_json}

## Data Sources
The report includes data from four independent sources:
SYSTEM (Prometheus): request latency, QPS, DB connections - infrastructure health
RETRIEVAL (PostgreSQL): path contribution, overlap, feedback correlation
EVAL (Langfuse + RAGBench): faithfulness, correctness, recall, token usage - quality metrics
PIPELINE_HISTORY (PostgreSQL): previous optimization cycles with their parameter changes, eval metrics, and outcomes (passed/rolled_back)

## Optimization Memory
The pipeline_history section shows what parameter changes were attempted in previous flywheel cycles
and what the results were. IMPORTANT rules:
1. If a parameter change was previously tried and resulted in ROLLED_BACK (degraded), do NOT suggest the same change again.
2. If a parameter value was previously used and had better metrics than the current value, consider reverting to it.
3. Look at the trend across multiple eval results, not just the latest one.
4. NEVER suggest ping-pong changes (cycling between the same two values for the same parameter).
1. PATH CONTRIBUTION: How each retrieval path (vector, BM25, graph) performs independently
2. EVAL HISTORY: Quality metrics (faithfulness, relevance, precision, recall, hit_rate) from last 5 evaluation runs
3. PROMETHEUS METRICS: System-level performance (request latency, QPS, LLM inference time)

## Analysis Guidelines

1. Use pipeline_history to understand what was tried before. Do NOT suggest changes that were already rolled back.
2. For each metric path (vector, BM25, graph), evaluate:
   - Independent contribution rate: how often does this path find unique chunks?
   - Overlap rate: how much does this path overlap with others?
   - Response time impact
3. For each suggestion, provide:
   - What to change
   - Why (cite specific data points)
   - Expected impact on faithfulness/relevance/response time
   - Risk level (low/medium/high)
4. Consider these parameters for tuning:
   - GRAPHRAG_TOP_K (currently {current_top_k})
   - GRAPHRAG_CHUNK_SIZE (currently {current_chunk_size})
   - GRAPHRAG_CHUNK_OVERLAP (currently {current_overlap})
   - BM25 query format or index rebuild
   - Graph expand max_hops (currently 2)
5. Use feedback correlation data if available:
   - What retrieval patterns are associated with low ratings?
   - What patterns are associated with high ratings?

## Output Format

Return a JSON object with this EXACT structure:
{{
  "analysis_timestamp": "ISO timestamp",
  "data_summary": {{
    "total_queries_analyzed": 0,
    "low_rated_query_count": 0,
    "high_rated_query_count": 0,
    "param_combinations_tested": 0
  }},
  "suggestions": [
    {{
      "id": 1,
      "action": "change_top_k" or "change_chunk_size" or "change_overlap" or "rebuild_index" or "other",
      "parameter": "GRAPHRAG_TOP_K",
      "current_value": 5,
      "proposed_value": 3,
      "reasoning": "Data citations for why this change is suggested",
      "expected_impact": "What metric should improve",
      "risk": "low" or "medium" or "high"
    }}
  ],
  "overall_assessment": "Brief summary of system health",
  "recommended_priority": "Which suggestion to act on first"
}}
"""


def load_analysis() -> dict[str, Any] | None:
    if not os.path.exists(REPORT_PATH):
        logger.warning("no_analysis_report", path=REPORT_PATH)
        return None
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_llm_analysis(report: dict[str, Any]) -> dict[str, Any]:
    prompt = ANALYSIS_PROMPT.format(
        report_json=json.dumps(report, indent=2, ensure_ascii=False, default=str),
        current_top_k=settings.GRAPHRAG_TOP_K,
        current_chunk_size=settings.GRAPHRAG_CHUNK_SIZE,
        current_overlap=settings.GRAPHRAG_CHUNK_OVERLAP,
    )
    response = _ANALYST_LLM.invoke(prompt)
    content = response.content.strip()
    try:
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        return json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("llm_parse_failed", error=str(e))
        return {
            "error": "Parse failed",
            "raw_response": content,
            "suggestions": [],
            "overall_assessment": "LLM analysis failed",
        }


def save_suggestion(suggestion: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(SUGGESTION_PATH), exist_ok=True)
    suggestion["saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(SUGGESTION_PATH, "w", encoding="utf-8") as f:
        json.dump(suggestion, f, indent=2, ensure_ascii=False, default=str)
    logger.info("suggestion_saved", path=SUGGESTION_PATH)
    return SUGGESTION_PATH


def print_report(s: dict[str, Any]) -> None:
    print("=" * 60)
    print("OPTIMIZATION SUGGESTION REPORT")
    print("=" * 60)
    ds = s.get("data_summary", {})
    print(f"\nQueries analyzed: {ds.get('total_queries_analyzed', 'N/A')}")
    print(f"Low-rated: {ds.get('low_rated_query_count', 'N/A')} | High-rated: {ds.get('high_rated_query_count', 'N/A')}")
    print(f"Param combos: {ds.get('param_combinations_tested', 'N/A')}")
    print(f"\nOverall: {s.get('overall_assessment', 'N/A')}")
    print(f"Priority: {s.get('recommended_priority', 'N/A')}")
    for sug in s.get("suggestions", []):
        print(f"\n  [{sug.get('risk','?').upper()}] {sug.get('action','?')}")
        print(f"    {sug.get('parameter','?')}: {sug.get('current_value','?')} -> {sug.get('proposed_value','?')}")
        print(f"    Why: {sug.get('reasoning','?')}")
        print(f"    Impact: {sug.get('expected_impact','?')}")
    print(f"\nReport: {SUGGESTION_PATH}")
    print(f"Approve: python scripts/optimize_rag.py --apply")
    print("=" * 60)


def run_optimization() -> dict[str, Any]:
    report = load_analysis()
    if not report:
        return {"status": "skipped", "reason": "Run analyze_retrieval.py first"}
    logger.info("optimization_started")
    suggestion = run_llm_analysis(report)
    save_suggestion(suggestion)
    print_report(suggestion)
    return suggestion


def apply_suggestion(path: str | None = None) -> dict[str, Any]:
    path = path or SUGGESTION_PATH
    if not os.path.exists(path):
        return {"status": "error", "reason": f"Not found: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        suggestion = json.load(f)
    changes = {}
    for s in suggestion.get("suggestions", []):
        p = s.get("parameter")
        v = s.get("proposed_value")
        if p and v is not None:
            changes[p] = v
    if not changes:
        return {"status": "no_changes"}
    logger.info("optimization_applied", changes=changes)
    print("Changes logged. Update .env.development and rebuild:")
    for k, v in changes.items():
        print(f"  {k} = {v}")
    print("  make rebuild-index")
    return {"status": "logged", "changes": changes}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", nargs="?", const=None, help="Apply suggestion report")
    args = parser.parse_args()
    if args.apply is not None:
        apply_suggestion(args.apply)
    else:
        run_optimization()
