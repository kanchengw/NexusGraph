"""Analyze retrieval metrics from PostgreSQL for data-driven optimization.

Reads retrieval_metric, evalresult, and feedback tables to produce:
1. path_contribution_report - which path contributes independently
2. feedback_correlation_report - what retrieval patterns lead to bad ratings
3. param_experiment_report - how different params affect metrics
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

from sqlmodel import Session, create_engine, text

from app.core.config import settings as _as
_DB_URL = f"postgresql://{_as.POSTGRES_USER}:{_as.POSTGRES_PASSWORD}@{_as.POSTGRES_HOST}:{_as.POSTGRES_PORT}/{_as.POSTGRES_DB}"
_engine = create_engine(_DB_URL, pool_pre_ping=True)

REPORT_DIR = "evals/reports"


import json
import pathlib

EVAL_RESULTS_DIR = pathlib.Path("evals/results")




async def load_langfuse_llm_metrics(minutes: int = 60) -> dict:
    """Query Langfuse API for LLM metrics: token usage, latency, per-path durations."""
    from app.core.config import settings
    pf = settings.LANGFUSE_PUBLIC_KEY or ""
    sf = settings.LANGFUSE_SECRET_KEY or ""
    host = settings.LANGFUSE_HOST or "https://cloud.langfuse.com"
    if not pf or not sf:
        return {"error": "Langfuse not configured"}
    import base64
    import httpx
    auth = base64.b64encode(f"{pf}:{sf}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    limit = 50

    result = {
        "total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0,
        "avg_llm_latency_ms": 0, "llm_span_count": 0,
        "avg_vector_search_ms": 0, "vector_search_count": 0,
        "avg_bm25_search_ms": 0, "bm25_search_count": 0,
        "avg_graph_expand_ms": 0, "graph_expand_count": 0,
    }

    async def _fetch_spans(name: str) -> list[dict]:
        """Fetch observations by name from Langfuse API."""
        params = {"limit": limit, "name": name}
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{host}/api/public/observations", headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("data", [])
        return []

    try:
        # 1. LLM inference spans (auto-captured by langfuse_callback_handler)
        llm_spans = await _fetch_spans("Langfuse")
        total_latency = 0
        for s in llm_spans:
            usage = s.get("usage", {}) or {}
            result["total_input_tokens"] += usage.get("input", 0)
            result["total_output_tokens"] += usage.get("output", 0)
            result["total_cost"] += usage.get("totalCost", 0)
            if s.get("endTime") and s.get("startTime"):
                lat = (s["endTime"] - s["startTime"]) / 1000000
                total_latency += lat
                result["llm_span_count"] += 1
        if result["llm_span_count"] > 0:
            result["avg_llm_latency_ms"] = round(total_latency / result["llm_span_count"], 1)

        # 2. Per-path retrieval spans (written by graphrag/__init__.py)
        for span_name, count_key, avg_key in [
            ("vector_search", "vector_search_count", "avg_vector_search_ms"),
            ("bm25_search", "bm25_search_count", "avg_bm25_search_ms"),
            ("graph_expand", "graph_expand_count", "avg_graph_expand_ms"),
        ]:
            spans = await _fetch_spans(span_name)
            total_dur = 0
            for s in spans:
                if s.get("endTime") and s.get("startTime"):
                    dur = (s["endTime"] - s["startTime"]) / 1000000
                    total_dur += dur
                    result[count_key] += 1
            if result[count_key] > 0:
                result[avg_key] = round(total_dur / result[count_key], 1)

    except Exception as e:
        result["error"] = str(e)
    return result



    files = sorted(EVAL_RESULTS_DIR.glob("graphrag_eval*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    history = []
    for f in files[:n]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            history.append({
                "file": f.name,
                "faithfulness": metrics.get("faithfulness"),
                "relevance": metrics.get("relevance"),
                "context_precision": metrics.get("context_precision"),
                "answer_correctness": metrics.get("answer_correctness"),
                "context_recall": metrics.get("context_recall"),
                "hit_rate": metrics.get("hit_rate"),
                "avg_response_time_ms": metrics.get("avg_response_time_ms"),
                "total_tokens": metrics.get("total_tokens"),
                "num_samples": data.get("config", {}).get("num_samples"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return history


async def load_prometheus_metrics():
    result = {}
    try:
        import httpx
        # System-level metrics from Prometheus (NOT LLM-level)
        # LLM latency/tokens come from Langfuse via eval report
        queries = {
            "avg_request_latency_5m": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
            "request_rate_5m": "rate(http_requests_total[5m])",
            "db_connections": "db_connections",
        }
        async with httpx.AsyncClient() as client:
            for name, query in queries.items():
                r = await client.get("http://localhost:9090/api/v1/query", params={"query": query}, timeout=5)
                if r.status_code == 200:
                    result[name] = r.json().get("data", {}).get("result", [])
    except Exception as e:
        result["error"] = str(e)
    return result



    os.makedirs(REPORT_DIR, exist_ok=True)


def run_sql(sql: str) -> list[dict[str, Any]]:
    with Session(_engine) as session:
        result = session.execute(text(sql))
        cols = result.keys()
        return [dict(zip(cols, row)) for row in result.fetchall()]


# ─── Report 1: Path Contribution ─────────────────────────────


def path_contribution_report(days: int = 7) -> dict[str, Any]:
    """Analyze how much each retrieval path contributes independently."""
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    rows = run_sql(f"""
        SELECT
            COUNT(*) AS total_queries,
            AVG(vector_count)::numeric(10,2) AS avg_vector,
            AVG(bm25_count)::numeric(10,2) AS avg_bm25,
            AVG(graph_count)::numeric(10,2) AS avg_graph,
            AVG(unique_chunks)::numeric(10,2) AS avg_unique,
            AVG(vector_only)::numeric(10,2) AS avg_vector_only,
            AVG(bm25_only)::numeric(10,2) AS avg_bm25_only,
            AVG(graph_only)::numeric(10,2) AS avg_graph_only,
            AVG(overlap_vector_bm25)::numeric(10,2) AS avg_overlap,
            AVG(response_time_ms)::numeric(10,0) AS avg_response_ms
        FROM retrievalmetric
        WHERE timestamp >= '{since}'
    """)

    report = {
        "period_days": days,
        "sample_size": rows[0]["total_queries"] if rows else 0,
        "averages": rows[0] if rows else {},
    }

    if rows and rows[0]["total_queries"] > 0:
        r = rows[0]
        total = r["avg_unique"] or 1
        report["contribution_rates"] = {
            "vector_independent_pct": round(r["avg_vector_only"] / total * 100, 1),
            "bm25_independent_pct": round(r["avg_bm25_only"] / total * 100, 1),
            "graph_independent_pct": round(r["avg_graph_only"] / total * 100, 1),
            "vector_bm25_overlap_pct": round(r["avg_overlap"] / total * 100, 1),
        }
        report["diagnosis"] = _diagnose_paths(report["contribution_rates"])

    return report


def _diagnose_paths(rates: dict[str, float]) -> list[str]:
    diag = []
    if rates.get("bm25_independent_pct", 100) < 5:
        diag.append("BM25 independent rate < 5%: BM25 path needs improvement")
    if rates.get("graph_independent_pct", 100) < 3:
        diag.append("Graph expand independent rate < 3%: graph quality may be low")
    if rates.get("vector_independent_pct", 0) > 80:
        diag.append("Vector dominates (>80%): other paths not contributing independently")
    if rates.get("vector_bm25_overlap_pct", 0) > 80:
        diag.append("Vector-BM25 overlap > 80%: two paths are redundant")
    if not diag:
        diag.append("All paths contributing: no critical issues detected")
    return diag


# ─── Report 2: Feedback Correlation ─────────────────────────


def feedback_correlation_report(days: int = 30) -> dict[str, Any]:
    """Correlate retrieval metrics with user feedback ratings."""
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    # Queries with low ratings (rating <= 2) vs high ratings (rating >= 4)
    low = run_sql(f"""
        SELECT
            AVG(r.vector_count)::numeric(10,2) AS avg_vector,
            AVG(r.bm25_count)::numeric(10,2) AS avg_bm25,
            AVG(r.graph_count)::numeric(10,2) AS avg_graph,
            AVG(r.unique_chunks)::numeric(10,2) AS avg_unique,
            AVG(r.vector_only)::numeric(10,2) AS avg_vector_only,
            AVG(r.response_time_ms)::numeric(10,0) AS avg_ms,
            COUNT(*) AS cnt
        FROM retrievalmetric r
        JOIN feedback f ON r.query = f.query
        WHERE f.rating <= 2 AND f.created_at >= '{since}'
    """)

    high = run_sql(f"""
        SELECT
            AVG(r.vector_count)::numeric(10,2) AS avg_vector,
            AVG(r.bm25_count)::numeric(10,2) AS avg_bm25,
            AVG(r.graph_count)::numeric(10,2) AS avg_graph,
            AVG(r.unique_chunks)::numeric(10,2) AS avg_unique,
            AVG(r.vector_only)::numeric(10,2) AS avg_vector_only,
            AVG(r.response_time_ms)::numeric(10,0) AS avg_ms,
            COUNT(*) AS cnt
        FROM retrievalmetric r
        JOIN feedback f ON r.query = f.query
        WHERE f.rating >= 4 AND f.created_at >= '{since}'
    """)

    return {
        "period_days": days,
        "low_rated_queries": low[0] if low else {"cnt": 0},
        "high_rated_queries": high[0] if high else {"cnt": 0},
        "diagnosis": _diagnose_feedback(low[0] if low else None, high[0] if high else None),
    }


def _diagnose_feedback(low: dict | None, high: dict | None) -> list[str]:
    diag = []
    if not low or low["cnt"] == 0:
        diag.append("No low-rated queries with matching metrics yet")
        return diag
    if not high or high["cnt"] == 0:
        diag.append("No high-rated queries with matching metrics yet")
        return diag

    if (low.get("avg_vector_only", 5) or 5) > (high.get("avg_vector_only", 3) or 3) + 2:
        diag.append("Low-rated queries retrieve MORE unique chunks than high-rated: possible noise")
    if (low.get("avg_ms", 1000) or 1000) > (float(high.get("avg_ms", 500)) or 500) * 1.5:
        diag.append("Low-rated queries have significantly higher latency")
    if (low.get("avg_bm25", 0) or 0) < (high.get("avg_bm25", 1) or 1):
        diag.append("BM25 contributes less to low-rated queries: check keyword quality")
    if not diag:
        diag.append("No clear pattern detected between ratings and retrieval metrics")
    return diag


# ─── Report 3: Parameter Experiment ─────────────────────────


def param_experiment_report() -> dict[str, Any]:
    """Compare eval results across different parameter combinations."""
    rows = run_sql("""
        SELECT
            top_k,
            chunk_size,
            chunk_overlap,
            AVG(faithfulness)::numeric(10,3) AS avg_faithfulness,
            AVG(relevance)::numeric(10,3) AS avg_relevance,
            AVG(context_precision)::numeric(10,3) AS avg_precision,
            COUNT(*) AS num_runs
        FROM evalresult
        GROUP BY top_k, chunk_size, chunk_overlap
        ORDER BY avg_faithfulness DESC
    """)

    if not rows:
        return {"message": "No eval results yet. Run 'make eval-rag' first."}

    best = rows[0]
    return {
        "param_combinations": rows,
        "best_params": {
            "top_k": best["top_k"],
            "chunk_size": best["chunk_size"],
            "chunk_overlap": best["chunk_overlap"],
        },
        "diagnosis": _diagnose_params(rows),
    }


def _diagnose_params(rows: list[dict]) -> list[str]:
    diag = []
    if len(rows) < 2:
        diag.append("Only one parameter combination tested. Run 'optimize_rag.py' for comparison.")
        return diag
    best = rows[0]
    for r in rows[1:]:
        diff = best["avg_faithfulness"] - r["avg_faithfulness"]
        if abs(diff) < 0.02:
            diag.append(
                f"top_k={r['top_k']}/chunk={r['chunk_size']} vs "
                f"top_k={best['top_k']}/chunk={best['chunk_size']}: "
                f"faithfulness diff < 2% (not significant)"
            )
    if not diag:
        diag.append("Parameter differences are significant: keep tracking")
    return diag


# ─── Main ──────────────────────────────────────────────────


def load_eval_history(n: int = 5) -> list[dict]:
    """Load most recent eval results from JSON files."""
    files = sorted(EVAL_RESULTS_DIR.glob("graphrag_eval*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    history = []
    for f in files[:n]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            history.append({
                "file": f.name,
                "faithfulness": metrics.get("faithfulness"),
                "relevance": metrics.get("relevance"),
                "context_precision": metrics.get("context_precision"),
                "answer_correctness": metrics.get("answer_correctness"),
                "context_recall": metrics.get("context_recall"),
                "hit_rate": metrics.get("hit_rate"),
                "avg_response_time_ms": metrics.get("avg_response_time_ms"),
                "total_tokens": metrics.get("total_tokens"),
                "num_samples": data.get("config", {}).get("num_samples"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return history


async def generate_all_reports(days: int = 7) -> dict[str, Any]:
    os.makedirs(REPORT_DIR, exist_ok=True)
    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "path_contribution": path_contribution_report(days),
        "feedback_correlation": feedback_correlation_report(days),
        "param_experiment": param_experiment_report(),
        "eval_history": load_eval_history(5),
        "prometheus_metrics": await load_prometheus_metrics(),
        "langfuse_llm_metrics": await load_langfuse_llm_metrics(),
    }

    path = os.path.join(REPORT_DIR, "analysis_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"Report saved to {path}")
    full_path = os.path.join(REPORT_DIR, "analysis_report.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"Full report (with eval history + Prometheus) saved to {full_path}")

    # Write Langfuse metrics to PG for Grafana
    try:
        lf_metrics = result.get("langfuse_llm_metrics", {})
        if lf_metrics and "error" not in lf_metrics:
            from app.models.langfuse_snapshot import LangfuseSnapshot
            snap = LangfuseSnapshot(
                vector_search_avg_ms=lf_metrics.get("avg_vector_search_ms", 0),
                vector_search_count=lf_metrics.get("vector_search_count", 0),
                bm25_search_avg_ms=lf_metrics.get("avg_bm25_search_ms", 0),
                bm25_search_count=lf_metrics.get("bm25_search_count", 0),
                graph_expand_avg_ms=lf_metrics.get("avg_graph_expand_ms", 0),
                graph_expand_count=lf_metrics.get("graph_expand_count", 0),
                llm_inference_avg_ms=lf_metrics.get("avg_llm_latency_ms", 0),
                llm_span_count=lf_metrics.get("llm_span_count", 0),
                total_input_tokens=lf_metrics.get("total_input_tokens", 0),
                total_output_tokens=lf_metrics.get("total_output_tokens", 0),
                total_cost=lf_metrics.get("total_cost", 0),
            )
            with Session(_engine) as ses:
                ses.add(snap)
                ses.commit()
    except Exception as e:
        logger.warning("langfuse_snapshot_write_failed", error=str(e))

    return result


def print_report_summary(report: dict[str, Any]) -> None:
    print("=" * 60)
    print("RETRIEVAL ANALYSIS REPORT")
    print("=" * 60)

    pc = report.get("path_contribution", {})
    print(f"\n[1] Path Contribution (last {pc.get('period_days', '?')}d)")
    print(f"    Queries analyzed: {pc.get('sample_size', 0)}")
    rates = pc.get("contribution_rates", {})
    if rates:
        for k, v in rates.items():
            print(f"    {k}: {v}%")
    for d in pc.get("diagnosis", []):
        print(f"    >> {d}")

    fc = report.get("feedback_correlation", {})
    print(f"\n[2] Feedback Correlation (last {fc.get('period_days', '?')}d)")
    lo = fc.get("low_rated_queries", {})
    hi = fc.get("high_rated_queries", {})
    print(f"    Low-rated queries: {lo.get('cnt', 0)}")
    print(f"    High-rated queries: {hi.get('cnt', 0)}")
    for d in fc.get("diagnosis", []):
        print(f"    >> {d}")

    pe = report.get("param_experiment", {})
    print(f"\n[3] Parameter Experiments")
    combos = pe.get("param_combinations", [])
    if combos:
        print(f"    Combinations tested: {len(combos)}")
        bp = pe.get("best_params", {})
        if bp:
            print(f"    Best: top_k={bp.get('top_k')}, chunk={bp.get('chunk_size')}")
    else:
        print(f"    {pe.get('message', 'No data')}")
    for d in pe.get("diagnosis", []):
        print(f"    >> {d}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze retrieval metrics")
    parser.add_argument("--days", type=int, default=7, help="Lookback period in days")
    parser.add_argument("--summary", action="store_true", help="Print summary to console")
    args = parser.parse_args()

    report = asyncio.run(generate_all_reports(days=args.days))
    if args.summary:
        print_report_summary(report)
