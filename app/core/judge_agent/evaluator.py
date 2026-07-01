"""RAGBench evaluation pipeline for GraphRAG.

Evaluates GraphRAG performance using RAGBench techqa dataset.
Metrics: faithfulness, relevance, context_precision, answer_correctness, context_recall.
Cross-model LLM-as-Judge using qwen-plus for evaluation.
"""

from __future__ import annotations

import json
import os
from typing import Any

from datasets import load_dataset
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings
from app.core.graphrag.retriever import GraphRAGRetriever
import time
from app.core.logging import logger
from app.models.eval_result import EvalResult
from sqlmodel import Session, create_engine

_EVAL_DB_URL = f'postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}'
_eval_engine = create_engine(_EVAL_DB_URL, pool_pre_ping=True)

# Cross-model judge: use qwen-plus for more objective evaluation
JUDGE_LLM = ChatOpenAI(
    model="qwen-plus",
    api_key=SecretStr(settings.EVALUATION_API_KEY or settings.LLM_API_KEY),
    base_url=settings.EVALUATION_BASE_URL or settings.LLM_BASE_URL,
    temperature=0,
    max_tokens=1024,
)

async def evaluate_all_metrics(question: str, answer: str, context: str, ground_truth: str, context_chunks: list[str], ground_truth_docs: list[str]) -> dict[str, float]:
    """Evaluate all 5 metrics in a single LLM judge call."""
    context_str = context[:3000]
    chunk_text = "\n---\n".join(context_chunks[:5])[:2000] if context_chunks else "(no chunks)"
    gt_str = "\n".join([str(d) for d in ground_truth_docs[:3]])[:1500] if ground_truth_docs else "(no docs)"
    prompt = f"""You are an evaluation judge for a RAG system. Given the following, score ALL 5 metrics from 0.0 to 1.0.

Question: {question}
Answer: {answer}
Context (retrieved):
{context_str}
Retrieved Chunks:
{chunk_text}
Ground Truth Answer: {ground_truth}
Ground Truth Documents:
{gt_str}

Score each metric:
1. faithfulness - Are all claims in the answer supported by the context?
2. relevance - How directly does the answer address the question?
3. context_precision - What proportion of retrieved chunks are relevant to the question?
4. answer_correctness - How accurate and complete is the answer vs ground truth?
5. context_recall - Does the retrieved context cover ALL key info from ground truth documents?

Return ONLY valid JSON with 5 keys. Example:
{{
  "faithfulness": 0.85,
  "relevance": 0.90,
  "context_precision": 0.75,
  "answer_correctness": 0.80,
  "context_recall": 0.70
}}
"""
    response = await JUDGE_LLM.ainvoke(prompt)
    try:
        import json, re
        text = response.content.strip()
        m = re.search(r"`(?:json)?\s*([\s\S]*?)\s*`", text)
        if m:
            text = m.group(1)
        scores = json.loads(text)
        return {k: max(0.0, min(1.0, float(scores.get(k, 0.0)))) for k in ["faithfulness", "relevance", "context_precision", "answer_correctness", "context_recall"]}
    except Exception as e:
        logger.warning("eval_batch_parse_error", error=str(e), raw=response.content[:200])
        return {"faithfulness": 0.0, "relevance": 0.0, "context_precision": 0.0, "answer_correctness": 0.0, "context_recall": 0.0}


async def run_evaluation(
    split: str = "test",
    num_samples: int = 100,
    output_path: str = "evals/results/graphrag_eval.json",
) -> dict[str, Any]:
    """Run evaluation on RAGBench techqa dataset."""
    logger.info("eval_started", split=split, num_samples=num_samples)
    dataset = load_dataset("galileo-ai/ragbench", "techqa", split=split, )

    results = []
    retriever = GraphRAGRetriever()

    try:
        for i, row in enumerate(dataset):
            if num_samples > 0 and i >= num_samples:
                break

            question = row["question"]
            ground_truth = row["response"]

            # Run GraphRAG retrieval
            rag_result = await retriever.local_search(question)
            context_chunks = [r.get("text", "") for r in rag_result.get("vector_context", [])]
            context_str = "\n".join(context_chunks[:5])

            # Get answer from LLM
            answer_prompt = f"""Answer the following question based on the provided context.

Context: {context_str}
Question: {question}

Provide a concise, accurate answer:"""
            answer_response = await JUDGE_LLM.ainvoke(answer_prompt)
            answer = answer_response.content.strip()

            # Evaluate
            t0 = time.time()
            scores = await evaluate_all_metrics(question, answer, context_str, ground_truth, context_chunks, row.get("documents", []))
            faithfulness = scores["faithfulness"]
            relevance = scores["relevance"]
            context_precision = scores["context_precision"]
            answer_correctness = scores["answer_correctness"]
            context_recall = scores["context_recall"]
            elapsed_ms = int((time.time() - t0) * 1000)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "context_precision": context_precision,
                "answer_correctness": answer_correctness,
                "context_recall": context_recall,                "response_time_ms": elapsed_ms,
            })

            logger.info(
                "eval_sample",
                sample=i,
                faithfulness=round(faithfulness, 3),
                relevance=round(relevance, 3),
                context_precision=round(context_precision, 3),
            )

    finally:
        await retriever.close()

    # Aggregate metrics
    metrics = {
        "faithfulness": sum(r["faithfulness"] for r in results) / len(results) if results else 0,
        "relevance": sum(r["relevance"] for r in results) / len(results) if results else 0,
        "context_precision": sum(r["context_precision"] for r in results) / len(results) if results else 0,
        "answer_correctness": sum(r["answer_correctness"] for r in results) / len(results) if results else 0,
        "context_recall": sum(r["context_recall"] for r in results) / len(results) if results else 0,        "avg_response_time_ms": sum(r["response_time_ms"] for r in results) / len(results) if results else 0,
    }

    report = {
        "config": {
            "split": split,
            "num_samples": num_samples,
            "retriever": "GraphRAG (vector + graph)",
            "judge_model": "qwen-plus",
        },
        "metrics": metrics,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Write to PostgreSQL
    import uuid
    run_id = str(uuid.uuid4())[:8]
    trace_id = str(uuid.uuid4())
    try:
        from sqlmodel import Session
        er = EvalResult(
            run_id=run_id,
            trace_id=trace_id,
            faithfulness=metrics["faithfulness"],
            relevance=metrics["relevance"],
            context_precision=metrics["context_precision"],
            answer_correctness=metrics["answer_correctness"],
            context_recall=metrics["context_recall"],            top_k=settings.GRAPHRAG_TOP_K,
            chunk_size=settings.GRAPHRAG_CHUNK_SIZE,
            chunk_overlap=settings.GRAPHRAG_CHUNK_OVERLAP,
            judge_model=report["config"]["judge_model"],
            num_samples=num_samples,
            split=split,
        )
        with Session(_eval_engine) as ses:
            ses.add(er)
            ses.commit()
        logger.info("eval_result_saved", run_id=run_id)
    except Exception as e:
        logger.warning("eval_result_save_failed", error=str(e))

    logger.info("eval_complete", metrics=metrics, output=output_path)
    return report

import httpx

async def get_token_usage_from_langfuse(trace_name: str = None, since_minutes: int = 60) -> dict:
    """Get token usage from Langfuse API for recent traces."""
    from app.core.config import settings
    pf = settings.LANGFUSE_PUBLIC_KEY or ""
    sf = settings.LANGFUSE_SECRET_KEY or ""
    host = settings.LANGFUSE_HOST or "https://cloud.langfuse.com"
    if not pf or not sf:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    import base64
    auth = base64.b64encode(f"{pf}:{sf}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    params = {"limit": 10}
    if trace_name:
        params["name"] = trace_name
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{host}/api/public/observations", headers=headers, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                total_in = sum(o.get("usage", {}).get("input", 0) for o in data.get("data", []))
                total_out = sum(o.get("usage", {}).get("output", 0) for o in data.get("data", []))
                return {"input_tokens": total_in, "output_tokens": total_out, "total_tokens": total_in + total_out}
    except Exception as e:
        logger.warning("langfuse_token_fetch_error", error=str(e))
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

async def compare_with_baseline() -> dict[str, Any]:
    """Compare GraphRAG results with RAGBench baseline scores."""
    baseline = {
        "ragas_faithfulness": 0.82,
        "trulens_groundedness": 0.78,
        "gpt3_adherence": 0.85,
    }

    with open("evals/results/graphrag_eval.json", "r", encoding="utf-8") as f:
        graphrag_results = json.load(f)

    comparison = {
        "baseline": baseline,
        "graphrag": graphrag_results["metrics"],
        "delta": {
            k: round(graphrag_results["metrics"].get("faithfulness", 0) - v, 3)
            for k, v in baseline.items()
        },
    }
    return comparison


