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

async def evaluate_faithfulness(question: str, answer: str, context: str) -> float:
    """Evaluate answer faithfulness against context (0-1)."""
    prompt = f"""You are an evaluation judge. Rate the faithfulness of the answer based on the context.

Context: {context}
Question: {question}
Answer: {answer}

Score the answer from 0 to 1 based on whether all claims in the answer are supported by the context.
Return ONLY a number between 0 and 1 (e.g., 0.85).

Faithfulness score:"""
    response = await JUDGE_LLM.ainvoke(prompt)
    try:
        return float(response.content.strip())
    except (ValueError, TypeError):
        return 0.0

async def evaluate_relevance(question: str, answer: str) -> float:
    """Evaluate answer relevance to the question (0-1)."""
    prompt = f"""You are an evaluation judge. Rate the relevance of the answer to the question.

Question: {question}
Answer: {answer}

Score from 0 to 1 based on how directly the answer addresses the question.
Return ONLY a number between 0 and 1.

Relevance score:"""
    response = await JUDGE_LLM.ainvoke(prompt)
    try:
        return float(response.content.strip())
    except (ValueError, TypeError):
        return 0.0

async def evaluate_context_precision(relevant_chunks: list[str], question: str) -> float:
    """Evaluate what proportion of retrieved context is actually relevant."""
    if not relevant_chunks:
        return 0.0
    relevant_count = 0
    for chunk in relevant_chunks[:5]:
        prompt = f"""Is the following context relevant to the question "{question}"?
Answer ONLY "yes" or "no".

Context: {chunk[:500]}"""
        response = await JUDGE_LLM.ainvoke(prompt)
        if response.content.strip().lower().startswith("yes"):
            relevant_count += 1
    return relevant_count / min(len(relevant_chunks), 5)

async def evaluate_answer_correctness(question: str, answer: str, ground_truth: str) -> float:
    """Evaluate answer correctness against ground truth (0-1)."""
    prompt = f"""You are an evaluation judge. Rate the correctness of the answer compared to the ground truth.

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}

Score from 0 to 1 based on how accurate and complete the answer is compared to the ground truth.
A correct answer covers the key information from the ground truth without hallucination.
Return ONLY a number between 0 and 1.

Correctness score:"""
    response = await JUDGE_LLM.ainvoke(prompt)
    try:
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        logger.warning("eval_score_parse_error", metric="answer_correctness", raw=response.content)
        return 0.0

async def evaluate_context_recall(retrieved_chunks: list[str], ground_truth_docs: list[str]) -> float:
    """Evaluate whether the retrieved context contains all info needed to answer (0-1)."""
    if not ground_truth_docs or not retrieved_chunks:
        return 0.0
    context_str = "\n".join(retrieved_chunks[:5])[:2000]
    ground_truth_str = "\n".join([str(d) for d in ground_truth_docs[:3]])[:1500]
    prompt = f"""You are an evaluation judge. Rate the context recall.

Ground Truth Information: {ground_truth_str}
Retrieved Context: {context_str}

Score from 0 to 1 based on whether ALL key information from the ground truth is covered by the retrieved context.
Return ONLY a number between 0 and 1.

Context recall score:"""
    response = await JUDGE_LLM.ainvoke(prompt)
    try:
        return max(0.0, min(1.0, float(response.content.strip())))
    except (ValueError, TypeError):
        return 0.0

async def run_evaluation(
    split: str = "test",
    num_samples: int = 50,
    output_path: str = "evals/results/graphrag_eval.json",
) -> dict[str, Any]:
    """Run evaluation on RAGBench techqa dataset."""
    logger.info("eval_started", split=split, num_samples=num_samples)
    dataset = load_dataset("galileo-ai/ragbench", "techqa", split=split, )

    results = []
    retriever = GraphRAGRetriever()

    try:
        for i, row in enumerate(dataset):
            if i >= num_samples:
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
            faithfulness = await evaluate_faithfulness(question, answer, context_str)
            relevance = await evaluate_relevance(question, answer)
            context_precision = await evaluate_context_precision(context_chunks, question)
            answer_correctness = await evaluate_answer_correctness(question, answer, ground_truth)
            context_recall = await evaluate_context_recall(context_chunks, row.get("documents", []))
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


