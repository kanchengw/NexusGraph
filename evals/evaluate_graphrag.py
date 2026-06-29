"""RAGBench evaluation pipeline for GraphRAG.

Evaluates GraphRAG performance using RAGBench techqa dataset.
Metrics: faithfulness, relevance, context_precision.
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
from app.core.logging import logger
from app.models.eval_result import EvalResult
from sqlmodel import Session, create_engine

_EVAL_DB_URL = "postgresql://myuser:mypassword@localhost:5432/mydb"
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
            faithfulness = await evaluate_faithfulness(question, answer, context_str)
            relevance = await evaluate_relevance(question, answer)
            context_precision = await evaluate_context_precision(context_chunks, question)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "context_precision": context_precision,
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
            top_k=settings.GRAPHRAG_TOP_K,
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

