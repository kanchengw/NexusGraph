import pathlib
import re

p = pathlib.Path("evals/evaluate_graphrag.py")
code = p.read_text(encoding="utf-8")

changes = 0

# 1. Add new eval functions after context_precision return
old1 = "    return relevant_count / min(len(relevant_chunks), 5)"
new1 = """
async def evaluate_answer_correctness(question: str, answer: str, ground_truth: str) -> float:
    prompt = f"You are an evaluation judge. Rate the correctness of the answer compared to the ground truth.\\n\\nQuestion: {question}\\n\\nGround Truth: {ground_truth}\\n\\nAnswer: {answer}\\n\\nScore from 0 to 1 based on how factually correct the answer is.\\nReturn ONLY a number between 0 and 1.\\nCorrectness score:"
    response = await JUDGE_LLM.ainvoke(prompt)
    try:
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        logger.warning("eval_score_parse_error", metric="answer_correctness", raw=response.content)
        return 0.0


async def evaluate_context_recall(retrieved_chunks: list[str], ground_truth_docs: list[str]) -> float:
    if not ground_truth_docs or not retrieved_chunks:
        return 0.0
    relevant = sum(1 for c in retrieved_chunks if any(str(d)[:50].lower() in c.lower() for d in ground_truth_docs[:3]))
    return min(1.0, relevant / min(len(ground_truth_docs), 3))


async def evaluate_hit_rate(retrieved_ids: list[str], relevant_id: str | None) -> float:
    if not relevant_id:
        return 0.0
    return 1.0 if relevant_id in retrieved_ids else 0.0
"""
if old1 in code:
    code = code.replace(old1, old1 + new1)
    changes += 1
    print("1. Added new eval functions")
else:
    print("1. WARN: old1 not found")

# 2. Add timing + new metrics in eval loop
old2 = "            faithfulness = await evaluate_faithfulness(question, answer, context_str)"
new2 = """            t0 = time.time()
            faithfulness = await evaluate_faithfulness(question, answer, context_str)"""
if old2 in code:
    code = code.replace(old2, new2)
    changes += 1
    print("2. Added timing")
else:
    print("2. WARN: old2 not found")

# 3. Add context_precision line after faithfulness/relevance
old3 = "            context_precision = await evaluate_context_precision(context_chunks, question)"
new3 = old3 + """
            answer_correctness = await evaluate_answer_correctness(question, answer, ground_truth)
            context_recall = await evaluate_context_recall(context_chunks, row.get("documents", []))
            hit_rate = await evaluate_hit_rate(
                [r.get("chunk_id", "") for r in rag_result.get("vector_context", [])],
                row.get("documents", [None])[0] if row.get("documents") else None,
            )
            elapsed_ms = int((time.time() - t0) * 1000)"""
if old3 in code:
    code = code.replace(old3, new3)
    changes += 1
    print("3. Added new metric calls")
else:
    print("3. WARN: old3 not found")

# 4. Add new fields to results dict
old4 = "                'context_precision': context_precision,"
new4 = old4 + """
                'answer_correctness': answer_correctness,
                'context_recall': context_recall,
                'hit_rate': hit_rate,
                'response_time_ms': elapsed_ms,"""
if old4 in code:
    code = code.replace(old4, new4)
    changes += 1
    print("4. Added results fields")
else:
    print("4. WARN: old4 not found")

# 5. Update aggregate metrics
old5 = "        'context_precision': sum(r['context_precision'] for r in results) / len(results) if results else 0,"
new5 = old5 + """
        'answer_correctness': sum(r['answer_correctness'] for r in results) / len(results) if results else 0,
        'context_recall': sum(r['context_recall'] for r in results) / len(results) if results else 0,
        'hit_rate': sum(r['hit_rate'] for r in results) / len(results) if results else 0,
        'avg_response_time_ms': sum(r['response_time_ms'] for r in results) / len(results) if results else 0,"""
if old5 in code:
    code = code.replace(old5, new5)
    changes += 1
    print("5. Updated aggregate metrics")
else:
    print("5. WARN: old5 not found")

# 6. Add import time if not present
if "import time" not in code:
    code = code.replace("from app.core.logging import logger", "import time\nfrom app.core.logging import logger")
    changes += 1
    print("6. Added import time")
else:
    print("6. import time already exists")

# 7. Update EvalResult
old7 = "            context_precision=metrics['context_precision'],"
new7 = old7 + """
            answer_correctness=metrics['answer_correctness'],
            context_recall=metrics['context_recall'],
            hit_rate=metrics['hit_rate'],
            avg_response_time_ms=metrics['avg_response_time_ms'],"""
if old7 in code:
    code = code.replace(old7, new7)
    changes += 1
    print("7. Updated EvalResult")
else:
    print("7. WARN: old7 not found")

p.write_text(code, encoding="utf-8")
lines = code.split(chr(10))
print(f"\\nTotal lines: {len(lines)}, changes: {changes}")

try:
    compile(code, "evals/evaluate_graphrag.py", "exec")
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")