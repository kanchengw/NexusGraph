import pathlib

# Read current file, strip BOM if present
p = pathlib.Path("evals/evaluate_graphrag.py")
raw = p.read_bytes()
if raw[:3] == b"\xef\xbb\xbf":
    raw = raw[3:]
code = raw.decode("utf-8")
lines = code.split(chr(10))
print(f"Read {len(lines)} lines, BOM stripped: {raw[:3] != p.read_bytes()[:3]}")

# Fix BOM issue by rewriting
p.write_bytes(raw)
print("Rewritten without BOM")

# Now do replacements
changes = 0

# 1. Add new functions
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

# 2. Add timing
old2 = "            faithfulness = await evaluate_faithfulness(question, answer, context_str)"
new2 = "            t0 = time.time()\\n            faithfulness = await evaluate_faithfulness(question, answer, context_str)"
if old2 in code:
    code = code.replace(old2, new2)
    changes += 1
    print("2. Added timing")
else:
    print("2. WARN: old2 not found")

# 3. Add new metric calls
old3 = "            context_precision = await evaluate_context_precision(context_chunks, question)"
new3 = old3 + "\\n            answer_correctness = await evaluate_answer_correctness(question, answer, ground_truth)\\n            context_recall = await evaluate_context_recall(context_chunks, row.get(\\"documents\\", []))\\n            hit_rate = await evaluate_hit_rate(\\n                [r.get(\\"chunk_id\\", \\"\\") for r in rag_result.get(\\"vector_context\\", [])],\\n                row.get(\\"documents\\", [None])[0] if row.get(\\"documents\\") else None,\\n            )\\n            elapsed_ms = int((time.time() - t0) * 1000)"
if old3 in code:
    code = code.replace(old3, new3)
    changes += 1
    print("3. Added new metric calls")
else:
    print("3. WARN: old3 not found")

# 4-7. Simple replacements with correct quote style
old4 = '                "context_precision": context_precision,\\n            })'
new4 = '                "context_precision": context_precision,\\n                "answer_correctness": answer_correctness,\\n                "context_recall": context_recall,\\n                "hit_rate": hit_rate,\\n                "response_time_ms": elapsed_ms,\\n            })'
if old4 in code:
    code = code.replace(old4, new4)
    changes += 1
    print("4. Added results fields")
else:
    print("4. WARN: old4 not found")

old5 = """        "context_precision": sum(r["context_precision"] for r in results) / len(results) if results else 0,
    }"""
new5 = """        "context_precision": sum(r["context_precision"] for r in results) / len(results) if results else 0,
        "answer_correctness": sum(r["answer_correctness"] for r in results) / len(results) if results else 0,
        "context_recall": sum(r["context_recall"] for r in results) / len(results) if results else 0,
        "hit_rate": sum(r["hit_rate"] for r in results) / len(results) if results else 0,
        "avg_response_time_ms": sum(r["response_time_ms"] for r in results) / len(results) if results else 0,
    }"""
if old5 in code:
    code = code.replace(old5, new5)
    changes += 1
    print("5. Updated aggregate metrics")
else:
    print("5. WARN: old5 not found")

# 6. Add import time
if "import time" not in code:
    code = code.replace("from app.core.logging import logger", "import time\\nfrom app.core.logging import logger")
    changes += 1
    print("6. Added import time")

p.write_text(code, encoding="utf-8")
print(f"\\nTotal lines: {len(code.split(chr(10)))}, changes: {changes}")

try:
    compile(code, "evals/evaluate_graphrag.py", "exec")
    print("Syntax: OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")