import pathlib

# Read current evaluator
p = pathlib.Path("evals/evaluate_graphrag.py")
code = p.read_text(encoding="utf-8")
lines = code.split(chr(10))
print(f"Current: {len(lines)} lines")
# Find the evaluate_faithfulness function
for i, l in enumerate(lines):
    if "def evaluate_" in l or "async def evaluate_" in l:
        print(f"  L{i+1}: {l.strip()}")
    if "async def run_evaluation" in l:
        print(f"  L{i+1}: {l.strip()}")
    if "compare_with_baseline" in l:
        print(f"  L{i+1}: {l.strip()}")