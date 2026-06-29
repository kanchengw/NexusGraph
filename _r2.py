import pathlib
p = pathlib.Path("evals/evaluate_graphrag.py")
lines = p.read_text(encoding="utf-8").split(chr(10))
for i, l in enumerate(lines):
    if i >= 85 and i <= 170:
        print(f"L{i+1}: {l}")