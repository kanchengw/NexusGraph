import pathlib
p = pathlib.Path("app/core/graphrag/retriever.py")
code = p.read_text(encoding="utf-8")
# Find local_search return
lines = code.split(chr(10))
for i, l in enumerate(lines):
    if "def local_search" in l or "return {" in l or "metrics" in l:
        print(f"L{i+1}: {l}")
    if i > 0 and "response_time" in lines[i-1]:
        print(f"L{i}: {l}")
    if "response_time" in l:
        print(f"L{i+1}: {l}")