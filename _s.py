import pathlib

c = pathlib.Path("plan.md").read_text(encoding="utf-8")
lines = c.split(chr(10))
print(f"Total lines: {len(lines)}")

# Find section headers
for i, l in enumerate(lines):
    l2 = l.strip()
    if l2.startswith("##") or l2.startswith("###") or l2.startswith("#"):
        print(f"  L{i+1}: {l2}")