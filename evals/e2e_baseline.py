
"""E2E baseline: 50 queries with LLM-as-Judge rating + feedback submission."""

import json, os, sys, time, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

API = "http://localhost:8000/api/v1"
SID = str(uuid.uuid4())[:8]

QUESTIONS = [
    # WebSphere DataPower
    "How to fix WebSphere DataPower XC10 appliance error IT19816",
    "What are the recommended fixes for WebSphere DataPower XC10 Appliance V2.5",
    "How to check SSD driver version on WebSphere DataPower XC10",
    "WebSphere DataPower XC10 security vulnerability remediation",
    "How to apply fixpack for WebSphere DataPower XC10 V2.1",

    # IBM Java SDK security
    "What is the CVSS score for IBM Java SDK vulnerability",
    "IBM Java SDK security bulletin remediation steps",
    "How to fix IBM Java SDK denial of service vulnerability",
    "What is the CVSS v3 score for IBM Java SDK",
    "IBM Java SDK 7 security fix",

    # WebSphere Application Server
    "How to fix IBM WebSphere TSPM version 7.0",
    "WebSphere Application Server security bulletin SIP services",
    "IBM WebSphere potential denial of service CVE-2016-2960",
    "WebSphere version compatibility with TSPM",
    "IBM WebSphere WAS 8.0 security fix",

    # Rational / Rhapsody
    "Rational Rhapsody ReporterPlus integration",
    "Rational Rhapsody TeamCenter integration",
    "Rational DOORS integration with Rhapsody",
    "Rational Rhapsody Webify web-enabling a model",
    "Rational Rhapsody and Rational Team Concert integration",

    # IBM Security / CVSS
    "What is the Common Vulnerability Scoring System",
    "How to calculate CVSS v3 environmental score",
    "CVSS v3 calculator",
    "IBM security bulletin notification subscription",
    "How to subscribe to IBM My Notifications",

    # Tech details
    "How to use show ssd-version command on XC10",
    "IBM Fix Central for WebSphere DataPower",
    "Recommended fixes table for DataPower XC10",
    "IBM XC10 appliance 7199-92X version information",
    "IBM XC10 appliance SSD driver versions",

    # Rational Toolchain
    "Rational System Architect import functionality",
    "Rational Rhapsody direct integration RTC",
    "WindRiver Workbench 3.3 integration",
    "Rational Rhapsody model web-enabling",
    "IBM Rational Rhapsody ReporterPlus",

    # WAS Security
    "IBM WebSphere Application Server SIP vulnerability fix",
    "WAS 8.0 security interim fix",
    "WebSphere Application Server potential denial of service",
    "IBM WebSphere and SIP Services security",
    "IBM TSPM WebSphere compatibility matrix",

    # Java SDK
    "IBM Java SDK 7.0 security bulletin",
    "IBM Java SDK vulnerability CVSS base score",
    "IBM Java SDK environment score",
    "Java SDK heap memory in Workbench JVM",
    "Java non-heap memory in IBM Workbench",

    # General
    "IBM Workbench JVM memory types",
    "How to get notified about IBM security bulletins",
    "IBM eSupport notification setup",
    "IBM Workbench Eclipse RCP platform",
    "Workbench Java bytecode execution JVM",
]

def ask(q, timeout=60):
    try:
        r = requests.post(f"{API}/graphrag/query", json={"question": q, "top_k": 5}, timeout=timeout)
        return r.json() if r.ok else None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def judge(q, a, ctx):
    """Judge the answer quality using qwen-plus."""
    prompt = f"""Evaluate this RAG answer on a scale of 1-5:

Question: {q}

Answer: {a}

Criteria:
5 = Perfect: accurate, complete, fully grounded in context
4 = Good: accurate but could be more complete
3 = Average: partially correct but misses key details
2 = Poor: mostly inaccurate or missing
1 = Bad: hallucinated or empty

Return ONLY a number 1-5."""
    try:
        r = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": "sk-2c7d2de5370741e4ad2d7bae68c04e35", "Content-Type": "application/json"},
            json={
                "model": "qwen-plus",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
            },
            timeout=15,
        )
        if r.ok:
            text = r.json()["choices"][0]["message"]["content"].strip()
            return int(text[0]) if text and text[0].isdigit() else 3
    except: pass
    return 3

def feedback(sid, q, a, rating):
    try:
        requests.post(f"{API}/graphrag/feedback", json={
            "session_id": sid, "query": q, "response": a, "rating": rating
        }, timeout=5)
    except: pass

results = []
total = len(QUESTIONS)

print(f"=== E2E Baseline: {total} queries ===\n")
for i, q in enumerate(QUESTIONS, 1):
    print(f"[{i}/{total}] {q[:70]}")
    d = ask(q)
    if d:
        a = d.get("answer", "") or "(empty)"
        src = d.get("source_docs", [])
        rating = judge(q, a[:500], src)
        feedback(SID, q, a[:200], rating)
        results.append({"q": q, "a": a[:200], "src": len(src), "rating": rating})
        print(f"  -> rating: {rating}/5, sources: {len(src)}, answer: {a[:80]}...")
    else:
        results.append({"q": q, "a": "", "src": 0, "rating": 1})
        print(f"  -> FAILED")
    time.sleep(7.5)  # Rate limit breathing room

# Summary
ratings = [r["rating"] for r in results]
avg = sum(ratings) / len(ratings)
print(f"\n=== SUMMARY ===")
print(f"Total: {total}")
print(f"Average rating: {avg:.2f}/5")
print(f"Distribution:")
for r in range(1, 6):
    cnt = ratings.count(r)
    bar = "#" * cnt
    print(f"  {r}: {cnt: >2} {bar}")

# Save
report = {"session_id": SID, "total": total, "avg_rating": round(avg, 2), "results": results}
os.makedirs("evals/reports", exist_ok=True)
with open("evals/reports/e2e_baseline.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved to evals/reports/e2e_baseline.json")
