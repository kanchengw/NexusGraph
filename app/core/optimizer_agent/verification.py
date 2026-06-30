"""Verification report generation for optimization changes.
Generates before/after comparison report for human review (HIL B).
"""
from __future__ import annotations
import json
import os
from typing import Any
from app.core.logging import logger


def load_eval_metrics(path: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics", {})


def generate_verification_report(
    before_path: str,
    after_path: str,
    bad_case_count: int = 0,
    good_case_count: int = 0,
) -> dict[str, Any]:
    """Generate before/after comparison report for human review.
    Does NOT decide verdict - that is HIL B decision.
    """
    before = load_eval_metrics(before_path)
    after = load_eval_metrics(after_path)
    metrics = set(before.keys()) | set(after.keys())
    deltas = {}
    for m in sorted(metrics):
        b = before.get(m, 0.0)
        a = after.get(m, 0.0)
        deltas[m] = {"before": b, "after": a, "delta": round(a - b, 4)}
    report = {
        "status": "awaiting_review",
        "sample_sizes": {
            "bad_cases": bad_case_count,
            "good_cases": good_case_count,
            "total": bad_case_count + good_case_count,
        },
        "aggregate_comparison": deltas,
    }
    os.makedirs("evals/reports", exist_ok=True)
    report_path = "evals/reports/verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("verification_report_generated", path=report_path)
    return report


def check_verdict(verdict: str, report_path: str = "evals/reports/verification_report.json") -> bool:
    """Write human verdict to the verification report.
    Called by HIL B (Offline Admin UI or CLI).
    """
    if not os.path.exists(report_path):
        return False
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    report["status"] = "reviewed"
    report["human_verdict"] = verdict
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("verdict_recorded", verdict=verdict)
    return True


def read_verdict(report_path: str = "evals/reports/verification_report.json") -> str | None:
    """Read human verdict from verification report. Returns None if not yet reviewed."""
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if report.get("status") == "reviewed":
        return report.get("human_verdict")
    return None
