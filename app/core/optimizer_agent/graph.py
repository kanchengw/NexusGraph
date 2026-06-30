"""Optimizer Agent - LangGraph state and graph with HIL."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from app.core.logging import logger

class OptimizerState(BaseModel):
    days: int = 7
    status: str = "idle"  # idle, analyzing, waiting_approval, applying, done, failed
    report: dict[str, Any] = Field(default_factory=dict)
    suggestion: dict[str, Any] = Field(default_factory=dict)
    approved: bool = False
    error: str = ""

def create_optimizer_graph():
    from langgraph.graph import StateGraph, END
    from langgraph.types import Command
    
    async def analyze_node(state: OptimizerState) -> dict:
        from app.core.optimizer_agent.analyzer import generate_all_reports
        try:
            report = generate_all_reports(days=state.days)
            return {"status": "analyzing", "report": report}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def optimize_node(state: OptimizerState) -> dict:
        from app.core.optimizer_agent.optimizer import run_llm_analysis
        try:
            suggestion = run_llm_analysis(state.report)
            return {"status": "waiting_approval", "suggestion": suggestion}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def wait_approval_node(state: OptimizerState) -> Command:
        """HIL: pause and wait for human approval."""
        from app.core.optimizer_agent.optimizer import print_report
        print_report(state.suggestion)
        print("\nApprove? Run: python -c 'from app.core.optimizer_agent import approve; approve()'")
        raise __import__('langgraph').errors.GraphInterrupt(
            "Waiting for human approval of optimization suggestion"
        )
    
    async def apply_node(state: OptimizerState) -> dict:
        from app.core.optimizer_agent.optimizer import apply_suggestion
        try:
            result = apply_suggestion()
            return {"status": "applying", "error": ""}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    builder = StateGraph(OptimizerState)
    builder.add_node("analyze", analyze_node)
    builder.add_node("optimize", optimize_node)
    builder.add_node("wait_approval", wait_approval_node)
    builder.add_node("apply", apply_node)
    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "optimize")
    builder.add_edge("optimize", "wait_approval")
    builder.add_conditional_edges("wait_approval", 
        lambda s: "apply" if s.approved else "wait_approval")
    builder.add_edge("apply", END)
    return builder.compile()

optimizer_graph = create_optimizer_graph()

def approve():
    """Approve the pending optimization suggestion."""
    # Simple: updates the suggestion file with approved flag
    import json, os
    path = "evals/reports/optimization_suggestion.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            sug = json.load(f)
        sug["approved"] = True
        sug["approved_at"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat()
        with open(path, "w") as f:
            json.dump(sug, f, indent=2)
        logger.info("optimization_approved")
        print("Approved. Run flywheel to apply.")
