"""Judge Agent - LangGraph state and graph."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

class JudgeState(BaseModel):
    split: str = "test"
    num_samples: int = 50
    status: str = "idle"
    results: dict[str, Any] = Field(default_factory=dict)
    error: str = ""

def create_judge_graph():
    """Build the Judge Agent graph."""
    from langgraph.graph import StateGraph, END
    
    async def run_eval_node(state: JudgeState) -> dict:
        from app.core.judge_agent.evaluator import run_evaluation
        try:
            report = await run_evaluation(split=state.split, num_samples=state.num_samples)
            return {"status": "completed", "results": report, "error": ""}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    builder = StateGraph(JudgeState)
    builder.add_node("evaluate", run_eval_node)
    builder.set_entry_point("evaluate")
    builder.add_edge("evaluate", END)
    return builder.compile()

judge_graph = create_judge_graph()
