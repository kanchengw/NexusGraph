"""Index Agent - LangGraph state and graph for KB ingestion."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from app.core.logging import logger

class IndexState(BaseModel):
    split: str = "train"
    max_docs: int = 100
    skip_entities: bool = False
    reset: bool = False
    status: str = "idle"
    chunks_created: int = 0
    entities_created: int = 0
    error: str = ""

def create_index_graph():
    from langgraph.graph import StateGraph, END
    from app.core.graphrag.indexer import KnowledgeBaseIndexer
    
    async def load_and_index_node(state: IndexState) -> dict:
        try:
            indexer = KnowledgeBaseIndexer()
            if state.reset:
                from app.core.graphrag.models import clear_database, init_neo4j_schema
                await clear_database()
                await init_neo4j_schema()
            
            await indexer.index_all(
                split=state.split, 
                extract_entities=not state.skip_entities,
                max_entity_docs=state.max_docs if not state.skip_entities else 0
            )
            return {"status": "completed", "chunks_created": state.max_docs * 10}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    builder = StateGraph(IndexState)
    builder.add_node("index", load_and_index_node)
    builder.set_entry_point("index")
    builder.add_edge("index", END)
    return builder.compile()

index_graph = create_index_graph()

async def run_index_agent(split="train", max_docs=100, skip_entities=False, reset=False):
    state = IndexState(split=split, max_docs=max_docs, skip_entities=skip_entities, reset=reset)
    result = await index_graph.ainvoke(state)
    return result
