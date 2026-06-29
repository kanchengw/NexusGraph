"""GraphRAG tool for LangGraph agents.

Allows the agent to query the knowledge graph for context-aware answers.
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.core.graphrag.retriever import GraphRAGRetriever
from app.core.logging import logger


@tool
async def graphrag_search(query: str, top_k: int = 5) -> str:
    """Search the GraphRAG knowledge graph for relevant context.

    Combines vector similarity search with knowledge graph traversal.
    Use this when the user asks about technical topics related to the
    knowledge base (TechQA / technical documentation).

    Args:
        query: The search query or question.
        top_k: Number of top chunks to retrieve (default 5).

    Returns:
        Formatted context string with vector and graph results.
    """
    retriever = GraphRAGRetriever()
    try:
        result = await retriever.local_search(query, top_k)
        context_parts = []

        vec_results = result.get("vector_context", [])
        if vec_results:
            context_parts.append("=== Vector Search Results ===")
            for r in vec_results[:3]:
                text = r.get("text", "")[:300]
                title = r.get("document_title", "Unknown")
                score = r.get("score", 0)
                context_parts.append(f"[Doc: {title} | Score: {score:.3f}]\n{text}\n")

        graph_results = result.get("graph_context", [])
        if graph_results:
            context_parts.append("=== Graph Expansion Results ===")
            for r in graph_results[:3]:
                text = r.get("text", "")[:200]
                entities = r.get("source_entities", [])
                expanded = r.get("expanded_entities", [])
                context_parts.append(f"[Entities: {', '.join(entities[:5])}]\n{text}\n")

        return "\n".join(context_parts) if context_parts else "No relevant context found."
    except Exception as e:
        logger.exception("graphrag_tool_error", query=query)
        return f"Error searching knowledge graph: {e!s}"
    finally:
        await retriever.close()
