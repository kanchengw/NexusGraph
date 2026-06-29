"""GraphRAG API routes.

Exposes indexing and query endpoints for the GraphRAG knowledge graph.
"""

import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, create_engine

from app.core.graphrag.models import clear_database
from app.core.graphrag.retriever import GraphRAGRetriever
from app.core.logging import logger
from app.core.observability import get_langfuse
from app.models.retrieval_metric import RetrievalMetric

_METRIC_DB_URL = "postgresql://myuser:mypassword@localhost:5432/mydb"
_metric_engine = create_engine(_METRIC_DB_URL, pool_pre_ping=True)

router = APIRouter(prefix="/graphrag", tags=["GraphRAG"])


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=50, description="Chunks per path")


class QueryResponse(BaseModel):
    question: str
    answer: str = ""
    source_docs: list[str] = []
    vector_context: list[dict] = []
    bm25_context: list[dict] = []
    graph_context: list[dict] = []
    metrics: dict = {}


@router.post("/query", response_model=QueryResponse)
async def query_graphrag(request: QueryRequest) -> QueryResponse:
    """Query with 3-path retrieval: vector + BM25 + graph expand."""
    retriever = GraphRAGRetriever()
    start_time = time.monotonic()

    # Generate global trace_id for Online/Offline correlation
    trace_id = str(uuid.uuid4())

    # Langfuse trace
    lf = get_langfuse()
    try:
        trace = lf.trace(id=trace_id, name="graphrag_query", input=request.question) if lf else None
    except Exception:
        trace = None

    try:
        result = await retriever.local_search(request.question, request.top_k)
        metrics = result.get("metrics", {})

        # Generate answer from retrieved context
        ctx_parts = []
        for ctx_key in ["vector_context", "bm25_context", "graph_context"]:
            for chunk in result.get(ctx_key, [])[:3]:
                text = chunk.get("text", chunk.get("content", ""))
                if text:
                    ctx_parts.append(text)
        context_text = "\n---\n".join(ctx_parts[:15])

        if context_text:
            from app.services.llm.registry import LLMRegistry
            from app.core.config import settings
            llm = LLMRegistry.get(settings.DEFAULT_LLM_MODEL)
            # Collect doc titles for display
            doc_titles = []
            for ctx_key in ["vector_context", "bm25_context", "graph_context"]:
                for chunk in result.get(ctx_key, []):
                    title = chunk.get("document_title", chunk.get("source", ""))
                    if title and title not in doc_titles:
                        doc_titles.append(title)
            result["source_docs"] = doc_titles[:10]

            answer = llm.invoke(
                f"Context:\n{context_text}\n\n"
                f"Question: {request.question}\n\n"
                f"Instructions: Answer based ONLY on the context above. "
                f"If the context does not contain relevant information to answer the question, "
                f"say 'The knowledge base does not contain information about this topic.' "
                f"Do not make up answers. Do not guess.\n\n"
                f"Answer:"
            ).content
            result["answer"] = answer
        else:
            result["answer"] = "(no relevant context found)"
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(
            "graphrag_query",
            question=request.question,
            metrics=metrics,
            response_time_ms=elapsed_ms,
        )

        # Langfuse: log retrieval metrics as trace span
        if trace:
            trace.update(
                output=f"vector={metrics.get('vector_count',0)}, bm25={metrics.get('bm25_count',0)}, graph={metrics.get('graph_count',0)}",
                metadata={
                    "retrieval_metrics": metrics,
                    "response_time_ms": elapsed_ms,
                    "retrieval_type": "3_path_fusion",
                },
            )
            # Create per-path spans
            trace.span(name="vector_search", input=request.question, output=f"{metrics.get('vector_count',0)} chunks")
            trace.span(name="bm25_search", input=request.question, output=f"{metrics.get('bm25_count',0)} chunks")
            trace.span(name="graph_expand", input=str(request.question)[:50], output=f"{metrics.get('graph_count',0)} chunks")

        # Write retrieval metrics to PostgreSQL
        try:
            rm = RetrievalMetric(
                trace_id=trace_id,
                query=request.question[:200],
                answer=result.get("answer", "")[:500],
                vector_count=metrics.get("vector_count", 0),
                bm25_count=metrics.get("bm25_count", 0),
                graph_count=metrics.get("graph_count", 0),
                unique_chunks=metrics.get("unique_chunks", 0),
                vector_only=metrics.get("vector_only", 0),
                bm25_only=metrics.get("bm25_only", 0),
                graph_only=metrics.get("graph_only", 0),
                overlap_vector_bm25=metrics.get("overlap_vector_bm25", 0),
                response_time_ms=elapsed_ms,
            )
            with Session(_metric_engine) as ses:
                ses.add(rm)
                ses.commit()
        except Exception as e:
            logger.warning("metric_write_failed", error=str(e))

        return QueryResponse(
            question=request.question,
            answer=result.get("answer", ""),
            source_docs=result.get("source_docs", []),
            vector_context=result.get("vector_context", []),
            bm25_context=result.get("bm25_context", []),
            graph_context=result.get("graph_context", []),
            metrics=metrics,
        )
    except Exception as e:
        logger.exception("graphrag_query_failed", question=request.question)
        if trace:
            trace.update(status="error", metadata={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await retriever.close()


@router.post("/clear")
async def clear_knowledge_graph() -> dict:
    """Clear all data from the knowledge graph."""
    await clear_database()
    return {"status": "cleared", "message": "Knowledge graph cleared"}


@router.get("/health")
async def graphrag_health() -> dict:
    """Check GraphRAG service health."""
    retriever = GraphRAGRetriever()
    try:
        driver = await retriever._ensure_driver()
        async with driver.session(database="neo4j") as session:
            result = await session.run("MATCH (n) RETURN count(n) AS count")
            record = await result.single()
            count = record["count"] if record else 0
        return {"status": "healthy", "nodes": count}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
    finally:
        await retriever.close()
