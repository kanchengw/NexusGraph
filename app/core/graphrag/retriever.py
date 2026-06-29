"""GraphRAG retrieval pipeline.

3-path retrieval: vector search + BM25 keyword search + graph expansion.
Each path returns results + metrics for multi-path attribution.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.graphrag.models import get_neo4j_driver
from app.core.logging import logger
from app.services.embeddings import embedding_service


FULLTEXT_INDEX_NAME = "chunk_text_ft"

class GraphRAGRetriever:
    """Retriever with 3-path fusion: vector + BM25 + graph expand."""

    def __init__(self) -> None:
        self.driver = None

    async def _ensure_driver(self):
        if self.driver is None:
            self.driver = await get_neo4j_driver()
        return self.driver

    async def vector_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Path 1: Vector similarity search on Chunk nodes."""
        k = top_k or settings.GRAPHRAG_TOP_K
        query_embedding = await embedding_service.embed_query(query)
        driver = await self._ensure_driver()
        async with driver.session(database="neo4j") as session:
            result = await session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $k, $embedding)
                YIELD node AS chunk, score
                OPTIONAL MATCH (chunk)-[:PART_OF]->(d:Document)
                RETURN chunk.id AS chunk_id,
                       chunk.text AS text,
                       chunk.chunk_index AS chunk_index,
                       d.id AS document_id,
                       d.title AS document_title,
                       score
                ORDER BY score DESC
                """,
                index_name=settings.GRAPHRAG_VECTOR_INDEX_NAME,
                k=k,
                embedding=query_embedding,
            )
            records = await result.data()
        return records

    async def bm25_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Path 2: BM25 keyword search via Neo4j fulltext index."""
        k = top_k or settings.GRAPHRAG_TOP_K
        driver = await self._ensure_driver()
        # Clean query for fulltext search: filter stop words, use default OR scoring
        STOP_WORDS = {"what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "or", "on", "at", "by", "with", "from", "as", "be", "this", "that", "it", "are", "was", "were", "been", "do", "does", "did", "will", "would", "can", "could", "may", "might", "shall", "should", "has", "have", "had", "how", "why", "when", "where", "which", "who", "whom"}
        terms = [t for t in query.split() if t.lower() not in STOP_WORDS][:10]
        clean_terms = " OR ".join(terms) if len(terms) > 1 else (terms[0] if terms else query)
        async with driver.session(database="neo4j") as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes($index_name, $query_text)
                    YIELD node AS chunk, score
                    OPTIONAL MATCH (chunk)-[:PART_OF]->(d:Document)
                    RETURN chunk.id AS chunk_id,
                           chunk.text AS text,
                           chunk.chunk_index AS chunk_index,
                           d.id AS document_id,
                           d.title AS document_title,
                           score
                    ORDER BY score DESC
                    LIMIT $k
                    """,
                    index_name=FULLTEXT_INDEX_NAME,
                    query_text=clean_terms,
                    k=k,
                )
                records = await result.data()
            except Exception as e:
                logger.warning("bm25_search_failed", error=str(e))
                records = []
        return records

    async def graph_expand(self, chunk_ids: list[str], max_hops: int = 2) -> list[dict[str, Any]]:
        """Path 3: Graph expansion from chunks through entities."""
        driver = await self._ensure_driver()
        async with driver.session(database="neo4j") as session:
            if not chunk_ids:
                return []
            result = await session.run(
                f"""
                MATCH (c:Chunk)
                WHERE c.id IN $chunk_ids
                MATCH (c)<-[:FROM_CHUNK]-(e:Entity)
                OPTIONAL MATCH (e)-[:RELATES_TO*1..{max_hops}]-(related:Entity)
                OPTIONAL MATCH (related)<-[:FROM_CHUNK]-(related_chunk:Chunk)
                RETURN DISTINCT
                    related_chunk.id AS chunk_id,
                    related_chunk.text AS text,
                    related_chunk.chunk_index AS chunk_index,
                    collect(DISTINCT e.name) AS source_entities,
                    collect(DISTINCT related.name) AS expanded_entities
                LIMIT 50
                """,
                chunk_ids=chunk_ids,
            )
            records = await result.data()
        return records

    async def local_search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """3-path fusion: vector + BM25 + graph expand with per-path metrics."""
        k = top_k or settings.GRAPHRAG_TOP_K

        # Path 1: Vector search
        vector_results = await self.vector_search(query, k)

        # Path 2: BM25 keyword search
        bm25_results = await self.bm25_search(query, k)

        # Path 3: Graph expand from top vector chunks
        chunk_ids = [r["chunk_id"] for r in vector_results[:3]]
        graph_results = await self.graph_expand(chunk_ids)

        # Compute per-path metrics
        vector_ids = {r["chunk_id"] for r in vector_results}
        bm25_ids = {r["chunk_id"] for r in bm25_results}
        graph_ids = {r["chunk_id"] for r in graph_results if r.get("chunk_id")}
        all_ids = vector_ids | bm25_ids | graph_ids

        metrics = {
            "vector_count": len(vector_results),
            "bm25_count": len(bm25_results),
            "graph_count": len(graph_results),
            "unique_chunks": len(all_ids),
            "vector_only": len(vector_ids - bm25_ids - graph_ids),
            "bm25_only": len(bm25_ids - vector_ids - graph_ids),
            "graph_only": len(graph_ids - vector_ids - bm25_ids),
            "overlap_vector_bm25": len(vector_ids & bm25_ids),
        }

        return {
            "vector_context": vector_results,
            "bm25_context": bm25_results,
            "graph_context": graph_results,
            "metrics": metrics,
        }

    async def close(self) -> None:
        if self.driver:
            await self.driver.close()
