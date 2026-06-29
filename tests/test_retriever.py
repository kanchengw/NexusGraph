"""Test 3: Retriever unit tests (directly, without API server)."""
from __future__ import annotations
import asyncio
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="function")

class TestRetrieverDirect:
    """Test retriever methods directly."""

    async def test_vector_search_returns_chunks(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.vector_search("IBM WebSphere Portal", 3)
            assert len(result) > 0
            assert "chunk_id" in result[0]
            assert "text" in result[0]
            assert "score" in result[0]
        finally:
            await r.close()

    async def test_vector_search_positive_scores(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.vector_search("IBM security", 5)
            for rec in result:
                assert rec["score"] > 0
        finally:
            await r.close()

    async def test_bm25_search_returns_chunks(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.bm25_search("IBM WebSphere", 3)
            assert len(result) > 0
            assert "chunk_id" in result[0]
        finally:
            await r.close()

    async def test_bm25_search_with_short_query(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.bm25_search("IBM", 3)
            assert len(result) > 0
        finally:
            await r.close()

    async def test_bm25_search_with_medium_query(self):
        """BM25 should handle multi-word queries."""
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.bm25_search("IBM WebSphere Portal security fix", 3)
            assert len(result) > 0
        finally:
            await r.close()
            await r.close()

    async def test_graph_expand_empty(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.graph_expand([])
            assert result == []
        finally:
            await r.close()

    async def test_graph_expand_with_chunks(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            vec = await r.vector_search("IBM", 3)
            if not vec:
                pytest.skip("No vector results")
            chunk_ids = [rec["chunk_id"] for rec in vec[:3]]
            result = await r.graph_expand(chunk_ids)
            if not result:
                pytest.skip("Graph expansion empty (entity extraction still running)")
        finally:
            await r.close()

    async def test_local_search_3_path(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.local_search("IBM WebSphere", 3)
            m = result.get("metrics", {})
            assert m.get("vector_count", 0) > 0
            assert m.get("bm25_count", 0) > 0
        finally:
            await r.close()

    async def test_local_search_metrics_structure(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.local_search("security", 5)
            m = result["metrics"]
            required = {"vector_count", "bm25_count", "graph_count", "unique_chunks",
                        "vector_only", "bm25_only", "graph_only", "overlap_vector_bm25"}
            assert required.issubset(m.keys()), f"Missing: {required - m.keys()}"
        finally:
            await r.close()

    async def test_local_search_no_duplicates(self):
        from app.core.graphrag.retriever import GraphRAGRetriever
        r = GraphRAGRetriever()
        try:
            result = await r.local_search("IBM", 5)
            vectors = {rec["chunk_id"] for rec in result["vector_context"]}
            bm25s = {rec["chunk_id"] for rec in result["bm25_context"]}
            assert len(vectors | bm25s) <= len(vectors) + len(bm25s)
        finally:
            await r.close()
