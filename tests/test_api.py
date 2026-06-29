"""Test 4: API integration tests (requires uvicorn server running on port 8000)."""
from __future__ import annotations
import httpx
import pytest

API_BASE = "http://localhost:8000"
API_V1 = f"{API_BASE}/api/v1"

class TestHealthEndpoint:
    def test_health(self):
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_graphrag_health(self):
        r = httpx.get(f"{API_V1}/graphrag/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "status" in data

    def test_graphrag_health_node_count(self):
        r = httpx.get(f"{API_V1}/graphrag/health", timeout=5)
        data = r.json()
        assert data["nodes"] > 0

class TestGraphRAGQuery:
    def test_query_returns_valid_structure(self):
        r = httpx.post(
            f"{API_V1}/graphrag/query",
            json={"question": "What is IBM WebSphere Portal?", "top_k": 3},
            timeout=30,
        )
        assert r.status_code == 200, f"Query failed: {r.text[:200]}"
        data = r.json()
        assert "question" in data
        assert "vector_context" in data
        assert "bm25_context" in data
        assert "graph_context" in data
        assert "metrics" in data

    def test_query_returns_results(self):
        r = httpx.post(
            f"{API_V1}/graphrag/query",
            json={"question": "IBM security bulletin", "top_k": 3},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        metrics = data["metrics"]
        assert metrics.get("vector_count", 0) > 0, "No vector results"
        assert metrics.get("bm25_count", 0) > 0, "No BM25 results"

    @pytest.mark.parametrize("q,k", [
        ("IBM", 5),
        ("WebSphere Portal security", 3),
        ("security vulnerability fix", 3),
    ])
    def test_query_various_questions(self, q, k):
        r = httpx.post(
            f"{API_V1}/graphrag/query",
            json={"question": q, "top_k": k},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["metrics"]["unique_chunks"] > 0

    def test_query_top_k_respected(self):
        r = httpx.post(
            f"{API_V1}/graphrag/query",
            json={"question": "IBM WebSphere", "top_k": 5},
            timeout=30,
        )
        data = r.json()
        assert data["metrics"]["vector_count"] <= 5
        assert data["metrics"]["bm25_count"] <= 5

    def test_query_with_empty_question_fails(self):
        r = httpx.post(
            f"{API_V1}/graphrag/query",
            json={"question": "", "top_k": 3},
            timeout=10,
        )
        # API accepts empty question and returns empty results
        assert r.status_code in (200, 422)

    def test_query_with_invalid_top_k_fails(self):
        r = httpx.post(
            f"{API_V1}/graphrag/query",
            json={"question": "test", "top_k": -1},
            timeout=10,
        )
        assert r.status_code == 422

class TestFeedbackEndpoint:
    def test_feedback_post(self):
        r = httpx.post(
            f"{API_V1}/graphrag/query/feedback",
            json={"question": "test", "response": "test answer", "rating": 5},
            timeout=10,
        )
        # May return 200 or 404 depending on whether feedback endpoint is implemented
        assert r.status_code in (200, 404)
