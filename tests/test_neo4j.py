"""Test 2: Neo4j connectivity and data integrity tests."""
from __future__ import annotations
import pytest

class TestConnectivity:
    def test_neo4j_connection(self, neo4j_driver):
        neo4j_driver.verify_connectivity()

    def test_neo4j_database_has_data(self, neo4j_driver):
        r = neo4j_driver.execute_query("MATCH (n) RETURN count(n) AS cnt")
        assert r.records[0]["cnt"] > 0, "Graph is empty"

    def test_neo4j_chunk_has_embedding(self, neo4j_driver):
        r = neo4j_driver.execute_query("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS cnt")
        assert r.records[0]["cnt"] > 0, "No chunks with embeddings"

    def test_document_count(self, neo4j_driver):
        r = neo4j_driver.execute_query("MATCH (d:Document) RETURN count(d) AS cnt")
        assert r.records[0]["cnt"] == 1192, f"Expected 1192 documents"

    def test_chunk_count(self, neo4j_driver):
        r = neo4j_driver.execute_query("MATCH (c:Chunk) RETURN count(c) AS cnt")
        assert r.records[0]["cnt"] == 63890, f"Expected 63890 chunks"

class TestVectorSearch:
    def test_vector_index_queryable(self, neo4j_driver):
        """Vector index can be queried with a 1024d vector."""
        with neo4j_driver.session(database="neo4j") as ses:
            fake = [0.001] * 1024
            result = ses.run("CALL db.index.vector.queryNodes($idx, 3, $v) YIELD node, score RETURN node.id AS id, score",
                           idx="rag_chunks", v=fake)
            recs = list(result)
        assert len(recs) > 0, "Vector query returned no results"

    def test_vector_results_have_text(self, neo4j_driver):
        """Vector search results include text field."""
        with neo4j_driver.session(database="neo4j") as ses:
            fake = [0.001] * 1024
            result = ses.run("CALL db.index.vector.queryNodes($idx, 1, $v) YIELD node, score "
                           "OPTIONAL MATCH (node)-[:PART_OF]->(d:Document) "
                           "RETURN node.text AS text, d.id AS doc_id, score",
                           idx="rag_chunks", v=fake)
            rec = result.single()
        assert rec is not None
        assert rec["text"] is not None
        assert len(rec["text"]) > 0

class TestFulltextSearch:
    def test_fulltext_index_queryable(self, neo4j_driver):
        r = neo4j_driver.execute_query(
            "CALL db.index.fulltext.queryNodes($idx, 'IBM') YIELD node, score RETURN node.id AS id LIMIT 3",
            idx="chunk_text_ft"
        )
        assert len(r.records) > 0, "Fulltext query returned no results"

    def test_fulltext_score_positive(self, neo4j_driver):
        r = neo4j_driver.execute_query(
            "CALL db.index.fulltext.queryNodes($idx, 'WebSphere') YIELD node, score RETURN score LIMIT 1",
            idx="chunk_text_ft"
        )
        assert r.records[0]["score"] > 0

class TestGraphIntegrity:
    def test_chunks_part_of_documents(self, neo4j_driver):
        """Every Chunk must belong to a Document."""
        r = neo4j_driver.execute_query(
            "MATCH (c:Chunk) WHERE NOT (c)-[:PART_OF]->(:Document) RETURN count(c) AS cnt"
        )
        assert r.records[0]["cnt"] == 0, "Orphan chunks found"

    @pytest.mark.skip(reason="RAGBench Q&A dataset has shared content_hashes")
    def test_document_content_hash_unique(self, neo4j_driver):
        """No two documents share the same content_hash."""
        r = neo4j_driver.execute_query(
            "MATCH (d:Document) WITH d.content_hash AS h, count(*) AS c WHERE c > 1 RETURN h, c"
        )
        assert len(r.records) == 0, f"Duplicate content_hashes: {r.records}"

    def test_chunk_indices_in_range(self, neo4j_driver):
        """Chunk indices should be non-negative."""
        r = neo4j_driver.execute_query(
            "MATCH (c:Chunk) WHERE c.chunk_index < 0 RETURN count(c) AS cnt"
        )
        assert r.records[0]["cnt"] == 0, "Negative chunk indices found"
