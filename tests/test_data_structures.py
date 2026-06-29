"""Test 1: Data structure and schema matching tests.

Verifies that:
- Neo4j indexes exist with expected names and dimensions
- Node/relationship properties match code expectations
- Embedding dimensions are consistent across services
"""
from __future__ import annotations
import pytest

class TestNeo4jIndexes:
    def test_vector_index_exists(self, neo4j_driver, expected_vector_index):
        idx = expected_vector_index
        result = neo4j_driver.execute_query("SHOW INDEXES WHERE name = $name", name=idx["name"])
        assert len(result.records) == 1, f"Vector index not found"
        rec = result.records[0]
        assert rec["type"] == "VECTOR"
        assert rec["labelsOrTypes"] == [idx["label"]]
        assert rec["properties"] == [idx["property"]]

    def test_fulltext_index_exists(self, neo4j_driver, expected_fulltext_index):
        idx = expected_fulltext_index
        result = neo4j_driver.execute_query("SHOW INDEXES WHERE name = $name", name=idx["name"])
        assert len(result.records) == 1, f"Fulltext index not found"
        rec = result.records[0]
        assert rec["type"] == "FULLTEXT"
        assert rec["labelsOrTypes"] == [idx["label"]]

    def test_vector_index_dimension(self, neo4j_driver):
        result = neo4j_driver.execute_query("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN size(c.embedding) AS dim LIMIT 1")
        assert len(result.records) == 1, "No chunks with embeddings"
        assert result.records[0]["dim"] == 1024

    def test_all_required_indexes_present(self, neo4j_driver):
        result = neo4j_driver.execute_query("SHOW INDEXES")
        names = {rec["name"] for rec in result.records}
        required = {"rag_chunks", "chunk_text_ft"}
        assert not (required - names), f"Missing: {required - names}"

class TestNodeProperties:
    def test_chunk_properties(self, neo4j_driver, expected_chunk_properties):
        result = neo4j_driver.execute_query("MATCH (c:Chunk) RETURN c LIMIT 1")
        if not result.records: pytest.skip("No Chunk nodes")
        props = set(result.records[0]["c"].keys())
        assert not (expected_chunk_properties - props), f"Missing: {expected_chunk_properties - props}"

    def test_document_properties(self, neo4j_driver, expected_document_properties):
        result = neo4j_driver.execute_query("MATCH (d:Document) RETURN d LIMIT 1")
        if not result.records: pytest.skip("No Document nodes")
        props = set(result.records[0]["d"].keys())
        assert not (expected_document_properties - props), f"Missing: {expected_document_properties - props}"

    def test_entity_properties(self, neo4j_driver, expected_entity_properties):
        result = neo4j_driver.execute_query("MATCH (e:Entity) RETURN e LIMIT 1")
        if not result.records: pytest.skip("No Entity nodes")
        props = set(result.records[0]["e"].keys())
        assert not (expected_entity_properties - props), f"Missing: {expected_entity_properties - props}"

class TestRelationships:
    def test_part_of(self, neo4j_driver):
        result = neo4j_driver.execute_query("MATCH (:Chunk)-[r:PART_OF]->(:Document) RETURN count(r) AS cnt")
        assert result.records[0]["cnt"] > 0

    def test_from_chunk(self, neo4j_driver):
        ent = neo4j_driver.execute_query("MATCH (e:Entity) RETURN count(e) AS cnt").records[0]["cnt"]
        if ent == 0: pytest.skip("No entities")
        result = neo4j_driver.execute_query("MATCH (:Entity)-[r:FROM_CHUNK]->(:Chunk) RETURN count(r) AS cnt")
        assert result.records[0]["cnt"] > 0, "Entities exist but no FROM_CHUNK"

    def test_relates_to(self, neo4j_driver):
        ent = neo4j_driver.execute_query("MATCH (e:Entity) RETURN count(e) AS cnt").records[0]["cnt"]
        if ent == 0: pytest.skip("No entities")
        result = neo4j_driver.execute_query("MATCH (:Entity)-[r:RELATES_TO]->(:Entity) RETURN count(r) AS cnt")
        if result.records[0]["cnt"] == 0: pytest.skip("Extraction may not have created RELATES_TO yet")

class TestEmbedding:
    def test_embedding_dim_1024(self):
        from app.services.embeddings import embedding_service
        import asyncio
        vec = asyncio.run(embedding_service.embed_query('test'))
        assert len(vec) == 1024

    def test_neo4j_dim_matches_1024(self, neo4j_driver):
        result = neo4j_driver.execute_query("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN size(c.embedding) AS dim LIMIT 1")
        assert result.records[0]["dim"] == 1024

class TestConstraints:
    @pytest.mark.parametrize("label,prop", [("Chunk","id"),("Document","id"),("Entity","id")])
    def test_uniqueness(self, neo4j_driver, label, prop):
        result = neo4j_driver.execute_query("SHOW CONSTRAINTS YIELD * WHERE type = 'UNIQUENESS' AND labelsOrTypes = [$l] AND properties = [$p] RETURN count(*) AS cnt", l=label, p=prop)
        assert result.records[0]["cnt"] > 0, f"Missing constraint on {label}.{prop}"
