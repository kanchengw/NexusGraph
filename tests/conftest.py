"""Conftest: shared fixtures for all tests."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_env_path = os.path.join(os.path.dirname(__file__), "..", ".env.development")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8", errors="ignore") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import pytest
from neo4j import GraphDatabase

@pytest.fixture(scope="session")
def neo4j_config():
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "neo4jpassword"),
    }

@pytest.fixture(scope="session")
def neo4j_driver(neo4j_config):
    driver = GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"]))
    driver.verify_connectivity()
    yield driver
    driver.close()

@pytest.fixture
def expected_chunk_properties():
    return {"id", "text", "chunk_index", "document_id", "embedding"}

@pytest.fixture
def expected_document_properties():
    return {"id", "title", "content_hash", "split", "metadata"}

@pytest.fixture
def expected_entity_properties():
    return {"id", "name", "type", "description"}

@pytest.fixture
def expected_vector_index():
    return {"name": "rag_chunks", "label": "Chunk", "property": "embedding", "dimensions": 1024}

@pytest.fixture
def expected_fulltext_index():
    return {"name": "chunk_text_ft", "label": "Chunk", "properties": ["text"]}
