
"""Neo4j GraphRAG data models and schema initialization."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase, GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

from app.core.config import settings
from app.core.logging import logger


CREATE_CONSTRAINTS_QUERIES = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
]

CREATE_INDEX_QUERIES = [
    "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)",
    "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
    "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relation)",
    "CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]",
]

# Entity vector index name
ENTITY_VECTOR_INDEX_NAME = "rag_entities"


async def get_neo4j_driver():
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    return driver


async def init_neo4j_schema():
    driver = await get_neo4j_driver()
    try:
        async with driver.session(database="neo4j") as session:
            for query in CREATE_CONSTRAINTS_QUERIES:
                await session.run(query)
                logger.info("neo4j_constraint_created", query=query[:60])

            for query in CREATE_INDEX_QUERIES:
                await session.run(query)
                logger.info("neo4j_index_created", query=query[:60])

        sync_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        try:
            # Chunk vector index
            create_vector_index(
                sync_driver,
                name=settings.GRAPHRAG_VECTOR_INDEX_NAME,
                label="Chunk",
                embedding_property="embedding",
                dimensions=1024,
                similarity_fn="cosine",
            )
            logger.info("neo4j_chunk_vector_index_created", name=settings.GRAPHRAG_VECTOR_INDEX_NAME)
        except Exception as e:
            logger.warning("chunk_vector_index_may_exist", error=str(e))

        try:
            # Entity vector index (same dimensions, same embedding model)
            create_vector_index(
                sync_driver,
                name=ENTITY_VECTOR_INDEX_NAME,
                label="Entity",
                embedding_property="embedding",
                dimensions=1024,
                similarity_fn="cosine",
            )
            logger.info("neo4j_entity_vector_index_created", name=ENTITY_VECTOR_INDEX_NAME)
        except Exception as e:
            logger.warning("entity_vector_index_may_exist", error=str(e))

        finally:
            sync_driver.close()
    finally:
        await driver.close()


async def clear_database():
    driver = await get_neo4j_driver()
    try:
        async with driver.session(database="neo4j") as session:
            await session.run("MATCH (n) DETACH DELETE n")
            logger.info("neo4j_database_cleared")
    finally:
        await driver.close()
