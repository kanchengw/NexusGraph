#!/usr/bin/env python
"""CLI entry point for ingesting knowledge base into Neo4j GraphRAG."""

from __future__ import annotations

import argparse
import asyncio

from app.core.graphrag.indexer import KnowledgeBaseIndexer, run_indexing
from app.core.graphrag.models import init_neo4j_schema, clear_database
from app.core.logging import logger


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base into Neo4j GraphRAG")
    parser.add_argument("--split", default="train", help="Dataset split: train, validation, test")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate Neo4j schema")
    args = parser.parse_args()

    logger.info("ingest_started", split=args.split, reset=args.reset)

    if args.reset:
        await clear_database()
        await init_neo4j_schema()

    indexer = KnowledgeBaseIndexer()
    await indexer.index_all(split=args.split)

    logger.info("ingest_complete", split=args.split)


if __name__ == "__main__":
    asyncio.run(main())
