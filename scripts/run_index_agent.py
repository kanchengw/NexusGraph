#!/usr/bin/env python
"""CLI entry point for the Index Agent - KB construction."""
from __future__ import annotations
import argparse, asyncio, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.index_agent.indexer import run_indexing
from app.core.logging import logger


async def main():
    parser = argparse.ArgumentParser(description="Index Agent - build/rebuild knowledge base")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-docs", type=int, default=0, help="Limit number of documents (0 = all)")
    parser.add_argument("--skip-entities", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    
    logger.info("index_agent_started",
                split=args.split, max_docs=args.max_docs,
                skip_entities=args.skip_entities, reset=args.reset)
    
    await run_indexing(
        split=args.split,
        reset=args.reset,
        skip_entities=args.skip_entities,
        max_docs=args.max_docs,
    )
    
    logger.info("index_agent_complete")


if __name__ == "__main__":
    asyncio.run(main())
