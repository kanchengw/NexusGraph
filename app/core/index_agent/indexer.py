"""Index Agent - KB construction, delegating to KnowledgeBaseIndexer."""
from __future__ import annotations
from app.core.graphrag.indexer import KnowledgeBaseIndexer, run_indexing
from app.core.logging import logger

__all__ = ["KnowledgeBaseIndexer", "run_indexing"]
