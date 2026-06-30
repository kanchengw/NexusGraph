"""Embedding service supporting DashScope (default) and local sentence-transformers (local mode)."""

from __future__ import annotations
import os

import asyncio
from functools import lru_cache

from openai import AsyncOpenAI
from pydantic import SecretStr

from app.core.config import settings


class DashScopeEmbedding:
    """Embedding via DashScope text-embedding-v3 (default)."""

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=SecretStr(settings.EMBEDDING_API_KEY).get_secret_value(),
            base_url=settings.EMBEDDING_BASE_URL,
        )
        self._model = settings.EMBEDDING_MODEL
        self._dimensions = 1024

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model, input=texts, dimensions=self._dimensions,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def embed_query(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model, input=[text], dimensions=self._dimensions,
        )
        return response.data[0].embedding


class LocalSentenceEmbedding:
    """Embedding via local sentence-transformers (mxbai-embed-large-v1, 1024d).

    Uses mixedbread-ai/mxbai-embed-large-v1 loaded via sentence-transformers.
    Model is cached in ~/.cache/huggingface/hub/ after first download.
    """

    def __init__(self):
        self._model_name = settings.LOCAL_EMBEDDING_MODEL
        self._dimensions = settings.LOCAL_EMBEDDING_DIM
        self._model = None  # Lazy-loaded on first call
        # Ensure HF mirror for China mainland
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        if "TRANSFORMERS_OFFLINE" not in os.environ:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def _load_model(self):
        """Load the sentence-transformers model (synchronous, cached)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = await asyncio.to_thread(self._load_model)
        embeddings = await asyncio.to_thread(model.encode, texts, show_progress_bar=False)
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        model = await asyncio.to_thread(self._load_model)
        embedding = await asyncio.to_thread(model.encode, [text], show_progress_bar=False)
        return embedding[0].tolist()


if settings.ENABLE_LOCAL:
    embedding_service: DashScopeEmbedding | LocalSentenceEmbedding = LocalSentenceEmbedding()
else:
    embedding_service = DashScopeEmbedding()