"""Embedding service wrapping DashScope text-embedding-v3 via direct API call."""

from openai import AsyncOpenAI
from pydantic import SecretStr

from app.core.config import settings


class EmbeddingService:
    """Service for generating embeddings using DashScope text-embedding-v3."""

    def __init__(self):
        """Initialize the embedding service."""
        self._client = AsyncOpenAI(
            api_key=SecretStr(settings.EMBEDDING_API_KEY).get_secret_value(),
            base_url=settings.EMBEDDING_BASE_URL,
        )
        self._model = settings.EMBEDDING_MODEL
        self._dimensions = 1024

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        # Sort by index to preserve order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=[text],
            dimensions=self._dimensions,
        )
        return response.data[0].embedding


embedding_service = EmbeddingService()
