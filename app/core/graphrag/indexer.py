"""GraphRAG indexing pipeline.

Loads RAGBench/techqa, chunks documents, generates embeddings,
extracts entities/relations via neo4j-graphrag LLMEntityRelationExtractor,
and writes to Neo4j.
"""

from __future__ import annotations

import hashlib
from typing import Any

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)

from app.core.config import settings
from app.core.graphrag.models import get_neo4j_driver, init_neo4j_schema
from app.core.logging import logger
from app.services.embeddings import embedding_service


# Define the schema for entity extraction
ENTITY_SCHEMA = GraphSchema(
    node_types=(
        NodeType(
            label="Entity",
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="type", type="STRING"),
                PropertyType(name="description", type="STRING"),
            ],
            additional_properties=True,
        ),
    ),
    relationship_types=(
        RelationshipType(
            label="RELATES_TO",
            properties=[
                PropertyType(name="relation", type="STRING"),
            ],
            additional_properties=True,
        ),
    ),
)


class KnowledgeBaseIndexer:
    """Indexes a knowledge base into Neo4j GraphRAG."""

    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.GRAPHRAG_CHUNK_SIZE,
            chunk_overlap=settings.GRAPHRAG_CHUNK_OVERLAP,
        )
        self.driver = None
        # Entity extraction LLM: local mode uses deepseek via Ollama, cloud uses DashScope
        if settings.ENABLE_LOCAL:
            self._entity_llm = OpenAILLM(
                model_name=settings.LOCAL_LLM_MODEL,
                base_url=settings.LOCAL_OLLAMA_BASE_URL,
                api_key="ollama",
                model_params={
                    "temperature": 0.1,
                    "max_tokens": 2048,
                },
            )
        else:
            self._entity_llm = OpenAILLM(
                model_name="qwen3.6-flash",
                base_url=settings.LLM_BASE_URL,
                api_key=settings.LLM_API_KEY,
                model_params={
                    "temperature": 0.1,
                    "max_tokens": 2048,
                },
            )
        self._entity_extractor = LLMEntityRelationExtractor(
            llm=self._entity_llm,
            use_structured_output=True,
            create_lexical_graph=True,  # We handle Document->Chunk separately
            max_concurrency=3,
        )

    async def _ensure_driver(self):
        if self.driver is None:
            self.driver = await get_neo4j_driver()
        return self.driver

    async def load_ragbench_techqa(self, split: str = "train") -> list[dict[str, Any]]:
        """Load RAGBench techqa dataset from Hugging Face."""
        logger.info("loading_ragbench_techqa", split=split)
        dataset = load_dataset("galileo-ai/ragbench", "techqa", split=split)
        records = []
        for i, row in enumerate(dataset):
            doc_text = "\n\n".join(row.get("documents", []))
            if doc_text.strip():
                records.append({
                    "id": f"techqa_{split}_{i}",
                    "title": f"TechQA Document {i}",
                    "text": doc_text,
                    "split": split,
                    "metadata": {
                        "question": row.get("question", ""),
                        "answer": row.get("response", ""),
                    },
                })
        logger.info("ragbench_docs_loaded", split=split, count=len(records))
        return records

    async def _extract_entities_from_chunks(
        self, doc_id: str, chunks: list[tuple[str, str]]
    ) -> Neo4jGraph | None:
        """Extract entities from a list of (chunk_id, chunk_text) pairs."""
        text_chunks = TextChunks(
            chunks=[
                TextChunk(text=text, index=idx, metadata={"chunk_id": cid})
                for idx, (cid, text) in enumerate(chunks)
            ]
        )
        try:
            result = await self._entity_extractor.run(
                chunks=text_chunks,
                schema=ENTITY_SCHEMA,
            )
            return result
        except Exception as e:
            logger.warning(
                "entity_extraction_failed_for_doc",
                doc_id=doc_id,
                error=str(e),
            )
            return None

    async def _write_entity_graph_to_neo4j(
        self, graph: Neo4jGraph, doc_id: str, chunk_map: dict[str, str]
    ) -> None:
        """Write extracted entity graph to Neo4j."""
        driver = await self._ensure_driver()
        async with driver.session(database="neo4j") as session:
            # Write Entity nodes
            for node in graph.nodes:
                await session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.label = $label,
                        e.description = $description
                    """,
                    id=node.id,
                    name=node.properties.get("name", node.id),
                    label=node.label,
                    description=str(node.properties.get("description", ""))[:500],
                )

            # Write FROM_CHUNK relationships (Entity->Chunk) from lexical graph
            # (extractor may create lexical graph chunks too)
            for rel in graph.relationships:
                if rel.type == "FROM_CHUNK":
                    await session.run(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:Entity {id: $entity_id})
                        MERGE (e)-[:FROM_CHUNK]->(c)
                        """,
                        chunk_id=rel.start_node_id,
                        entity_id=rel.end_node_id,
                    )
                elif rel.type == "RELATES_TO":
                    await session.run(
                        """
                        MATCH (s:Entity {id: $source_id})
                        MATCH (t:Entity {id: $target_id})
                        MERGE (s)-[:RELATES_TO {relation: $relation}]->(t)
                        """,
                        source_id=rel.start_node_id,
                        target_id=rel.end_node_id,
                        relation=rel.properties.get("relation", ""),
                    )

    async def _write_chunk_and_embedding(
        self, session, chunk_id: str, chunk_text: str,
        chunk_idx: int, doc_id: str, embedding_vec: list[float]
    ) -> None:
        await session.run(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text = $text,
                c.chunk_index = $chunk_index,
                c.document_id = $document_id,
                c.embedding = $embedding
            WITH c
            MATCH (d:Document {id: $document_id})
            MERGE (c)-[:PART_OF]->(d)
            """,
            id=chunk_id,
            text=chunk_text,
            chunk_index=chunk_idx,
            document_id=doc_id,
            embedding=embedding_vec,
        )

    async def index_document(
        self, doc: dict[str, Any], extract_entities: bool = True
    ) -> None:
        """Index a single document into Neo4j."""
        driver = await self._ensure_driver()
        doc_id = doc["id"]
        content_hash = hashlib.sha256(doc["text"].encode()).hexdigest()

        async with driver.session(database="neo4j") as session:
            await session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.title = $title,
                    d.content_hash = $content_hash,
                    d.split = $split,
                    d.metadata = $metadata
                """,
                id=doc_id,
                title=doc["title"],
                content_hash=content_hash,
                split=doc["split"],
                metadata=str(doc.get("metadata", {})),
            )

            raw_chunks = self.text_splitter.split_text(doc["text"])
            if not raw_chunks:
                return

            # Batch embedding
            if settings.ENABLE_LOCAL:
                from app.services.embeddings import LocalSentenceEmbedding
                _local_embedder = LocalSentenceEmbedding()
                _embed = _local_embedder.embed_documents
            else:
                _embed = embedding_service.embed_documents

            all_embeddings = []
            for batch_start in range(0, len(raw_chunks), 10):
                batch = raw_chunks[batch_start:batch_start + 10]
                batch_embeddings = await _embed(batch)
                all_embeddings.extend(batch_embeddings)
            embeddings = all_embeddings

            chunk_ids = []
            chunks_with_text = []
            for chunk_idx, chunk_text in enumerate(raw_chunks):
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                chunk_ids.append(chunk_id)
                chunks_with_text.append((chunk_id, chunk_text))
                embedding_vec = embeddings[chunk_idx] if embeddings else []
                await self._write_chunk_and_embedding(
                    session, chunk_id, chunk_text, chunk_idx, doc_id, embedding_vec
                )

        # Entity extraction (separate session, batched)
        if extract_entities and chunks_with_text:
            graph = await self._extract_entities_from_chunks(doc_id, chunks_with_text)
            if graph and (graph.nodes or graph.relationships):
                await self._write_entity_graph_to_neo4j(graph, doc_id, {})
                logger.debug(
                    "entities_extracted",
                    doc_id=doc_id,
                    nodes=len(graph.nodes),
                    relationships=len(graph.relationships),
                )

        logger.debug("document_indexed", doc_id=doc_id, chunks=len(raw_chunks))

    async def index_all(
        self, split: str = "train", extract_entities: bool = True,
        max_entity_docs: int = 100
    ) -> None:
        """Index all documents from dataset."""
        records = await self.load_ragbench_techqa(split)
        for i, doc in enumerate(records):
            # Only extract entities for the first N documents (performance)
            do_entities = extract_entities and i < max_entity_docs
            await self.index_document(doc, extract_entities=do_entities)
            if (i + 1) % 100 == 0:
                logger.info(
                    "indexing_progress",
                    split=split, completed=i + 1, total=len(records)
                )
        if self.driver:
            await self.driver.close()
        logger.info("indexing_complete", split=split, total=len(records))

    async def clear_all(self) -> None:
        from app.core.graphrag.models import clear_database
        await clear_database()


async def run_indexing(
    split: str = "train", reset: bool = False, skip_entities: bool = False
) -> None:
    """Run the full indexing pipeline."""
    if reset:
        logger.info("resetting_neo4j_schema")
        await init_neo4j_schema()
    indexer = KnowledgeBaseIndexer()
    await indexer.index_all(split, extract_entities=not skip_entities)
