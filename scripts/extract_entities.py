# -*- coding: utf-8 -*-
"""Extract entities from existing Neo4j chunks - batches of 10 chunks per LLM call."""
from __future__ import annotations
import asyncio, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks, DocumentInfo
from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType, PropertyType, RelationshipType
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from app.core.config import settings
from app.core.graphrag.models import get_neo4j_driver
import logging
for _lg in ["httpx","httpcore","neo4j"]: logging.getLogger(_lg).setLevel(logging.WARNING)
from app.core.logging import logger

SCHEMA = GraphSchema(
    node_types=(NodeType(label="Entity", properties=[PropertyType(name="name", type="STRING"), PropertyType(name="type", type="STRING"), PropertyType(name="description", type="STRING")], additional_properties=True),),
    relationship_types=(RelationshipType(label="RELATES_TO", properties=[PropertyType(name="relation", type="STRING")], additional_properties=True),),
)

async def get_remaining_docs(driver, max_docs=200, split="train"):
    """Get docs that don't have entities yet, up to max_docs total with entities."""
    async with driver.session(database="neo4j") as session:
        # Already done
        done = await session.run("""
            MATCH (e:Entity)-[:FROM_CHUNK]->(:Chunk)-[:PART_OF]->(d:Document {split: $sp})
            RETURN d.id AS doc_id ORDER BY d.id
        """, sp=split)
        done_ids = set()
        async for rec in done:
            done_ids.add(rec["doc_id"])
        
        # All docs up to target
        all_docs = await session.run("""
            MATCH (d:Document {split: $sp})
            RETURN d.id AS doc_id ORDER BY d.id LIMIT $md
        """, sp=split, md=max_docs)
        all_ids = []
        async for rec in all_docs:
            all_ids.append(rec["doc_id"])
        
        already_count = sum(1 for did in all_ids if did in done_ids)
        need_count = min(max_docs - already_count, max_docs - len(done_ids))
        
        remaining = [did for did in all_ids if did not in done_ids][:need_count]
        
        # Get chunks for remaining docs
        docs_data = []
        for doc_id in remaining[:need_count]:
            r = await session.run(
                "MATCH (d:Document {id: $did})<-[:PART_OF]-(c:Chunk) RETURN c {.id, .text, .chunk_index} ORDER BY c.chunk_index",
                did=doc_id
            )
            chunks = []
            async for rec in r:
                chunks.append(dict(rec["c"]))
            if chunks:
                docs_data.append({"doc_id": doc_id, "chunks": chunks})
        
        return docs_data, already_count

async def write_extracted_graph(driver, graph):
    async with driver.session(database="neo4j") as session:
        for node in graph.nodes:
            if node.label == "Entity":
                nm = node.properties.get("name", node.id)
                tp = node.properties.get("type", "")
                desc = str(node.properties.get("description", ""))[:500]
                await session.run("MERGE (e:Entity {id: $id}) SET e.name = $nm, e.type = $tp, e.description = $desc", id=node.id, nm=nm, tp=tp, desc=desc)
        for rel in graph.relationships:
            if rel.type == "FROM_CHUNK":
                await session.run("MATCH (e:Entity {id: $eid}) MATCH (c:Chunk {id: $cid}) MERGE (e)-[:FROM_CHUNK]->(c)", eid=rel.start_node_id, cid=rel.end_node_id)
            elif rel.type == "RELATES_TO":
                rv = rel.properties.get("relation", "")
                await session.run("MATCH (s:Entity {id: $sid}) MATCH (t:Entity {id: $tid}) MERGE (s)-[:RELATES_TO {relation: $r}]->(t)", sid=rel.start_node_id, tid=rel.end_node_id, r=rv)

async def run_entity_extraction(max_docs=200, split="train", max_concurrency=3, batch_chunks=10):
    llm = OpenAILLM(model_name="qwen3.6-flash", base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY,
                    model_params={"temperature": 0.1, "max_tokens": 2048, "timeout": 60})
    extractor = LLMEntityRelationExtractor(llm=llm, use_structured_output=True, create_lexical_graph=True, max_concurrency=max_concurrency)
    driver = await get_neo4j_driver()
    sem = asyncio.Semaphore(max_concurrency)
    
    try:
        docs_data, already_done = await get_remaining_docs(driver, max_docs=max_docs, split=split)
        logger.info("entity_extraction_started", remaining=len(docs_data), already_done=already_done, target=max_docs)
        
        async def process_doc(doc):
            doc_id = doc["doc_id"]
            chunks = doc["chunks"]
            async with sem:
                total_entities = 0
                # Process chunks in batches
                for batch_start in range(0, len(chunks), batch_chunks):
                    batch = chunks[batch_start:batch_start + batch_chunks]
                    text_chunks = TextChunks(chunks=[
                        TextChunk(text=c["text"], index=c["chunk_index"], uid=str(c["id"])) for c in batch
                    ])
                    doc_info = DocumentInfo(path=f"{doc_id}/batch_{batch_start}", document_type="inline_text")
                    try:
                        graph = await asyncio.wait_for(
                            extractor.run(chunks=text_chunks, document_info=doc_info, schema=SCHEMA),
                            timeout=120  # 2 min per batch
                        )
                        if graph and (graph.nodes or graph.relationships):
                            await write_extracted_graph(driver, graph)
                            total_entities += len(graph.nodes)
                    except asyncio.TimeoutError:
                        logger.warning("batch_timeout", doc=doc_id, batch=batch_start)
                    except Exception as e:
                        logger.warning("batch_failed", doc=doc_id, batch=batch_start, error=str(e)[:100])
                return doc_id, total_entities, len(chunks)
        
        completed = 0
        tasks = [process_doc(doc) for doc in docs_data]
        for coro in asyncio.as_completed(tasks):
            try:
                doc_id, ents, chk_count = await coro
                completed += 1
                logger.info("doc_done", doc=doc_id, entities=ents, chunks=chk_count, progress=f"{completed}/{len(docs_data)}")
            except Exception as e:
                logger.error("doc_fatal", error=str(e)[:100])
    finally:
        await driver.close()
    logger.info("entity_extraction_complete", processed=len(docs_data))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--batch-chunks", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run_entity_extraction(max_docs=args.max_docs, split=args.split, max_concurrency=args.concurrency, batch_chunks=args.batch_chunks))
