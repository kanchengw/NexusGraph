# NexusGraph — Agent Development Guide

## Project Overview
Production-grade GraphRAG demo with 3-path retrieval (vector + BM25 + graph expand), RAGBench offline evaluation, and data flywheel. Knowledge base: TechQA (1,192 docs → 63,890 chunks). Graph DB: Neo4j. LLM: DashScope Qwen.

## Quick Commands
`ash
# Server (Windows)
conda activate newML && python _run_server.py

# Index knowledge base
conda run -n newML python scripts/ingest_knowledge_base.py

# Entity extraction
conda run -n newML python scripts/extract_entities.py --max-docs 50

# Tests
conda run -n newML python -m pytest tests/ -v

# Offline eval
make offline-pipeline

# Docker
make stack-online      # full stack: app + Neo4j + PG + Prometheus + Grafana
make stack-offline     # data layer only
`

## Project Structure
`
NexusGraph/
├── app/api/v1/graphrag/     # REST endpoints (query, health, clear, feedback)
├── app/core/graphrag/       # Indexer, Retriever (3-path), Neo4j models
├── app/core/langgraph/      # Agent graph + tools (graphrag_search, web_search)
├── app/models/              # SQLModel: RetrievalMetric, EvalResult, Feedback
├── app/services/            # LLM registry, embeddings, memory
├── evals/                   # RAGBench evaluation with LLM-as-Judge
├── offline_agent/           # CLI: eval → analyze → optimize (human-in-loop)
├── scripts/                 # ingest, extract_entities, optimize_rag, analyze
├── tests/                   # 40+ tests: data structures, Neo4j, retriever
└── docker-compose.yml       # profiles: online / offline isolation
`

## Key Technical Details

### 3-Path Retrieval (retriever.py)
| Path | Method | Query |
|------|--------|-------|
| Vector | Neo4j vector index (1024d, cosine) | db.index.vector.queryNodes('rag_chunks') |
| BM25 | Neo4j fulltext index (stop words filtered) | db.index.fulltext.queryNodes('chunk_text_ft') |
| Graph | Entity expansion (FROM_CHUNK → RELATES_TO) | Cypher traversal, 1-2 hops |

### Entity Extraction (indexer.py / extract_entities.py)
- Tool: LLMEntityRelationExtractor from neo4j-graphrag
- Backend: qwen3.6-flash via DashScope (OpenAI-compatible API)
- Schema: Entity nodes + RELATES_TO + FROM_CHUNK relationships
- Batch: 10 chunks per LLM call, concurrency=3

### Knowledge Graph Schema (models.py)
- (:Document {id, title, split})
- (:Chunk {id, text, chunk_index, embedding})
- (:Entity {id, name, type, description})
- (:Document)<-[:PART_OF]-(:Chunk)
- (:Entity)-[:FROM_CHUNK]->(:Chunk)
- (:Entity)-[:RELATES_TO {relation}]->(:Entity)

### Evaluation (evaluate_graphrag.py)
- Framework: RAGBench (TechQA test split)
- Judge: qwen-plus (cross-model, avoids self-evaluation bias)
- Metrics: faithfulness, relevance, context_precision
- Results stored in PostgreSQL + Langfuse

### Data Flywheel (offline_agent/)
1. User query → feedback → eval → analyze → LLM optimization suggestions
2. Human approves parameter changes (top_k, chunk_size)
3. Re-index with optimized params → re-evaluate → monitor trend

### Observability
- Langfuse: full trace per query (vector/Bm25/graph spans + metadata)
- PostgreSQL: RetrievalMetric per query, EvalResult per run, Feedback
- Prometheus + Grafana: system metrics (QPS, latency, memory)

### Configuration (.env.development)
`env
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DEFAULT_LLM_MODEL=qwen3.6-flash
EMBEDDING_MODEL=text-embedding-v3
GRAPHRAG_CHUNK_SIZE=512
GRAPHRAG_CHUNK_OVERLAP=64
LANGFUSE_HOST=https://jp.cloud.langfuse.com
`

## Coding Rules
- All imports at top of file. No lazy imports.
- Log with structlog: lowercase_underscore event names, no f-strings in events
- Async I/O throughout (neo4j async driver, asyncio)
- No OPENAI_API_KEY references — all models use DashScope
- Type hints on all function signatures

## Key Dependencies
- fastapi, uvicorn — web server
- neo4j, neo4j-graphrag — graph DB + entity extraction
- langchain-text-splitters — document chunking
- sqlmodel, psycopg — PostgreSQL ORM
- langfuse — LLM observability
- datasets (HuggingFace) — RAGBench loading
- prometheus-client, grafana — metrics/monitoring
- pydantic — config validation
