# NexusGraph ！ Development Instructions

Refer to AGENTS.md for full project context.

## Identity
NexusGraph is a production-grade GraphRAG demo built on RAGBench (TechQA). This is a personal portfolio / resume project.

## Current State
- 1,192 documents indexed ★ 63,890 chunks (1024d embedding)
- 34 documents with entity extraction ★ ~11,700 entities
- 3-path retrieval: vector + BM25 + graph expand
- Server runs on Windows via run_server.py (SelectorEventLoop workaround)
- Docker Compose with profile isolation (online / offline)
- ~40 tests passing

## API Endpoints
- POST /api/v1/graphrag/query ！ 3-path retrieval
- GET /api/v1/graphrag/health ！ health check
- POST /api/v1/graphrag/clear ！ clear graph
- POST /api/v1/graphrag/feedback ！ user feedback

## Credentials (dev, DO NOT COMMIT)
All in .env.development (gitignored). Uses DashScope (NOT OpenAI).
- LLM: qwen3.6-flash (online), qwen-plus (judge)
- Embedding: text-embedding-v3
- Observability: Langfuse (jp.cloud.langfuse.com)
