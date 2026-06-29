# NexusGraph - Architecture & Implementation Plan

Knowledge Base: RAGBench/techqa | LLM: qwen3.6-flash (DashScope) | Judge: qwen-plus
Graph DB: Neo4j 5 | Relational DB: PostgreSQL 16 | Retrieval: 3-path (Vector + BM25 + Graph)

---

## 1. System Architecture

### Online Agent (FastAPI)
- API Gateway port 8000
- 3-Way Retriever: Vector + BM25 + Graph expand
- Optional LLM Reranker (GRAPHRAG_ENABLE_RERANKER)
- LangGraph Agent with tool calling
- Langfuse tracing per request
- Prometheus + Grafana monitoring

### Offline Pipeline
- RAGBench evaluation -> LLM-as-Judge (7 metrics)
- Analysis report (4 data sources: PostgreSQL + Prometheus + Langfuse + JSON)
- LLM-as-Optimizer with causal attribution
- Human-in-the-loop approval -> parameter auto-deploy

---

## 2. Retrieval Pipeline

### 3-Path Retrieval
| Path | Method | Backend |
|------|--------|---------|
| Vector | Dense semantic search (1024d cosine) | Neo4j vector index |
| BM25 | Keyword / sparse search | Neo4j fulltext index |
| Graph | Entity-relation expansion (1-2 hops) | FROM_CHUNK + RELATES_TO |

### Optional: LLM Reranker
- Uses qwen3.6-flash to score all unique chunks from 3 paths
- Disabled by default, enable with GRAPHRAG_ENABLE_RERANKER=true

---

## 3. Observability

| Tool | Data | Access Method |
|------|------|---------------|
| Langfuse | LLM trace, latency, token cost | HTTP API (cloud) |
| PostgreSQL | RetrievalMetric, EvalResult, Feedback | SQL |
| Prometheus | QPS, request latency, DB connections | HTTP API (TSDB) |
| Grafana | Visual dashboards (2 provisioned) | Prometheus + PostgreSQL |

---

## 4. Evaluation System

### Metrics (LLM-as-Judge, qwen-plus)
| Metric | Range | Status |
|--------|-------|--------|
| faithfulness | 0-1 | Done |
| relevance | 0-1 | Done |
| context_precision | 0-1 | Done |
| answer_correctness | 0-1 | Done |
| context_recall | 0-1 | Done |
| hit_rate | 0-1 | Done |
| avg_response_time_ms | - | Done |
| total_tokens | - | Done |

### Analysis Pipeline
- analyze_retrieval.py reads 4 sources -> analysis_report.json
- optimize_rag.py sends report to LLM -> optimization_suggestion.json
- Human reviews -> python optimize_rag.py --apply

---

## 5. Implementation Status

- [x] Infrastructure: Neo4j + PostgreSQL + Docker Compose
- [x] Index pipeline: chunk + embed + entity extraction
- [x] 3-path retrieval: Vector + BM25 + Graph
- [x] REST API endpoints (query, health, clear, feedback)
- [x] LLM Reranker (optional, toggleable)
- [x] LangGraph agent integration
- [x] Langfuse tracing
- [x] Prometheus + Grafana monitoring (2 dashboards)
- [x] RAGBench offline evaluation (7 metrics)
- [x] Retrieval metrics stored in PostgreSQL
- [x] Langfuse LLM metrics in analysis report
- [x] Prometheus system metrics in analysis report
- [x] Offline Agent CLI (eval/analyze/optimize)
- [x] Data flywheel: feedback -> analyze -> optimize
- [x] LLM-as-Optimizer with attribution rules
- [x] Online/Offline isolation
- [ ] User feedback score connected to eval pipeline
- [ ] A/B experiment framework
- [ ] Docker production deployment optimized
- [ ] Graph entity extraction: all 1192 docs (currently 50)

---

## 6. Models & API Config

| Parameter | Value |
|-----------|-------|
| LLM_BASE_URL | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| LLM_MODEL | qwen3.6-flash |
| JUDGE_MODEL | qwen-plus |
| EMBEDDING_MODEL | text-embedding-v3 |
| EMBEDDING_DIM | 1024 |
| NEO4J_URI | neo4j://localhost:7687 |
| LANGFUSE_HOST | https://jp.cloud.langfuse.com |
| GRAPHRAG_CHUNK_SIZE | 512 |
| GRAPHRAG_CHUNK_OVERLAP | 64 |
| GRAPHRAG_TOP_K | 5 |
| GRAPHRAG_ENABLE_RERANKER | false |
