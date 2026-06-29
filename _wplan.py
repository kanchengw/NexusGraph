import pathlib

plan = """# NexusGraph - Architecture & Implementation Plan

Knowledge Base: RAGBench/techqa | LLM: qwen3.6-flash (DashScope) | Judge: qwen-plus
Graph DB: Neo4j 5 | Relational DB: PostgreSQL 16 | Retrieval: 3-path (Vector + BM25 + Graph)

---

## 1. System Architecture

### Online Agent (FastAPI)
- API Gateway port 8000
- 3-Way Retriever: Vector + BM25 + Graph expand
- LangGraph Agent with tool calling
- Langfuse tracing per request
- Monitoring: Prometheus + Grafana

### Offline Agent (CLI)
- RAGBench evaluation pipeline
- LLM-as-Judge scoring (qwen-plus)
- Retrieval analysis & parameter optimization
- Human-in-the-loop approval

---

## 2. Retrieval Pipeline

### 3-Path Retrieval
| Path | Method | Backend |
|------|--------|---------|
| Vector | Dense semantic search (1024d cosine) | Neo4j vector index |
| BM25 | Keyword / sparse search | Neo4j fulltext index |
| Graph | Entity-relation expansion (1-2 hops) | FROM_CHUNK + RELATES_TO |

### Metrics tracked per query (PostgreSQL)
- vector_count, bm25_count, graph_count (per-path results)
- unique_chunks, overlap counts
- response_time_ms

---

## 3. Data Flywheel

1. User query -> retrieve -> answer -> log (Langfuse + PostgreSQL)
2. Feedback collection (POST /graphrag/feedback)
3. Offline eval (RAGBench) -> analyze -> optimize suggestion
4. Human approval -> parameter update -> re-index

---

## 4. Evaluation System

### Current Metrics (LLM-as-Judge, qwen-plus)
| Metric | Range | Description |
|--------|-------|-------------|
| faithfulness | 0-1 | Answer claims supported by context |
| relevance | 0-1 | Answer directly addresses question |
| context_precision | 0-1 | Retrieved chunks are useful |

### Planned Evaluation Improvements
| Priority | Metric | Approach | Status |
|----------|--------|----------|--------|
| P0 | answer_correctness | Compare against RAGBench ground truth | Not implemented |
| P0 | context_recall | Correct chunks ratio in retrieval results | Not implemented |
| P0 | hit_rate / MRR | Rank-aware retrieval quality | Not implemented |
| P0 | response_time | Include latency in eval report | PostgreSQL has data, not in eval |
| P1 | token_consumption | Pull from Langfuse API | Not implemented |
| P1 | user_feedback_score | Aggregate feedback ratings | Endpoint exists, not connected to eval |

### Optimization Strategy
- Semi-automated: LLM reads analysis report -> suggests params -> human approves
- Parameters: top_k (3/5/7), chunk_size (256/512/1024), overlap (32/64/128)
- Track eval score trend to validate improvements

---

## 5. Implementation Status

- [x] Infrastructure: Neo4j + PostgreSQL + Docker Compose
- [x] Index pipeline: chunk + embed + entity extraction
- [x] 3-path retrieval: Vector + BM25 + Graph
- [x] REST API endpoints (query, health, clear, feedback)
- [x] LangGraph agent integration
- [x] Langfuse tracing
- [x] Prometheus + Grafana monitoring
- [x] RAGBench offline evaluation
- [x] Retrieval metrics stored in PostgreSQL
- [x] Offline Agent CLI (eval/analyze/optimize)
- [x] Data flywheel: feedback -> eval -> optimize
- [x] Online/Offline Agent isolation
- [ ] answer_correctness metric
- [ ] context_recall metric
- [ ] hit_rate / MRR metric
- [ ] response_time in eval report
- [ ] token_consumption tracking
- [ ] Reranker (LLM as reranker)
- [ ] A/B experiment framework
- [ ] Docker production deployment optimized

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
| GRAPHRAG_VECTOR_INDEX | rag_chunks |
| FULLTEXT_INDEX | chunk_text_ft |

---

## 7. Future: Fine-tuning Pipeline

- Local training: LLaMA Factory + LoRA (8GB laptop)
- Lightweight deployment: vLLM on Render/Fly.io (1.8B quantized model)
- Decoupled training/inference architecture
- Online demo link for resume
- Full training metrics & screenshots in GitHub README
"""

pathlib.Path("plan.md").write_text(plan, encoding="utf-8")
print(f"Plan written: {len(plan)} chars, {len(plan.split(chr(10)))} lines")