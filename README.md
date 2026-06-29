# NexusGraph

> Production-grade GraphRAG Demo with 3-path knowledge retrieval, offline evaluation, LLM-as-Judge, and a data flywheel for continuous improvement.

Built on RAGBench (TechQA) knowledge base, with Neo4j as graph storage, DashScope (Qwen) as LLM/embedding backend, and Langfuse for full observability.

## Features

### 3-Path Retrieval
| Path | Method | Backend |
|---|---|---|
| Vector | Dense semantic search | Neo4j vector index (1024d, cosine) |
| BM25 | Keyword / sparse search | Neo4j fulltext index (stop words filtered) |
| Graph | Entity-relation expansion | Knowledge graph FROM_CHUNK + RELATES_TO |

### Knowledge Graph
- Entity extraction via neo4j-graphrag LLMEntityRelationExtractor (qwen3.6-flash)
- ~11,700 entities, ~9,800 typed relations across 34 documents
- Neo4j vector index (1024d) + fulltext index (text)

### Evaluation and Data Flywheel
- RAGBench offline eval: faithfulness / relevance / context_precision
- Cross-model LLM-as-Judge: qwen-plus evaluates qwen3.6-flash outputs
- Semi-automated optimizer with human-in-the-loop approval

### Observability
| Tool | Purpose |
|---|---|
| Langfuse | Full LLM trace per query (3-path spans + metadata) |
| PostgreSQL | RetrievalMetric / EvalResult / Feedback tables |
| Prometheus + Grafana | System QPS / latency / memory |

## Quick Start

### Two Ways to Use

| Mode | Description | Best For |
|---|---|---|
| **Skeleton** (default branch) | Clone and run index + entity extraction yourself | Users who want full build experience |
| **With Data** (Release asset) | Download pre-built Neo4j dump, restore directly | Users who want to see results quickly |

> Note: Even with pre-loaded data, querying still calls DashScope LLM for answer generation. API costs apply for LLM calls.

### Prerequisites

- Python 3.12+ (recommended: conda env)
- Docker and Docker Compose
- DashScope API Key (https://bailian.console.aliyun.com/)

### 1. Setup

```bash
git clone https://github.com/kanchengw/NexusGraph.git
cd NexusGraph
cp .env.example .env.development
# Edit .env.development -- fill in your DashScope API key
```

### 2. Start Infrastructure

```bash
# Data layer only (Neo4j + PostgreSQL)
docker compose --profile offline up -d

# Or full stack (+ app + Prometheus + Grafana)
docker compose --profile online up -d
```

### 3. Index Knowledge Base

```bash
conda activate newML
pip install -e .
python scripts/ingest_knowledge_base.py
python scripts/extract_entities.py --max-docs 50
```

### 4. Query

```bash
# Start server
conda run -n newML python run_server.py

# In another terminal:
curl -X POST http://localhost:8000/api/v1/graphrag/query -H "Content-Type: application/json" -d '{"question":"What is IBM WebSphere?","top_k":5}'
```

### 5. Offline Evaluation

```bash
conda run -n newML python -m offline_agent.cli pipeline
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /api/v1/graphrag/query | 3-path retrieval query |
| GET | /api/v1/graphrag/health | Service health + node count |
| POST | /api/v1/graphrag/clear | Clear knowledge graph |
| POST | /api/v1/graphrag/feedback | Submit feedback for flywheel |

## Project Structure

```
NexusGraph/
|-- app/api/v1/graphrag/     # REST endpoints
|-- app/core/graphrag/       # Indexer, Retriever, Neo4j models
|-- app/core/langgraph/      # Agent graph + tools
|-- app/models/              # SQLModel tables
|-- app/services/            # LLM registry, embeddings
|-- evals/                   # RAGBench evaluation pipeline
|-- offline_agent/           # CLI: eval / analyze / optimize
|-- scripts/                 # ingest, extract_entities, optimize_rag
|-- tests/                   # 40+ tests
|-- docker-compose.yml       # Online / Offline profiles
```

## Tech Stack

| Category | Technology |
|---|---|
| Framework | FastAPI + LangGraph |
| Graph DB | Neo4j 5 Community (APOC) |
| Vector Store | Neo4j vector index (1024d) |
| PostgreSQL | pgvector + SQLModel |
| LLM | DashScope Qwen3.6-flash / Qwen-Plus |
| Embedding | DashScope text-embedding-v3 |
| Observability | Langfuse, Prometheus, Grafana |
| Chunking | RecursiveCharacterTextSplitter (512/64) |

## Test Status

```bash
pytest tests/ -v
# 40 passed, 2 skipped, 0 failed
```

## License

MIT
