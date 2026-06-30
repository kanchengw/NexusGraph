# NexusGraph

> Production-grade GraphRAG Demo with 3-path retrieval, LLM reranker, offline evaluation, LLM-as-Judge, dual-layer memory, and automated data flywheel.

Built on RAGBench (TechQA) knowledge base (1,192 docs, 63,890 chunks). Graph DB: Neo4j 5. LLM: DashScope Qwen3.6-flash. Observability: Langfuse + Prometheus + Grafana.

***

## Architecture

### System Architecture

```mermaid
graph TB
    subgraph Client["Client"]
        U[User] -->|HTTP| API["API Gateway (8000)"]
    end
    subgraph Online["Online Serving"]
        OA["Online Agent\n(LangGraph)"] --> Ret["3-Path Retriever"]
        Ret --> VEC[Vector\n1024d cosine]
        Ret --> BM[BM25\nfulltext]
        Ret --> GR[Graph\nentity 1-2 hops]
        VEC & BM & GR --> MG[Merge & Deduplicate]
        MG --> RR["LLM Reranker\n(scores chunks 0-10)"]
        RR --> CTR["Context + Memory"]
        OA --> STM["Short-Term Memory\n(PG checkpoints)"]
        OA --> LTM["Long-Term Memory\n(mem0 + pgvector)"]
    end
    subgraph Backend["Backend Agents"]
        JA["Judge Agent\nevaluation"] --> PGE2[(PostgreSQL)]
        PGE2 --> OptA["Optimizer Agent\nanalysis + HIL"]
        OptA --> IA["Index Agent\nKB rebuild"]
    end
    subgraph Storage["Data Layer"]
        PG[("PostgreSQL\nmetrics + feedback\neval + memory")]
        Neo4j[("Neo4j\nknowledge graph")]
        Prom[("Prometheus\ntimeseries")]
    end
    API --> OA
    OA -->|Answer| U
    CTR --> Neo4j
    Neo4j -->|chunks + entities| Ret
    OA -.->|Metrics| Prom
    OA --> Trace[Langfuse]
    Neo4j --> PG
    Prom --> Dash[Grafana]
    PG --> Dash
    IA --> Neo4j
```

### Online Query Flow

```mermaid
flowchart TB
    C[Client] -->|POST /query| GW[API Gateway]
    subgraph Agent["Online Agent"]
        GW -->|session_id + user_id| AG[LangGraph Agent]
        AG --> MEM[memory_service.search]
        MEM -->|relevant memories| AG
        AG --> RET[graphrag_search tool]
        RET --> V[Vector: 1024d cosine]
        RET --> BM[BM25: fulltext index]
        RET --> GR[Graph: entity expand 1-2 hops]
        V & BM & GR --> FUSION[Merge & Deduplicate]
        FUSION --> RR["LLM Reranker\nscores each chunk 0-10"]
        RR --> TOPK[Top-k selection]
        TOPK --> CTX[Build context + system prompt]
        CTX --> GEN[LLM generate response]
        MEM --> GEN
        GEN -->|async| ADD[memory_service.add]
    end
    GW -->|trace| LF[Langfuse]
    GW -->|log| PG[(PostgreSQL)]
    GEN -->|answer| GW
    GW -->|response| C
```

### Backend Data Flywheel

```mermaid
flowchart TB
    subgraph Judge["Judge Agent"]
        direction TB
        RM[(PostgreSQL\nretrievalmetric)] -->|poll: conv_count >= 200| TRIG[Trigger evaluation]
        TRIG --> EVAL[run_evaluation\nRAGBench techqa]
        EVAL -->|faithfulness, relevance,\ncontext_precision| PGJ[(PostgreSQL\nevalresult table)]
    end
    subgraph Optimizer["Optimizer Agent"]
        direction TB
        PGO[(PostgreSQL\nevalresult)] -->|new row| ANLZ["analyzer.py\ngenerate_all_reports"]
        ANLZ -->|analysis_report| LLMO["LLM-as-Optimizer\n(qwen-plus)"]
        LLMO -->|suggestion.json| HIL{Human-in-Loop\nApproval}
        HIL -->|approve| APPLY["Apply params\nchunk_size, top_k, etc."]
        HIL -->|reject| DONE[Discard]
    end
    subgraph Index["Index Agent"]
        direction TB
        APPLY -->|new params| INDEX["rebuild KB\nchunk + embed\n+ entity extract"]
        INDEX --> N4J[(Neo4j\nupdated graph)]
    end
    N4J -->|re-indexed| ONLINE[Online Service\nserves new queries]
```

## Project Structure

```
NexusGraph/
+-- app/core/
|   +-- langgraph/          # Online Agent (graph + tools)
|   +-- graphrag/           # Retriever, Indexer
|   +-- judge_agent/        # Evaluation pipeline
|   +-- optimizer_agent/    # Analysis + LLM opt + HIL
|   +-- index_agent/        # KB construction
+-- app/api/                # REST endpoints
+-- app/models/             # SQLModel
+-- app/services/           # LLM, embeddings, memory
+-- offline_agent/          # CLI entry point
+-- scripts/                # Utility scripts
+-- grafana/                # Dashboard provisioning
+-- prometheus/             # Config
+-- docker-compose.yml      # Docker profiles
```

## Features

### Base Services (always-on infrastructure)
- **PostgreSQL** (pgvector): RetrievalMetric, Feedback, EvalResult, Memory vectors
- **Prometheus + Grafana**: QPS, latency, container metrics, 2 dashboards
- **cAdvisor**: Container resource usage

### Online Agent
- **FastAPI** server with LangGraph agent
- **Two-layer memory**: Short-term (session checkpoints) + Long-term (mem0 + pgvector cross-session)
- **3-path retrieval + LLM reranker**: Vector (1024d cosine) + BM25 (fulltext) + Graph (entity expand, 1-2 hops) + LLM rerank (always on)
- **Langfuse**: Full trace per query

### Backend Agents
| Agent | Directory | Responsibility |
|-------|-----------|----------------|
| **Judge Agent** | app/core/judge_agent/ | RAGBench evaluation, triggered by conversation count (default 200, configurable) |
| **Optimizer Agent** | app/core/optimizer_agent/ | Metrics analysis + LLM optimization + HIL approval |
| **Index Agent** | app/core/index_agent/ | KB construction (chunk, embed, entity extract) |

Data flywheel: Judge (triggered by conversation count) -> Optimizer (HIL) -> Index, automated via workflow.

### Local Mode (Index Agent Only)

ENABLE_LOCAL=true replaces cloud API calls **only for the Index Agent** (knowledge base construction). Online Agent always uses cloud DashScope.

| Component | Cloud (default) | Local (ENABLE_LOCAL=true) |
|-----------|----------------|--------------------------|
| Index Agent Embedding | DashScope text-embedding-v3 | mxbai-embed-large-v1 (sentence-transformers) |
| Index Agent Entity Extraction | DashScope qwen3.6-flash | deepseek-r1:8b (Ollama) |

```bash
# .env.development
ENABLE_LOCAL=true
LOCAL_OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
LOCAL_LLM_MODEL=deepseek-r1:8b
```

## Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- LLM API key (DashScope)

### Setup

```bash
git clone https://github.com/kanchengw/NexusGraph.git
cd NexusGraph
cp .env.example .env.development
# Configure your LLM API key and backend
```

### Start Online System

```bash
docker compose --profile online --env-file .env.production up -d
python run_server.py
```

### Index Knowledge Base

Choose one of two approaches:

**Option A — Download pre-built release (recommended)**
Download from GitHub Releases — contains a fully built Neo4j graph database with TechQA demo data.

```bash
docker compose --profile offline up -d neo4j
docker cp backups/neo4j.dump graphrag-neo4j:/data/
docker exec graphrag-neo4j neo4j-admin database load --from-path=/data/neo4j.dump --overwrite-destination=true
docker restart graphrag-neo4j
```

| Artifact | Count | Description |
|----------|-------|-------------|
| Documents | 1,192 | TechQA train split |
| Chunks | 63,890 | 512-char chunks, 64-char overlap |
| Entity Nodes | ~8K+ | Extracted from ~50 docs (deepseek-r1:8b local / qwen3.6-flash cloud) |
| Relationships | ~15K+ | Entity relations from ~50 docs |

**Option B — Build from scratch**

```bash
python scripts/run_index_agent.py --split train --max-docs 1192
```

> **Note**: With ENABLE_LOCAL=true, entity extraction uses local deepseek-r1:8b (no API key needed). Building full index takes ~30 min (cloud) or ~2-3 hr (local).

### Run Offline Pipeline

```bash
python -m offline_agent.cli --help
# Subcommands: eval, analyze, optimize, index, flywheel
```

## Production Deployment

### Docker Compose Profiles

| Profile     | Services                                                             | Use Case                             |
| ----------- | -------------------------------------------------------------------- | ------------------------------------ |
| **online**  | PostgreSQL + Neo4j + Prometheus + Grafana + cAdvisor + App (FastAPI) | Full stack online serving            |
| **offline** | Neo4j + PostgreSQL                                                   | Data layer only for offline analysis |

### Port Map

| Port | Service     | Public | Purpose           |
| ---- | ----------- | ------ | ----------------- |
| 8000 | FastAPI app | Yes    | API endpoint      |
| 3000 | Grafana     | Yes    | Dashboard UI      |
| 5432 | PostgreSQL  | No     | Internal DB       |
| 7687 | Neo4j Bolt  | No     | Graph DB          |
| 9090 | Prometheus  | No     | Metrics           |
| 8080 | cAdvisor    | No     | Container metrics |

### Data Persistence

| Volume           | Service    | Purpose                         |
| ---------------- | ---------- | ------------------------------- |
| neo4j\_data      | Neo4j      | Graph database                  |
| postgres\_data   | PostgreSQL | Metrics, feedback, eval, memory |
| prometheus\_data | Prometheus | Time-series metrics             |
| grafana\_data    | Grafana    | Dashboard settings              |

### Backup & Restore

```bash
bash scripts/backup-data.sh
bash scripts/restore-data.sh <backup.tar.gz>
```

## Configuration

### Required (.env.development)

| Variable            | Description      |
| ------------------- | ---------------- |
| LLM\_BASE\_URL      | LLM API endpoint |
| LLM\_API\_KEY       | LLM API key      |
| DEFAULT\_LLM\_MODEL | Online LLM model |
| EMBEDDING\_MODEL    | Embedding model  |

### Optional

| Variable                   | Default   | Description         |
| -------------------------- | --------- | ------------------- |
| GRAPHRAG\_CHUNK\_SIZE      | 512       | Chunk size (chars)  |
| GRAPHRAG\_CHUNK\_OVERLAP   | 64        | Chunk overlap       |
| GRAPHRAG\_TOP\_K           | 5         | Top chunks per path |
| GRAPHRAG\_ENABLE\_RERANKER | true      | LLM reranker        |
| EVALUATION\_LLM            | qwen-plus | Judge model         |

## API Endpoints

| Method | Path                            | Description     |
| ------ | ------------------------------- | --------------- |
| POST   | /api/v1/graphrag/query          | Retrieval query |
| GET    | /api/v1/graphrag/health         | Service health  |
| POST   | /api/v1/graphrag/feedback       | Submit feedback |
| GET    | /api/v1/graphrag/feedback/stats | Feedback stats  |

## License

Apache 2.0 - Copyright 2026 kanchengw
