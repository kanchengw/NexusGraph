# NexusGraph — Agent 开发指南

## 项目概述

生产级 GraphRAG 演示系统。知识库：RAGBench TechQA（1,192 文档，~64K chunks）。
LLM：DashScope Qwen3.6-flash。图数据库：Neo4j 5。可观测性：Langfuse + Prometheus + Grafana。

## 三层架构

### 1. Online Agent（在线服务）
- **FastAPI** + **LangGraph** agent
- **3路检索**：Vector（1024d cosine）+ BM25（fulltext）+ Graph（实体扩展 1-2 hops）
- **LLM Reranker**（始终开启，对 chunk 打分 0-10）
- **双层记忆**：短期（session checkpoints）+ 长期（mem0 + pgvector）
- 每次查询全链路 Trace 写入 Langfuse
- 检索指标写入 PostgreSQL（RetrievalMetric 表）

### 2. Backend Agents（离线分析）
| Agent | 目录 | 职责 |
|-------|------|------|
| **Judge** | app/core/judge_agent/ | RAGBench 评测，由对话轮数触发（默认 200 轮） |
| **Optimizer** | app/core/optimizer_agent/ | 指标分析 + LLM 优化建议 + HIL 审批 |
| **Index** | app/core/index_agent/ | 知识库构建（chunk embed 实体抽取） |

数据飞轮：Judge -> Optimizer (HIL) -> Index -> 自动循环

### 3. 常驻基础设施
| 服务 | 用途 |
|------|------|
| PostgreSQL | 检索指标、反馈、评测结果、记忆向量 |
| Neo4j | 知识图谱（Document -> Chunk -> Entity） |
| Prometheus | 时序指标（QPS、延迟） |
| Grafana | 可视化面板（System Overview + Retrieval Insights） |
| Langfuse | 全链路 LLM Trace |
| cAdvisor | 容器监控指标 |

## 代码结构

`
NexusGraph/
  app/
    api/v1/graphrag/           # REST endpoints
    core/
      graphrag/                # 核心：Indexer, Retriever, Neo4j models
      langgraph/               # LangGraph agent + tools
      index_agent/             # Index Agent（graph.py + indexer.py）
      judge_agent/             # Judge Agent
      optimizer_agent/         # Optimizer Agent
      prompts/                 # LLM prompt 模板
    models/                    # SQLModel：RetrievalMetric, EvalResult, Feedback
    services/                  # LLM registry, embeddings, memory
  offline_agent/               # CLI：eval -> analyze -> optimize -> index
  scripts/                     # 工具脚本
  tests/                       # 测试
  docker-compose.yml           # profiles: online / offline / base
`

## 关键技术细节

### 3路检索（app/core/graphrag/retriever.py）
| 路径 | 方法 | 查询方式 |
|------|------|----------|
| Vector | Neo4j vector index (1024d, cosine) | .queryNodes('rag_chunks') |
| BM25 | Neo4j fulltext index | .queryNodes('chunk_text_ft') |
| Graph | 实体扩展 | Cypher 1-2 跳遍历 |

合并后经 LLM Reranker 重新打分排序。

### 实体抽取（indexer.py / extract_entities.py）
- 工具：LLMEntityRelationExtractor（neo4j-graphrag）
- 模型：qwen3.6-flash via DashScope
- 批次：10 chunks/次，并发 3
- Schema：Entity节点 + RELATES_TO + FROM_CHUNK 关系

### Neo4j 图谱 Schema
`
(:Document {id, title, split})
    <-[:PART_OF]-(:Chunk {id, text, chunk_index, embedding})
        <-[:FROM_CHUNK]-(:Entity {id, name, type, description})
            -[:RELATES_TO {relation}]->(:Entity)
`

### 知识库索引（scripts/run_index_agent.py）
`ash
# 索引所有文档（chunk + embed + 实体抽取）
python scripts/run_index_agent.py --split train

# 限制文档数量
python scripts/run_index_agent.py --split train --max-docs 50

# 跳过实体抽取（仅 chunk + embed）
python scripts/run_index_agent.py --split train --skip-entities

# 重建（清空 Neo4j 后重来）
python scripts/run_index_agent.py --split train --reset
`

### 离线数据飞轮（offline_agent/）
`ash
# 完整一轮
python -m offline_agent.cli flywheel --once

# 分步执行
python -m offline_agent.cli eval --num-samples 50
python -m offline_agent.cli analyze --days 7
python -m offline_agent.cli optimize
python -m offline_agent.cli index --max-docs 200
`

### Docker Profiles
`ash
# 完整在线：PostgreSQL + Neo4j + App + Prometheus + Grafana + cAdvisor
docker compose --profile online --env-file .env.production up -d

# 仅数据层：PostgreSQL + Neo4j
docker compose --profile offline up -d

# 仅监控基础设施
docker compose --profile base up -d
`

## 持久化数据
| 组件 | 持久化目录 | 说明 |
|------|-----------|------|
| PostgreSQL | Docker volume postgres_data | 检索指标、反馈、评测、记忆 |
| Neo4j | Docker volume neo4j_data | 知识图谱 |
| Prometheus | Docker volume prometheus_data | 时序指标 |
| Grafana | Docker volume grafana_data | 面板配置 |

## 编码约定
- 所有 import 在文件顶部，禁止 lazy import
- 日志使用 structlog，事件名 lowercase_underscore，禁止 f-string 作为 event 内容
- 全程 async I/O（neo4j async driver, asyncio）
- 全部模型走 DashScope（无 OPENAI_API_KEY）
- 所有函数签名必须有 type hints
