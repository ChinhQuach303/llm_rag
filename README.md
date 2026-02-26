# Modular Enterprise RAG System (V3 MVP)

A highly robust, production-grade Retrieval-Augmented Generation (RAG) backend utilizing FastAPI, Celery, PostgreSQL, MinIO, and Qdrant.

## Architecture & Capabilities

This V3 "Real MVP" codebase has been entirely refactored to prioritize resilience, explicit architecture bounds, and zero-downtime execution:

- **Infra Isolation**: Uses PostgreSQL to eliminate SQLite concurrency locks and S3 MinIO to prevent dropped files on container restarts.
- **Fail-Safe Ingestion**: Asynchronous Celery workers utilize DeepSeek Vision OCR to extract complex PDFs. Network drops are mitigated by `tenacity` exponential backoffs, and chunk sizes are strictly bounded below 512 tokens using `tiktoken`.
- **Hybrid Search & Sliding Windows**: Uses Qdrant's sparse vector support (via fastembed BM25) strictly alongside standard dense cross-encoder reranking. We utilize Contextual Sliding Windows to pull `chunk_index ± 1` to enrich the context space continuously.
- **Industrial Semantic Caching**: Naive Redis lists are completely replaced with an ultra-fast Qdrant Native Semantic Cache (`semantic_cache_v3`).
- **Security**: Rate limits are enforced on all routes via `slowapi` backed by Redis.

## Getting Started

### 1. Requirements
- Docker & Docker Compose
- DeepSeek API Key (For OCR Ingestion)
- Relevant vLLM/Local Inference setup (For standard chat generation)

### 2. Configure Environment
Provide the required keys in `.env` based on `.env.example`. Check `core/config.py` for variables mapping.

### 3. Build & Run
The entire supporting cluster is managed via Docker:
```bash
docker-compose up --build -d
```

Ensure MinIO, Redis, Qdrant, Postgres, Celery, and the FastAPI application successfully boot. 

* The core backend serves locally at `http://localhost:8000`.

## API Documentation
Interactive Swagger docs run natively built-in via FastAPI: 
* Navigate to `http://localhost:8000/docs` while the server runs.

### Author
Refactored by [ChinhQuach303] within the V3 System Enhancement cycle.