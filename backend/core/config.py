import os

# --- Constants & Environment Variables ---
# Local LLM Config (vLLM)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")
LOCAL_LLM_KEY = os.getenv("LOCAL_LLM_KEY", "local-token") 
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# V2: Caching & Queue Config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")

# V2: DeepSeek OCR Config
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# V2: Security (JWT) Config
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Qdrant Vector DB Config
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vnu_knowledge_v2")

# Database Config
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin@localhost:5432/rag_db")

# V3 MinIO Config
MINIO_URL = os.getenv("MINIO_URL", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "raw-documents")

# Model Specifics
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
