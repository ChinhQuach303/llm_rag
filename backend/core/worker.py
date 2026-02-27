"""
core/worker.py
--------------
Defines the Celery distributed task logic for background document processing.
Offloads heavy computation (OCR rendering, chunking, and HNSW embeddings) away 
from the synchronous FastAPI event loop.
"""

import os
import logging
import uuid
from celery import Celery

from core.config import CELERY_BROKER_URL, REDIS_URL
from db.models import DocumentRecord
from core.database import SessionLocal
from services.pdf_parser import DocumentParserService
from services.embedding import EmbeddingService
from services.vector_store import VectorStoreService
from core.storage import MinioStorageService

logger = logging.getLogger(__name__)

celery_app = Celery(
    "rag_tasks",
    broker=CELERY_BROKER_URL,
    backend=REDIS_URL
)

# Enforce secure JSON serialization for Celery messages
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"]
)

@celery_app.task(bind=True, name="process_document_task")
def process_document_task(self, object_name: str, original_filename: str, tenant_id: str):
    """
    V3 Celery worker execution path.
    
    Why this decoupled design?
    1. Downloads the raw file safely from MinIO (S3) to local worker scratch space.
    2. Runs intensive Vision OCR and BGE-M3 Embeddings on identical scalable child processes.
    3. Cleans up the scratch file post-execution to prevent volume bloat.
    """
    logger.info(f"Worker started for MinIO object: {object_name}, tenant: {tenant_id}")
    
    # Initialize Heavy Services inside the worker process
    db = SessionLocal()
    parser = DocumentParserService(db)
    embedder = EmbeddingService()
    qdrant = VectorStoreService()
    storage = MinioStorageService()
    
    os.makedirs("data/worker_tmp", exist_ok=True)
    local_path = f"data/worker_tmp/{uuid.uuid4()}_{original_filename}"

    try:
        # Download from MinIO
        storage.download_file(object_name, local_path)
        
        doc_id = parser.process_and_save(local_path, original_filename, tenant_id)
        if doc_id:
            chunks = parser.chunk_document(doc_id)
            if chunks:
                multi_vectors = embedder.generate_multi_vectors(chunks)
                qdrant.upsert_multi_vector(chunks, multi_vectors, tenant_id)
                logger.info(f"Successfully processed {len(chunks)} chunks for {object_name}")
                return {"status": "success", "doc_id": str(doc_id), "chunks_processed": len(chunks)}
            else:
                return {"status": "failed", "error": "No chunks generated"}
        return {"status": "failed", "error": "Parser failed to save document"}
        
    except Exception as e:
        logger.error(f"Worker Error for {object_name}: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
        # Clean up local temp file downloaded by worker
        if os.path.exists(local_path):
             try:
                 os.remove(local_path)
             except Exception:
                 pass
