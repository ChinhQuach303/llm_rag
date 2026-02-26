"""
api/routes/ingest.py
--------------------
Defines the REST endpoints responsible for securely accepting raw documents,
storing them in persistent fault-tolerant object storage (MinIO), and dispatching
heavy extraction/embedding workloads to a background Celery worker queue.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request
import os
import uuid
import logging

from schemas.response import IngestResponse
from core.security import get_current_tenant
from core.worker import process_document_task
from core.storage import MinioStorageService
from core.rate_limiter import limiter

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ingest", response_model=IngestResponse)
@limiter.limit("20/minute")
async def ingest_endpoint(
    request: Request,
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_current_tenant)
):
    """
    Stateless File Ingestion API.
    
    Why this architecture?
    Instead of processing the PDF synchronously (which blocks the API and causes timeouts),
    we simply act as a fast pass-through router. We dump the raw file into an S3 bucket (MinIO)
    and pass the object reference to a distributed Celery Worker.
    """
    
    # 1. Upload to S3/MinIO to ensure no data loss if the FastAPI pod crashes.
    storage = MinioStorageService()
    object_name = f"{tenant_id}/{uuid.uuid4()}_{file.filename}"
    
    try:
         file_data = await file.read()
         storage.upload_file(object_name, file_data, content_type=file.content_type)
    except Exception as e:
         logger.error(f"MinIO upload failed: {e}")
         raise HTTPException(status_code=500, detail="Storage service unavailable.")
         
    # 2. Queue the heavy processing (OCR, Embedding) to the Celery Worker queue.
    # The API instantly responds to the client with a task_id for future status polling.
    task = process_document_task.delay(object_name, file.filename, tenant_id)
    
    return IngestResponse(
        task_id=task.id,
        status="QUEUED_TO_WORKER",
        message=f"File {file.filename} dispatched securely to S3 processing queue."
    )


