"""
api/routes/chat.py
------------------
Defines the REST endpoints for interacting with the Retrieval-Augmented Generation (RAG) pipeline.
It bridges external HTTP requests to the internal Qdrant/LLM coordination services.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from schemas.request import QueryRequest
from api.dependencies import get_rag_pipeline
from services.pipeline import RAGPipelineService
from core.security import get_current_tenant
from core.rate_limiter import limiter

router = APIRouter()

@router.post("/query")
@limiter.limit("50/minute")
async def query_endpoint(
    request: Request,
    query_body: QueryRequest,
    pipeline: RAGPipelineService = Depends(get_rag_pipeline),
    tenant_id: str = Depends(get_current_tenant)
):
    """
    Stateless Query API.
    
    Why StreamingResponse?
    LLM inferences can take 5-15 seconds for a complete paragraph to generate.
    Using Server-Sent Events (SSE) via StreamingResponse allows the client UI to
    render tokens in real-time instantly, vastly improving perceived performance/UX.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="System initializing...")

    history = [{"role": msg.role, "content": msg.content} for msg in query_body.history]
    
    # Trigger the RAG pipeline generator
    token_generator = pipeline.chat(
         query=query_body.query, 
         tenant_id=tenant_id,
         mode="qa",
         history=history
    )

    return StreamingResponse(token_generator, media_type="text/event-stream")
