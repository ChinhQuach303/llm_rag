"""
main.py
-------
Main entry point for the FastAPI Modular RAG System.
Configures application lifespan events (database initialization, global model caching),
boots up middleware (CORS, Rate Limiting), and registers REST API execution routes.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from core.database import init_db
from api.routes import ingest, chat
from api.dependencies import get_embedder, get_vector_store
from core.rate_limiter import limiter

# Configure standard logging format to standard output.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown lifecycle of the FastAPI application.
    
    1. Initializes Relational Database models.
    2. Performs 'Cold Start' model loading for heavy dependencies (e.g., SentenceTransformers).
       This guarantees immediate fast response times on the very first API hit instead
       of penalizing the first user with load-times.
    """
    logger.info("🚀 2026 Production API Base Starting...")
    
    # Establish base SQL tables
    init_db()
    
    # Pre-cache singletons into memory
    get_embedder()
    get_vector_store()
    
    yield
    
    logger.info("Backend Shutting Down cleanly...")

# Initialize Application Instance
app = FastAPI(title="2026 Modular RAG System", lifespan=lifespan)

# Attach Redis-backed Global Rate Limiter to prevent API abuse
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configured open CORS policy for external UI interconnects
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bind discrete domain routes to the root API
app.include_router(ingest.router, prefix="/v1", tags=["Ingestion"])
app.include_router(chat.router, prefix="/v1", tags=["Query"])

@app.get("/")
@limiter.limit("5/minute")
async def root(request: Request):
    """
    Lightweight health check endpoint.
    Verifies that the API server is reachable and active.
    """
    return {"status": "active", "architecture": "2026 Modular FastAPI Clean Architecture V3"}