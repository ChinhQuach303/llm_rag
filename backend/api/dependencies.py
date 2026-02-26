from fastapi import Depends
from typing import Generator
from sqlalchemy.orm import Session
from core.database import get_db

from services.vector_store import VectorStoreService
from services.embedding import EmbeddingService
from services.pdf_parser import DocumentParserService
from services.retriever import RetrieverService
from services.generator import GeneratorService
from services.pipeline import RAGPipelineService

# Global Singletons for Heavy ML Models 
# Avoids reloading large models per request
_embedder_instance = None
_vector_store_instance = None
_generator_instance = None
_retriever_instance = None
_pipeline_instance = None

def get_embedder() -> EmbeddingService:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = EmbeddingService()
    return _embedder_instance

def get_vector_store() -> VectorStoreService:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
    return _vector_store_instance

def get_document_parser(db: Session = Depends(get_db)) -> DocumentParserService:
    """Instantiated per request since it uses the DB session."""
    return DocumentParserService(db)

def get_rag_pipeline() -> RAGPipelineService:
    global _retriever_instance, _generator_instance, _pipeline_instance
    
    if _pipeline_instance is None:
        qdrant = get_vector_store()
        embedder = get_embedder()
        
        _retriever_instance = RetrieverService(qdrant, embedder)
        _generator_instance = GeneratorService()
        _pipeline_instance = RAGPipelineService(_retriever_instance, _generator_instance)
        
    return _pipeline_instance
