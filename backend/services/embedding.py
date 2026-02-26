"""
services/embedding.py
---------------------
Service boundary for generating numerical vectors from text.
It loads both heavy Dense Transformer models into memory and fast Sparse BM25 generators
to create multi-dimensional embeddings for advanced search.
"""

import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
try:
    from fastembed import SparseTextEmbedding
except ImportError:
    SparseTextEmbedding = None

from core.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Encoder service for unstructured text.
    
    Why CPU inference for embeddings?
    While GPUs are faster, embedding models like BGE-M3 run adequately fast on CPU via ONNX optimizations.
    This reserves VRAM strictly for the expensive LLM Generation phase.
    """
    def __init__(self):
        logger.info(f"Loading Dense Embedding Model: {EMBEDDING_MODEL_NAME}...")
        try:
            self.client = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self.dimension = EMBEDDING_DIMENSION
            logger.info("Dense Embedding model loaded.")
        except Exception as e:
            logger.error(f"Failed to load dense embedding model: {e}")
            self.client = None
            self.dimension = EMBEDDING_DIMENSION
            
        logger.info("Loading Sparse BM25 Model via FastEmbed...")
        try:
            if SparseTextEmbedding:
                self.sparse_client = SparseTextEmbedding(model_name="Qdrant/bm25")
                logger.info("Sparse model loaded.")
            else:
                self.sparse_client = None
                logger.warning("fastembed not installed. Sparse vectors will be omitted.")
        except Exception as e:
            logger.error(f"Failed to load sparse embedding model: {e}")
            self.sparse_client = None

    def generate_multi_vectors(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        if not chunks or not self.client:
            return []

        content_texts = []
        contextual_texts = []

        for chunk in chunks:
            text = chunk.text.replace("\n", " ").strip()
            content_texts.append(text)
            
            path = chunk.section_path if hasattr(chunk, 'section_path') and chunk.section_path else "Root"
            contextual_texts.append(f"{path}: {text}")

        logger.info(f"Generating Multi-Vectors (Dense + Sparse) for {len(chunks)} chunks...")
        
        try:
            content_embeddings = self.client.encode(content_texts, normalize_embeddings=True, show_progress_bar=False).tolist()
            contextual_embeddings = self.client.encode(contextual_texts, normalize_embeddings=True, show_progress_bar=False).tolist()
            
            if self.sparse_client:
                # Returns an iterator of SparseEmbedding objects
                sparse_embeddings = list(self.sparse_client.embed(content_texts))
            else:
                sparse_embeddings = [None] * len(chunks)

            multi_vectors = []
            for cont_emb, ctx_emb, sp_emb in zip(content_embeddings, contextual_embeddings, sparse_embeddings):
                multi_vectors.append({
                    "content": cont_emb,
                    "contextual": ctx_emb,
                    "sparse": sp_emb
                })
            return multi_vectors

        except Exception as e:
            logger.error(f"Multi-Vector error: {e}")
            return []
