"""
services/vector_store.py
------------------------
Manages the connection and lifecycle of Qdrant Collections.
Responsible for orchestrating the schema for both Dense and Sparse representations,
and configuring high-performance indexing via HNSW.
"""

import logging
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from core.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Qdrant Connection Manager and Data Ingestor.
    
    Why Qdrant?
    Chosen for its native support for Hybrid Search (Dense + Sparse/BM25) and
    advanced Prefetching (RRF) features which are critical for precision RAG.
    """
    def __init__(self, vector_size: int = 1024):
        self.collection_name = COLLECTION_NAME
        self.vector_size = vector_size
        
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=15)
            self._ensure_collection()
            logger.info("Connected to Qdrant successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def _ensure_collection(self):
        if not self.client:
            return
            
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        
        if not exists:
            logger.info(f"Creating Multi-Vector & Sparse collection '{self.collection_name}'...")
            vectors_config = {
                "content": models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                "contextual": models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
            }
            sparse_vectors_config = {
                "text-sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
            hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                hnsw_config=hnsw_config
            )
            
            # Index logic for multitenant
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
    def upsert_multi_vector(self, chunks: List[Any], multi_vectors: List[Dict[str, Any]], tenant_id: str):
        if not self.client or not chunks or not multi_vectors:
            return
            
        points = []
        for i, (chunk, vectors) in enumerate(zip(chunks, multi_vectors)):
            # Format sparse vector for Qdrant
            sparse_vec = vectors["sparse"]
            qdrant_sparse = models.SparseVector(
                indices=sparse_vec.indices.tolist() if hasattr(sparse_vec.indices, 'tolist') else list(sparse_vec.indices),
                values=sparse_vec.values.tolist() if hasattr(sparse_vec.values, 'tolist') else list(sparse_vec.values)
            )

            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "content": vectors["content"],
                    "contextual": vectors["contextual"],
                    "text-sparse": qdrant_sparse
                },
                payload={
                    "tenant_id": tenant_id,
                    "doc_id": str(chunk.doc_id),
                    "chunk_index": i,
                    "text": chunk.text,
                    "section_path": chunk.section_path,
                    "depth": chunk.depth,
                    "start_page": chunk.start_page
                }
            ))
            
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            logger.info(f"Upserted {len(points)} multi-vectors (Dense+Sparse) for {tenant_id}.")
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
