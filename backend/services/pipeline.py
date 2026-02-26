"""
services/pipeline.py
--------------------
Orchestrator for the entire Retrieval-Augmented Generation execution flow.
Fuses caching, retrieval, grounding evaluation, and LLM text generation into
a single, unified streaming pipeline hook for the API.
"""

import logging
import json
import uuid
from typing import Optional
from qdrant_client import models
from .retriever import RetrieverService
from .generator import GeneratorService

logger = logging.getLogger(__name__)

class MockChunk: # Helper for embedding caching strings
    def __init__(self, text):
        self.text = text
        self.section_path = ""

class RAGPipelineService:
    """
    Orchestrates the query pipeline cleanly.
    
    Why Semantic Caching?
    Identical or semantically similar queries (cosine > 0.95) skip the expensive 
    Reraking and LLM generation phases entirely, serving cached answers in <50ms.
    """
    def __init__(self, retriever: RetrieverService, generator: GeneratorService):
        self.retriever = retriever
        self.generator = generator
        self.qdrant_client = self.retriever.qdrant.client
        self.cache_collection = "semantic_cache_v3"
        
        self._ensure_cache_collection()

    def _ensure_cache_collection(self):
        if not self.qdrant_client:
            return
            
        try:
            collections = self.qdrant_client.get_collections()
            if not any(c.name == self.cache_collection for c in collections.collections):
                logger.info(f"Creating Qdrant Semantic Cache Collection '{self.cache_collection}'...")
                self.qdrant_client.create_collection(
                    collection_name=self.cache_collection,
                    vectors_config=models.VectorParams(
                        size=self.retriever.embedder.dimension,
                        distance=models.Distance.COSINE
                    )
                )
                self.qdrant_client.create_payload_index(
                    collection_name=self.cache_collection,
                    field_name="tenant_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant Cache Collection: {e}")

    def _check_semantic_cache(self, query: str, tenant_id: str) -> Optional[str]:
        if not self.qdrant_client:
            return None
            
        try:
            # Embed the new query
            query_vectors = self.retriever.embedder.generate_multi_vectors([MockChunk(query)])
            if not query_vectors:
                return None
                
            q_vec = query_vectors[0]["content"]
            
            tenant_filter = models.Filter(
                must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
            )
            
            # Fast ANN Search on Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.cache_collection,
                query_vector=q_vec,
                query_filter=tenant_filter,
                limit=1,
                score_threshold=0.95, # Strict threshold for caching
                with_payload=True
            )
            
            if search_result:
                best_match = search_result[0]
                logger.info(f"⚡ Qdrant Semantic Cache HIT (Score: {best_match.score:.4f}) for query.")
                return best_match.payload.get("response", "")
                
            logger.info("Semantic Cache MISS.")
            return None
            
        except Exception as e:
            logger.error(f"Semantic Cache Evaluation Error: {e}")
            return None

    def _save_semantic_cache(self, query: str, tenant_id: str, response: str):
         if not self.qdrant_client:
             return
             
         try:
             query_vectors = self.retriever.embedder.generate_multi_vectors([MockChunk(query)])
             if not query_vectors:
                 return
             
             q_vec = query_vectors[0]["content"]
             
             self.qdrant_client.upsert(
                 collection_name=self.cache_collection,
                 points=[
                     models.PointStruct(
                         id=str(uuid.uuid4()),
                         vector=q_vec,
                         payload={
                             "tenant_id": tenant_id,
                             "query": query,
                             "response": response
                         }
                     )
                 ],
                 wait=False
             )
         except Exception as e:
             logger.error(f"Failed to save semantic cache to Qdrant: {e}")

    def chat(self, query: str, tenant_id: str, mode: str = "qa", history: list = []):
        logger.info(f"Pipeline invoked | Tenant: {tenant_id} | Query: {query[:30]}")
        
        # 1. Attempt Semantic Cache first (Avoids DB/Disk/LLM)
        cached_response = self._check_semantic_cache(query, tenant_id)
        if cached_response:
             yield cached_response
             return
        
        # 2. Proceed with RAG Fallback
        reranked_hits = self.retriever.search_and_rerank(query, tenant_id, fetch_k=20, rerank_n=5)
        
        if mode == "debug":
             yield json.dumps({"reranked_top5": reranked_hits})
             return
             
        top_score = reranked_hits[0]["rerank_score"] if reranked_hits else 0.0
        
        if not self.generator.evaluate_groundedness(top_score, threshold=0.1):
             logger.warning(f"Query '{query}' rejected due to grounding failure.")
             yield "NOT FOUND (Insufficient Retrieval Context)" 
             return
             
        # 3. Stream and accumulate the generated response
        full_response = ""
        for token in self.generator.generate_stream(query, reranked_hits, history):
             full_response += token
             yield token
             
        # 4. Save to Semantic Cache asynchronously
        if full_response and "NOT FOUND" not in full_response:
             self._save_semantic_cache(query, tenant_id, full_response)

