"""
services/retriever.py
---------------------
Coordinates the Hybrid Search (BM25 + Dense) mechanism against Qdrant.
Applies Reciprocal Rank Fusion (RRF) to merge lexical and semantic matches,
then executes an intensive Cross-Encoder Reranking pass for maximum relevance precision.
"""

import logging
from typing import List, Dict, Any
from qdrant_client import models
from .vector_store import VectorStoreService
from .embedding import EmbeddingService
from sentence_transformers import CrossEncoder
from core.config import RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)

class RetrieverService:
    """
    Advanced Multi-Vector Retriever.
    
    Why Dual-Pass Retrieval?
    1. First Pass (Qdrant): Fast approximate nearest neighbors (HNSW + BM25) scaling to millions of chunks.
    2. Second Pass (Cross-Encoder): Computationally heavy pairwise scoring for the top N candidates
       to definitively sort out nuance before passing context to the LLM.
    """
    def __init__(self, qdrant_manager: VectorStoreService, embedder: EmbeddingService):
        self.qdrant = qdrant_manager
        self.embedder = embedder

        logger.info(f"Loading Reranker model: {RERANKER_MODEL_NAME}...")
        try:
            self.reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device='cpu')
            logger.info("Reranker model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Reranker: {e}")
            self.reranker_model = None

    def search_and_rerank(self, query: str, tenant_id: str, fetch_k: int = 20, rerank_n: int = 5) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Multi-Vector search (Dense + Sparse BM25), Cross-encoder reranking,
        and Sliding Window context augmentation.
        """
        logger.info(f"Executing Hybrid Multi-Vector Search for tenant: {tenant_id}")
        
        # 1. Retrieve
        class MockChunk:
            def __init__(self, t):
                self.text = t
                self.section_path = ""
        
        multi_vecs = self.embedder.generate_multi_vectors([MockChunk(query)])
        if not multi_vecs:
            return []
            
        q_vectors = multi_vecs[0]

        tenant_filter = models.Filter(
            must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
        )

        try:
            prefetch_list = [
                models.Prefetch(query=q_vectors["content"], using="content", limit=fetch_k),
                models.Prefetch(query=q_vectors["contextual"], using="contextual", limit=fetch_k)
            ]
            
            sp_vec = q_vectors.get("sparse")
            if sp_vec and len(list(sp_vec.indices)) > 0:
                q_sparse = models.SparseVector(
                    indices=sp_vec.indices.tolist() if hasattr(sp_vec.indices, 'tolist') else list(sp_vec.indices),
                    values=sp_vec.values.tolist() if hasattr(sp_vec.values, 'tolist') else list(sp_vec.values)
                )
                prefetch_list.append(models.Prefetch(query=q_sparse, using="text-sparse", limit=fetch_k))

            search_result = self.qdrant.client.search(
                collection_name=self.qdrant.collection_name,
                query_filter=tenant_filter,
                search_params=models.SearchParams(exact=False, hnsw_ef=64),
                prefetch=prefetch_list,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=fetch_k,
                with_payload=True
            )

            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "doc_id": hit.payload.get("doc_id", ""),
                    "chunk_index": hit.payload.get("chunk_index"),
                    "text": hit.payload.get("text", ""),
                    "section_path": hit.payload.get("section_path", ""),
                })

            logger.info(f"Retrieved {len(results)} chunks.")
            
            # 2. Rerank
            if self.reranker_model and results:
                pairs = [[query, doc["text"]] for doc in results]
                scores = self.reranker_model.predict(pairs)
                
                for i, doc in enumerate(results):
                    doc["rerank_score"] = float(scores[i])
                    
                results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:rerank_n]
            else:
                results = results[:rerank_n]

            # 3. Contextual Sliding Window
            enriched_results = []
            for doc in results:
                base_text = doc["text"]
                d_id = doc.get("doc_id")
                c_idx = doc.get("chunk_index")
                
                if d_id and c_idx is not None:
                    # Fetch adjacent chunks (i-1, i+1)
                    adj_filter = models.Filter(
                        must=[
                            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                            models.FieldCondition(key="doc_id", match=models.MatchValue(value=d_id)),
                        ],
                        should=[
                            models.FieldCondition(key="chunk_index", match=models.MatchValue(value=c_idx - 1)),
                            models.FieldCondition(key="chunk_index", match=models.MatchValue(value=c_idx + 1))
                        ]
                    )
                    adj_records, _ = self.qdrant.client.scroll(
                        collection_name=self.qdrant.collection_name,
                        scroll_filter=adj_filter,
                        limit=2,
                        with_payload=True
                    )
                    
                    # Prepend/Append based on index order
                    adj_records = sorted(adj_records, key=lambda x: x.payload.get("chunk_index", 0))
                    for rec in adj_records:
                        i = rec.payload.get("chunk_index")
                        t = rec.payload.get("text", "")
                        if i < c_idx:
                            base_text = f"{t}\n\n{base_text}"
                        elif i > c_idx:
                            base_text = f"{base_text}\n\n{t}"
                            
                doc["text"] = base_text
                enriched_results.append(doc)
            
            return enriched_results

        except Exception as e:
            logger.error(f"Error during Search/Rerank: {e}")
            return []
