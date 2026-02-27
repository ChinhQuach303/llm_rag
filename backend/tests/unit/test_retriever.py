import pytest
from unittest.mock import MagicMock, patch
from services.retriever import RetrieverService

@patch('services.retriever.CrossEncoder')
def test_search_and_rerank(mock_cross_encoder):
    mock_qdrant = MagicMock()
    mock_embedder = MagicMock()
    
    # Mock embedder returning some vectors
    mock_vecs = {
        "content": [0.1, 0.2],
        "contextual": [0.2, 0.1],
        "sparse": MagicMock(indices=[1], values=[0.5])
    }
    mock_embedder.generate_multi_vectors.return_value = [mock_vecs]
    
    # Mock Reranker Model
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.9] # high score
    mock_cross_encoder.return_value = mock_reranker
    
    class MockHit:
        def __init__(self, _id, score, payload):
            self.id = _id
            self.score = score
            self.payload = payload
            
    mock_hit = MockHit("1", 0.5, {
        "doc_id": "doc1",
        "chunk_index": 5,
        "text": "Base chunk",
        "section_path": "Root"
    })
    mock_qdrant.client.search.return_value = [mock_hit]
    
    # Mock Adjacent scroll hits for Sliding Window (chunk 4 and 6)
    mock_adj_4 = MockHit("4", 0.5, {"chunk_index": 4, "text": "Prev", "doc_id": "doc1"})
    mock_adj_6 = MockHit("6", 0.5, {"chunk_index": 6, "text": "Next", "doc_id": "doc1"})
    
    # Scroll returns tuples: (records, next_page_offset)
    mock_qdrant.client.scroll.return_value = ([mock_adj_4, mock_adj_6], None)
    
    retriever = RetrieverService(mock_qdrant, mock_embedder)
    
    enriched = retriever.search_and_rerank("Test Query", "tenant_xyz", fetch_k=1, rerank_n=1)
    
    assert len(enriched) == 1
    assert "Prev\n\nBase chunk\n\nNext" in enriched[0]["text"]
    assert enriched[0]["rerank_score"] == 0.9
