import json
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from glob import glob
import logging


class VectorStoreLocal:
    def __init__(self, model_name='intfloat/multilingual-e5-large', data_dir='data', output_dir='output', device=None):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.chunks = []
        self.embeddings = np.zeros((0, 1024), dtype=np.float32)  # Initialize with correct shape
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def check_environment(self):
        self.logger.info(f"Using device: {self.device}")
        if self.device == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")

    def load_model(self, offline_path=None):
        path = offline_path if offline_path and os.path.exists(offline_path) else self.model_name
        self.model = SentenceTransformer(path, device=self.device)
        self.logger.info("Model loaded successfully!")

    def load_embeddings(self, json_path=None, npy_path=None):
        json_path = json_path or os.path.join(self.output_dir, 'embeddings.json')
        try:
            if os.path.exists(json_path):
                self.logger.info(f"Loading embeddings from {json_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.chunks = [{k: v for k, v in item.items() if k != 'embedding'} for item in data]
                self.embeddings = np.array([item['embedding'] for item in data], dtype=np.float32)
            elif npy_path and os.path.exists(npy_path):
                self.logger.info(f"Loading embeddings from {npy_path}")
                self.embeddings = np.load(npy_path).astype(np.float32)
                json_files = glob(os.path.join(self.data_dir, '*.json'))
                if json_files:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        self.chunks = json.load(f)
                else:
                    self.chunks = []
            else:
                self.logger.warning(f"No embedding file found at {json_path}. Initializing empty store.")
                self.embeddings = np.zeros((0, 1024), dtype=np.float32)
                self.chunks = []

            self.embeddings = self._normalize_embeddings(self.embeddings)
            self.logger.info(f"Loaded {len(self.embeddings)} embeddings of dimension {self.embeddings.shape[1]}")
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            self.embeddings = np.zeros((0, 1024), dtype=np.float32)
            self.chunks = []

    def _normalize_embeddings(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)

    def save_embeddings(self, index_name='embeddings'):
        output_json = os.path.join(self.output_dir, f'{index_name}.json')
        output_npy = os.path.join(self.output_dir, f'{index_name}.npy')

        try:
            data = [
                {
                    **chunk,
                    'embedding': self.embeddings[i].tolist()
                } for i, chunk in enumerate(self.chunks)
            ]
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            np.save(output_npy, self.embeddings)
            self.logger.info(f"Saved embeddings to {output_json} and {output_npy}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")

    def add_vectors(self, new_chunks, batch_size=32):
        if self.model is None:
            self.load_model()

        texts = ["passage: " + chunk['content'] for chunk in new_chunks]
        new_embeddings = []

        self.logger.debug(f"Encoding {len(texts)} new chunks in batches of {batch_size}")
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding new embeddings"):
            batch = texts[i:i + batch_size]
            encoded = self.model.encode(batch, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
            new_embeddings.append(encoded)

        new_embeddings = np.vstack(new_embeddings).astype(np.float32)
        self.logger.debug(f"New embeddings shape: {new_embeddings.shape}")

        # Handle empty or uninitialized embeddings
        if self.embeddings.size == 0 or self.embeddings.shape[1] != new_embeddings.shape[1]:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.concatenate([self.embeddings, new_embeddings], axis=0)

        self.chunks.extend(new_chunks)
        self.logger.info(f"Added {len(new_chunks)} new vectors. Total embeddings: {len(self.embeddings)}")

    def search(self, query, k=5):
        if self.model is None:
            self.load_model()
        if self.embeddings is None or self.embeddings.size == 0:
            self.logger.warning("Embeddings not loaded or empty.")
            return []

        query_embedding = self.model.encode(["passage: " + query], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_k_idx = np.argsort(similarities)[-k:][::-1]

        return [
            {
                'content': self.chunks[i]['content'],
                'metadata': self.chunks[i].get('metadata', {}),
                'similarity': float(similarities[i])
            } for i in top_k_idx
        ]

    def test_search(self, queries, k=5):
        for query in queries:
            self.logger.info(f"\nQuery: {query}")
            results = self.search(query, k)
            for r in results:
                self.logger.info(f"- {r['content'][:100]}... (similarity: {r['similarity']:.4f})")
                self.logger.info(f"  Metadata: {r['metadata']}")

    def run(self, embedding_json=None, embedding_npy=None, queries=None, offline_model_path=None):
        self.check_environment()
        self.load_model(offline_model_path)
        self.load_embeddings(json_path=embedding_json, npy_path=embedding_npy)
        self.save_embeddings()
        if queries:
            self.test_search(queries)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    vector_store = VectorStoreLocal(
        model_name='intfloat/multilingual-e5-large',
        data_dir='data/json',
        output_dir='output'
    )

    test_queries = [
        "Mã trường của Trường Đại học Giáo dục là gì?",
        "Triết lý giáo dục của ĐHQGHN là gì?",
        "Chỉ tiêu tuyển sinh năm 2024 của ngành Sư phạm Toán là bao nhiêu?"
    ]

    vector_store.run(
        embedding_json='output/embeddings.json',
        queries=test_queries,
        offline_model_path=None
    )

vector_store.save_embeddings(index_name='my_index')