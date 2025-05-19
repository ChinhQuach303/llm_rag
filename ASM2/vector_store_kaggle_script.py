
import json
import numpy as np
import torch
import os
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class VectorStoreKaggle:
    def __init__(self, model_name='intfloat/multilingual-e5-large', data_dir='/kaggle/working', output_dir='/kaggle/working', device=None, index_type='hnsw'):
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_type = index_type
        self.model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_environment(self):
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}, Count: {torch.cuda.device_count()}")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
    
    def load_model(self, offline_path=None):
        try:
            if os.path.exists(offline_path or ''):
                self.model = SentenceTransformer(offline_path, device=self.device)
            else:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_embeddings(self, json_path=None):
        if json_path and os.path.exists(json_path):
            print(f"Loading embeddings from {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
            self.chunks = [{'id': item['id'], 'content': item['content'], 'metadata': item['metadata']} for item in embedding_data]
            self.embeddings = np.array([item['embedding'] for item in embedding_data], dtype=np.float32)
        else:
            raise FileNotFoundError("No valid embedding JSON file provided.")
        print(f"Loaded {len(self.embeddings)} embeddings with dimension {self.embeddings.shape[1]}")
    
    def build_index(self):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded.")
        dimension = self.embeddings.shape[1]
        try:
            if self.index_type == 'hnsw':
                self.index = faiss.IndexHNSWFlat(dimension, 32)
                self.index.hnsw.efConstruction = 40
            else:
                self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            print(f"Built FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            raise
    
    def save_index(self, index_name='vector_store'):
        index_path = os.path.join(self.output_dir, f'{index_name}.bin')
        faiss.write_index(self.index, index_path)
        print(f"Saved FAISS index to {index_path}")
        
        chunks_path = os.path.join(self.output_dir, f'{index_name}_chunks.json')
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved chunks to {chunks_path}")
    
    def search(self, query, k=3):
        if self.model is None:
            self.load_model()
        if self.index is None:
            raise ValueError("No index built.")
        query_embedding = self.model.encode([f"passage: {query}"], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, k=k)
        return [
            {'content': self.chunks[idx]['content'], 'metadata': self.chunks[idx]['metadata'], 'distance': distances[0][i]}
            for i, idx in enumerate(indices[0])
        ]
    
    def run(self, embedding_json=None, offline_model_path=None):
        self.check_environment()
        self.load_embeddings(json_path=embedding_json)
        self.build_index()
        self.save_index()