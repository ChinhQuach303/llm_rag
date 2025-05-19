
import json
import numpy as np
import torch
import os
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from glob import glob

class EmbeddingGeneratorKaggle:
    def __init__(self, model_name='intfloat/multilingual-e5-large', batch_size=32, data_dir='/kaggle/input/uet-rag-data', output_dir='/kaggle/working', device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_environment(self):
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}, Count: {torch.cuda.device_count()}")
        try:
            response = requests.get('https://huggingface.co', timeout=5)
            print(f"Internet connection: {response.status_code}")
            self.internet_available = True
        except:
            print("No internet connection.")
            self.internet_available = False
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
    
    def load_model(self, offline_path=None):
        try:
            if self.internet_available:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            elif offline_path and os.path.exists(offline_path):
                self.model = SentenceTransformer(offline_path, device=self.device)
            else:
                raise FileNotFoundError("No internet and no offline model.")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_chunks(self, pattern="*.json"):
        chunk_files = glob(os.path.join(self.data_dir, pattern))
        if not chunk_files:
            raise FileNotFoundError(f"No JSON chunk files found in {self.data_dir}")
        
        all_chunks = []
        all_contents = []
        for file_path in chunk_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                all_chunks.extend(chunks)
                all_contents.extend([chunk['content'] for chunk in chunks])
                print(f"Loaded {len(chunks)} chunks from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return all_chunks, all_contents
    
    def create_embeddings(self, contents):
        if self.model is None:
            self.load_model()
        passages = ["passage: " + content for content in contents]
        embeddings = []
        try:
            for i in tqdm(range(0, len(passages), self.batch_size), desc="Encoding embeddings"):
                batch = passages[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size
                )
                embeddings.append(batch_embeddings)
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            return np.vstack(embeddings)
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise
    
    def save_embeddings(self, chunks, embeddings, output_name='embeddings'):
        embedding_data = [
            {'id': chunk['id'], 'content': chunk['content'], 'embedding': embedding.tolist(), 'metadata': chunk['metadata']}
            for chunk, embedding in zip(chunks, embeddings)
        ]
        json_path = os.path.join(self.output_dir, f'{output_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, ensure_ascii=False, indent=2)
        print(f"Saved embeddings to {json_path}")
        
        npy_path = os.path.join(self.output_dir, f'{output_name}.npy')
        np.save(npy_path, embeddings)
        print(f"Saved embeddings to {npy_path}")
    
    def check_embeddings(self, embeddings, contents, num_samples=5):
        print(f"Total embeddings: {len(embeddings)}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Norms (should be ~1): min={norms.min()}, max={norms.max()}")
        
        sample_idx = np.random.choice(len(embeddings), num_samples, replace=False)
        for idx in sample_idx:
            similarities = cosine_similarity([embeddings[idx]], embeddings)[0]
            top_k = np.argsort(similarities)[-3:][::-1]
            print(f"\nChunk: {contents[idx][:100]}...")
            print("Top 3 similar chunks:")
            for k in top_k:
                print(f"- {contents[k][:100]}... (similarity: {similarities[k]:.4f})")
    
    def run(self, offline_model_path=None):
        self.check_environment()
        chunks, contents = self.load_chunks()
        embeddings = self.create_embeddings(contents)
        self.save_embeddings(chunks, embeddings)
        self.check_embeddings(embeddings, contents)