import json
import numpy as np
import torch
import os
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from glob import glob

class VectorStoreLocal:
    def __init__(self, model_name='intfloat/multilingual-e5-large', data_dir='data', output_dir='output', device=None, index_type='flat_l2'):
        """
        Khởi tạo vector store với FAISS cho môi trường local.
        Args:
            model_name (str): Tên model hoặc đường dẫn local (mặc định: multilingual-e5-large).
            data_dir (str): Thư mục chứa file embedding/chunk.
            output_dir (str): Thư mục lưu vector store và kết quả.
            device (str): Thiết bị ('cuda', 'cpu', hoặc None để tự động chọn).
            index_type (str): Loại FAISS index ('flat_l2', 'hnsw', 'ivf').
        """
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
        """Kiểm tra môi trường: GPU, thư mục đầu vào."""
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}, Count: {torch.cuda.device_count()}")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
    
    def load_model(self):
        """Tải model để encode query (nếu cần)."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Ensure model is downloaded or specify local path in model_name.")
            raise
    
    def load_embeddings(self, json_path=None, npy_path=None):
        """Đọc embedding từ file JSON hoặc NumPy."""
        if json_path and os.path.exists(json_path):
            print(f"Loading embeddings from {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
            self.chunks = [
                {
                    'id': item['id'],
                    'content': item['content'],
                    'metadata': item['metadata']
                } for item in embedding_data
            ]
            self.embeddings = np.array([item['embedding'] for item in embedding_data], dtype=np.float32)
        elif npy_path and os.path.exists(npy_path):
            print(f"Loading embeddings from {npy_path}")
            self.embeddings = np.load(npy_path).astype(np.float32)
            # Tìm file chunk JSON trong data_dir
            chunk_files = glob(os.path.join(self.data_dir, '*.json'))
            if chunk_files:
                self.chunks = self._load_chunks(chunk_files[0])
            else:
                print("Warning: No chunk JSON files found. Metadata may be missing.")
        else:
            raise FileNotFoundError("No valid embedding file provided (JSON or NumPy).")
        
        print(f"Loaded {len(self.embeddings)} embeddings with dimension {self.embeddings.shape[1]}")
    
    def _load_chunks(self, chunk_file):
        """Đọc file chunk JSON."""
        with open(chunk_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def build_index(self):
        """Tạo FAISS index từ embeddings."""
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings first.")
        
        dimension = self.embeddings.shape[1]
        try:
            if self.index_type == 'flat_l2':
                self.index = faiss.IndexFlatL2(dimension)
            elif self.index_type == 'hnsw':
                self.index = faiss.IndexHNSWFlat(dimension, 32)
                self.index.hnsw.efConstruction = 40
            elif self.index_type == 'ivf':
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
                self.index.train(self.embeddings)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            self.index.add(self.embeddings)
            print(f"Built FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            raise
    
    def save_index(self, index_name='vector_store'):
        """Lưu FAISS index và chunks vào file."""
        index_path = os.path.join(self.output_dir, f'{index_name}.bin')
        faiss.write_index(self.index, index_path)
        print(f"Saved FAISS index to {index_path}")
        
        chunks_path = os.path.join(self.output_dir, f'{index_name}_chunks.json')
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved chunks to {chunks_path}")
    
    def add_vectors(self, new_chunks, batch_size=32):
        """Thêm vector mới vào index."""
        if self.model is None:
            self.load_model()
        
        new_contents = [chunk['content'] for chunk in new_chunks]
        passages = ["passage: " + content for content in new_contents]
        try:
            new_embeddings = []
            for i in tqdm(range(0, len(passages), batch_size), desc="Encoding new embeddings"):
                batch = passages[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size
                )
                new_embeddings.append(batch_embeddings)
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            new_embeddings = np.vstack(new_embeddings).astype(np.float32)
            
            self.chunks.extend(new_chunks)
            self.embeddings = np.vstack([self.embeddings, new_embeddings]) if self.embeddings is not None else new_embeddings
            self.build_index()
            print(f"Added {len(new_chunks)} new vectors to index")
        except Exception as e:
            print(f"Error adding new vectors: {e}")
            raise
    
    def search(self, query, k=3):
        """Tìm kiếm top-k chunk tương đồng với query."""
        if self.model is None:
            self.load_model()
        if self.index is None:
            raise ValueError("No index built. Call build_index first.")
        
        query_embedding = self.model.encode([f"passage: {query}"], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, k=k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'content': self.chunks[idx]['content'],
                'metadata': self.chunks[idx]['metadata'],
                'distance': distances[0][i]
            })
        return results
    
    def test_search(self, queries, k=3):
        """Thử tìm kiếm với danh sách câu hỏi."""
        for query in queries:
            print(f"\nQuery: {query}")
            results = self.search(query, k=k)
            print(f"Top {k} chunks:")
            for result in results:
                print(f"- {result['content'][:100]}... (distance: {result['distance']:.4f})")
                print(f"  Metadata: {result['metadata']}")
    
    def run(self, embedding_json=None, embedding_npy=None, queries=None):
        """Chạy quy trình tạo vector store."""
        self.check_environment()
        self.load_embeddings(json_path=embedding_json, npy_path=embedding_npy)
        self.build_index()
        self.save_index()
        
        if queries:
            self.test_search(queries)

if __name__ == "__main__":
    # Cấu hình cho môi trường local
    vector_store = VectorStoreLocal(
        model_name='intfloat/multilingual-e5-large',  # Hoặc đường dẫn local: 'models/multilingual-e5-large'
        data_dir='data',
        output_dir='output',
        index_type='flat_l2'
    )
    
    # Các câu hỏi thử nghiệm
    test_queries = [
        "Mã trường của Trường Đại học Giáo dục là gì?",
        "Triết lý giáo dục của ĐHQGHN là gì?",
        "Chỉ tiêu tuyển sinh năm 2024 của ngành Sư phạm Toán là bao nhiêu?"
    ]
    
    # Chạy quy trình
    vector_store.run(
        embedding_json='data/embeddings.json',
        # embedding_npy='data/embeddings.npy',  # Uncomment nếu dùng .npy
        queries=test_queries
    )