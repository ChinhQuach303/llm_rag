import json
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from glob import glob

class EmbeddingGeneratorLocal:
    def __init__(self, model_name='intfloat/multilingual-e5-large', batch_size=32, data_dir='data', output_dir='output', device=None):
        """
        Khởi tạo lớp tạo embedding cho môi trường local.
        Args:
            model_name (str): Tên model hoặc đường dẫn local (mặc định: multilingual-e5-large).
            batch_size (int): Kích thước batch để encode.
            data_dir (str): Thư mục chứa file chunk (JSON).
            output_dir (str): Thư mục lưu kết quả (embeddings).
            device (str): Thiết bị ('cuda', 'cpu', hoặc None để tự động chọn).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_environment(self):
        """Kiểm tra môi trường: GPU, thư mục đầu vào."""
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}, Count: {torch.cuda.device_count()}")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
    
    def load_model(self):
        """Tải model để tạo embedding."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Ensure model is downloaded or specify local path in model_name.")
            raise
    
    def load_chunks(self, pattern="*.json"):
        """Đọc tất cả file chunk (JSON) từ thư mục đầu vào."""
        chunk_files = glob(os.path.join(self.data_dir, pattern))
        if not chunk_files:
            raise FileNotFoundError(f"No JSON chunk files found in {self.data_dir}")
        
        all_chunks = []
        all_contents = []
        for file_path in chunk_files:
            try:
                chunks = self._load_single_chunk_file(file_path)
                all_chunks.extend(chunks)
                all_contents.extend([chunk['content'] for chunk in chunks])
                print(f"Loaded {len(chunks)} chunks from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return all_chunks, all_contents
    
    def _load_single_chunk_file(self, file_path):
        """Đọc một file chunk JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_embeddings(self, contents):
        """Tạo embedding cho danh sách nội dung."""
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
        """Lưu embedding vào file JSON và NumPy."""
        # Lưu embedding JSON (với metadata)
        embedding_data = [
            {
                'id': chunk['id'],
                'content': chunk['content'],
                'embedding': embedding.tolist(),
                'metadata': chunk['metadata']
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        json_path = os.path.join(self.output_dir, f'{output_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, ensure_ascii=False, indent=2)
        print(f"Saved embeddings to {json_path}")
        
        # Lưu embedding NumPy (vector thô)
        npy_path = os.path.join(self.output_dir, f'{output_name}.npy')
        np.save(npy_path, embeddings)
        print(f"Saved embeddings to {npy_path}")
    
    def check_embeddings(self, embeddings, contents, num_samples=5):
        """Kiểm tra chất lượng embedding."""
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
    
    def test_search(self, queries, chunks, embeddings, k=3):
        """Thử tìm kiếm với các câu hỏi mẫu (sử dụng cosine similarity)."""
        for query in queries:
            query_embedding = self.model.encode([f"passage: {query}"], normalize_embeddings=True)
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            top_k = np.argsort(similarities)[-k:][::-1]
            print(f"\nQuery: {query}")
            print(f"Top {k} chunks:")
            for i, idx in enumerate(top_k):
                print(f"- {chunks[idx]['content'][:100]}... (similarity: {similarities[idx]:.4f})")
                print(f"  Metadata: {chunks[idx]['metadata']}")
    
    def run(self, queries=None):
        """Chạy toàn bộ quy trình tạo embedding."""
        self.check_environment()
        chunks, contents = self.load_chunks()
        embeddings = self.create_embeddings(contents)
        self.save_embeddings(chunks, embeddings)
        self.check_embeddings(embeddings, contents)
        
        if queries:
            self.test_search(queries, chunks, embeddings)

if __name__ == "__main__":
    # Cấu hình cho môi trường local
    generator = EmbeddingGeneratorLocal(
        model_name='intfloat/multilingual-e5-large',  # Hoặc đường dẫn local: 'models/multilingual-e5-large'
        batch_size=32,
        data_dir='data',
        output_dir='output'
    )
    
    # Các câu hỏi thử nghiệm
    test_queries = [
        "Giới thiệu chung về Trường Đại Học Khoa Học Xã hội và nhân văn ?",
        "Triết lý giáo dục của ĐHQGHN là gì?",
        "Chỉ tiêu tuyển sinh năm 2024 của ngành Sư phạm Toán Trường Đại Học Khoa Học Tự Nhiên là bao nhiêu?"
    ]
    
    # Chạy quy trình
    generator.run(queries=test_queries)