import json
import numpy as np
import torch
import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vector_store import VectorStoreLocal
from tqdm import tqdm

class QAChain:
    def __init__(self, 
                 embedding_model_name='intfloat/multilingual-e5-large',
                 llm_model_name='google/gemma-2b',
                 data_dir='output',
                 device=None,
                 max_new_tokens=150,
                 k=3):
        """
        Khởi tạo QA Chain cho RAG trên local.
        Args:
            embedding_model_name (str): Tên hoặc đường dẫn embedding model.
            llm_model_name (str): Tên hoặc đường dẫn LLM.
            data_dir (str): Thư mục chứa vector store và chunks.
            device (str): Thiết bị ('cuda', 'cpu', hoặc None để tự động chọn).
            max_new_tokens (int): Số token tối đa cho LLM.
            k (int): Số chunk truy xuất.
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.data_dir = data_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = max_new_tokens
        self.k = k
        self.vector_store = None
        self.llm = None
        self.tokenizer = None
    
    def check_environment(self):
        """Kiểm tra môi trường."""
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}, Count: {torch.cuda.device_count()}")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
    
    def load_vector_store(self):
        """Tải vector store từ FAISS index."""
        try:
            self.vector_store = VectorStoreLocal(
                model_name=self.embedding_model_name,
                data_dir=self.data_dir,
                output_dir=self.data_dir,
                device=self.device,
                index_type='hnsw'
            )
            # Tải index có sẵn
            index_path = os.path.join(self.data_dir, 'vector_store.bin')
            chunks_path = os.path.join(self.data_dir, 'vector_store_chunks.json')
            if not os.path.exists(index_path) or not os.path.exists(chunks_path):
                raise FileNotFoundError("Vector store files not found. Run main.py first.")
            
            self.vector_store.index = faiss.read_index(index_path)
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.vector_store.chunks = json.load(f)
            self.vector_store.load_model()
            print(f"Loaded vector store with {self.vector_store.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
    
    def load_llm(self):
        """Tải LLM và tokenizer."""
        try:
            print(f"Loading LLM: {self.llm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto'
            )
            print("LLM loaded successfully!")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Ensure LLM is downloaded or specify local path.")
            raise
    
    def retrieve_chunks(self, query):
        """Truy xuất top-k chunk liên quan đến câu hỏi."""
        try:
            results = self.vector_store.search(query, k=self.k)
            return results
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def generate_answer(self, query, chunks):
        """Sinh câu trả lời từ query và chunks."""
        if not chunks:
            return "Không tìm thấy thông tin liên quan."
        
        if self.llm is None or self.tokenizer is None:
            self.load_llm()
        
        # Tạo prompt
        context = "\n".join([f"- {chunk['content']}" for chunk in chunks])
        prompt = f"""Dựa trên thông tin sau, trả lời câu hỏi một cách ngắn gọn và chính xác bằng tiếng Việt:

**Thông tin:**
{context}

**Câu hỏi:** {query}

**Câu trả lời:** """
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Lấy phần câu trả lời sau prompt
            answer = answer.split("**Câu trả lời:**")[-1].strip()
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Lỗi khi sinh câu trả lời."
    
    def answer_query(self, query):
        """Trả lời một câu hỏi."""
        print(f"\nQuery: {query}")
        chunks = self.retrieve_chunks(query)
        if not chunks:
            print("No relevant chunks found.")
            return "Không tìm thấy thông tin liên quan."
        
        print(f"Top {self.k} chunks:")
        for chunk in chunks:
            print(f"- {chunk['content'][:100]}... (distance: {chunk['distance']:.4f})")
            print(f"  Metadata: {chunk['metadata']}")
        
        answer = self.generate_answer(query, chunks)
        print(f"Answer: {answer}")
        return answer
    
    def test_qa(self, queries):
        """Thử nghiệm QA chain với danh sách câu hỏi."""
        for query in queries:
            self.answer_query(query)
    
    def run(self, queries=None):
        """Chạy QA chain."""
        self.check_environment()
        self.load_vector_store()
        
        if queries:
            print("Testing QA chain with queries...")
            self.test_qa(queries)

if __name__ == "__main__":
    # Cấu hình QA chain
    qa_chain = QAChain(
        embedding_model_name='intfloat/multilingual-e5-large',  # Hoặc 'models/multilingual-e5-large'
        llm_model_name='google/gemma-2b',  # Hoặc 'models/gemma-2b'
        data_dir='output',
        max_new_tokens=150,
        k=3
    )
    
    # Các câu hỏi thử nghiệm
    test_queries = [
        "Mã trường của Trường Đại học Giáo dục là gì?",
        "Triết lý giáo dục của ĐHQGHN là gì?",
        "Chỉ tiêu tuyển sinh năm 2024 của ngành Sư phạm Toán là bao nhiêu?",
        "Phương thức xét tuyển của ĐHQGHN năm 2024 là gì?"
    ]
    
    # Chạy QA chain
    qa_chain.run(queries=test_queries)