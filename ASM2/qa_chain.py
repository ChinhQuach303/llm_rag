import json
import numpy as np
import torch
import os
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
                 k=5):
        """
        Khởi tạo QA Chain cho RAG trên local, không dùng FAISS.
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
    
    def load_vector_store(self, offline_embedding_path=None):
        """Tải vector store từ JSON hoặc NumPy (không dùng FAISS)."""
        try:
            self.vector_store = VectorStoreLocal(
                model_name=self.embedding_model_name,
                data_dir=self.data_dir,
                output_dir=self.data_dir,
                device=self.device
            )
            # Tải embeddings và chunks
            json_path = os.path.join(self.data_dir, 'vector_store.json')
            npy_path = os.path.join(self.data_dir, 'vector_store.npy')
            if not os.path.exists(json_path) and not os.path.exists(npy_path):
                raise FileNotFoundError("Vector store files (vector_store.json or vector_store.npy) not found. Run vector store generation first.")
            
            self.vector_store.load_embeddings(json_path=json_path, npy_path=npy_path)
            print(f"Loaded vector store with {len(self.vector_store.embeddings)} vectors")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
    
    def load_llm(self, offline_llm_path=None):
        """Tải LLM và tokenizer."""
        try:
            print(f"Loading LLM: {self.llm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                offline_llm_path if offline_llm_path else self.llm_model_name
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                offline_llm_path if offline_llm_path else self.llm_model_name,
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
        
        # Tạo prompt cải tiến
        context = "\n".join([f"- {chunk['content']}" for chunk in chunks])
        prompt = f"""Dựa trên thông tin sau, trả lời câu hỏi bằng tiếng Việt, ngắn gọn, chính xác và giữ nguyên thông tin gốc nếu có:

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
            print(f"- {chunk['content'][:100]}... (similarity: {chunk['similarity']:.4f})")
            print(f"  Metadata: {chunk['metadata']}")
        
        answer = self.generate_answer(query, chunks)
        print(f"Answer: {answer}")
        return answer
    
    def test_qa(self, queries):
        """Thử nghiệm QA chain với danh sách câu hỏi."""
        for query in queries:
            self.answer_query(query)
    
    def run(self, queries=None, offline_embedding_path=None, offline_llm_path=None):
        """Chạy QA chain."""
        self.check_environment()
        self.load_vector_store(offline_embedding_path=offline_embedding_path)
        if queries:
            print("Testing QA chain with queries...")
            self.test_qa(queries)

if __name__ == "__main__":
    # Cấu hình QA chain
    qa_chain = QAChain(
        embedding_model_name='intfloat/multilingual-e5-large',
        llm_model_name='mistralai/Mistral-7B-v0.1',  
        data_dir='output',
        max_new_tokens=150,
        k=5
    )
        
    # Các câu hỏi thử nghiệm
    test_queries = [
        "Mã trường của Trường Đại học Giáo dục là gì?",
        "Triết lý giáo dục của ĐHQGHN là gì?",
        "Chỉ tiêu tuyển sinh năm 2024 của ngành Sư phạm Toán là bao nhiêu?",
        "Phương thức xét tuyển của ĐHQGHN năm 2024 là gì?"
    ]
    
    # Chạy QA chain
    qa_chain.run(
        queries=test_queries,
        offline_embedding_path='models/multilingual-e5-large',  # Thay bằng đường dẫn local nếu có
        offline_llm_path='models/gemma-2b'  # Thay bằng đường dẫn local nếu có
    )