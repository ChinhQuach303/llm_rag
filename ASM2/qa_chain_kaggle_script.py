
import json
import numpy as np
import torch
import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from vector_store_kaggle_script import VectorStoreKaggle
from tqdm import tqdm

class QAChainKaggle:
    def __init__(self, 
                 embedding_model_name='intfloat/multilingual-e5-large',
                 llm_model_name='google/gemma-2b',
                 data_dir='/kaggle/working',
                 device=None,
                 max_new_tokens=150,
                 k=3):
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
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}, Count: {torch.cuda.device_count()}")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
    
    def load_vector_store(self, offline_model_path=None):
        try:
            self.vector_store = VectorStoreKaggle(
                model_name=self.embedding_model_name,
                data_dir=self.data_dir,
                output_dir=self.data_dir,
                device=self.device,
                index_type='hnsw'
            )
            index_path = os.path.join(self.data_dir, 'vector_store.bin')
            chunks_path = os.path.join(self.data_dir, 'vector_store_chunks.json')
            if not os.path.exists(index_path) or not os.path.exists(chunks_path):
                raise FileNotFoundError("Vector store files not found.")
            self.vector_store.index = faiss.read_index(index_path)
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.vector_store.chunks = json.load(f)
            self.vector_store.load_model(offline_path=offline_model_path)
            print(f"Loaded vector store with {self.vector_store.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise
    
    def load_llm(self):
        try:
            print(f"Loading LLM: {self.llm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16,
                device_map='auto'
            )
            print("LLM loaded successfully!")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise
    
    def retrieve_chunks(self, query):
        try:
            return self.vector_store.search(query, k=self.k)
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def generate_answer(self, query, chunks):
        if not chunks:
            return "Không tìm thấy thông tin liên quan."
        if self.llm is None or self.tokenizer is None:
            self.load_llm()
        
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
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.split("**Câu trả lời:**")[-1].strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Lỗi khi sinh câu trả lời."
    
    def answer_query(self, query):
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
        for query in queries:
            self.answer_query(query)
    
    def run(self, queries=None, offline_model_path=None):
        self.check_environment()
        self.load_vector_store(offline_model_path=offline_model_path)
        if queries:
            print("Testing QA chain...")
            self.test_qa(queries)