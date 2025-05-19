import torch
import json
import os
from glob import glob
from embedding_kaggle_script import EmbeddingGeneratorKaggle
from vector_store_kaggle_script import VectorStoreKaggle
from qa_chain_kaggle_script import QAChainKaggle

class RAGPipelineKaggle:
    def __init__(self, model_name='intfloat/multilingual-e5-large', llm_model_name='google/gemma-2b', data_dir='/kaggle/input/uet-rag-data', output_dir='/kaggle/working', batch_size=32, device=None, index_type='hnsw'):
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_type = index_type
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_environment(self):
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}, Count: {torch.cuda.device_count()}")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
    
    def chunk_text(self, max_length=512, min_length=50):
        text_files = glob(os.path.join(self.data_dir, '*.txt'))
        if not text_files:
            print("No text files found. Skipping chunking.")
            return
        
        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                sentences = text.split('. ')
                chunks = []
                current_chunk = ""
                chunk_id = 0
                for sentence in sentences:
                    sentence = sentence.strip() + '. '
                    if len(current_chunk) + len(sentence) <= max_length:
                        current_chunk += sentence
                    else:
                        if len(current_chunk) >= min_length:
                            chunks.append({
                                'id': f"{os.path.basename(text_file).split('.')[0]}_{chunk_id}",
                                'content': current_chunk.strip(),
                                'metadata': {'source': text_file, 'type': 'text'}
                            })
                            chunk_id += 1
                        current_chunk = sentence
                if len(current_chunk) >= min_length:
                    chunks.append({
                        'id': f"{os.path.basename(text_file).split('.')[0]}_{chunk_id}",
                        'content': current_chunk.strip(),
                        'metadata': {'source': text_file, 'type': 'text'}
                    })
                
                chunk_file = os.path.join(self.output_dir, f"chunks_{os.path.basename(text_file).split('.')[0]}.json")
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                print(f"Saved {len(chunks)} chunks to {chunk_file}")
            except Exception as e:
                print(f"Error chunking {text_file}: {e}")
    
    def generate_embeddings(self, offline_model_path=None):
        generator = EmbeddingGeneratorKaggle(
            model_name=self.model_name,
            batch_size=self.batch_size,
            data_dir=self.output_dir,
            output_dir=self.output_dir,
            device=self.device
        )
        try:
            generator.run(offline_model_path=offline_model_path)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def build_vector_store(self, offline_model_path=None):
        vector_store = VectorStoreKaggle(
            model_name=self.model_name,
            data_dir=self.output_dir,
            output_dir=self.output_dir,
            device=self.device,
            index_type=self.index_type
        )
        try:
            vector_store.run(
                embedding_json=os.path.join(self.output_dir, 'embeddings.json'),
                offline_model_path=offline_model_path
            )
        except Exception as e:
            print(f"Error building vector store: {e}")
            raise
    
    def test_pipeline(self, queries, offline_model_path=None):
        qa_chain = QAChainKaggle(
            embedding_model_name=self.model_name,
            llm_model_name=self.llm_model_name,
            data_dir=self.output_dir,
            device=self.device,
            k=3
        )
        try:
            qa_chain.run(
                queries=queries,
                offline_model_path=offline_model_path
            )
        except Exception as e:
            print(f"Error testing pipeline: {e}")
            raise
    
    def run(self, run_chunking=True, queries=None, offline_model_path=None):
        self.check_environment()
        if run_chunking:
            print("Step 1: Chunking text files...")
            self.chunk_text()
        print("Step 2: Generating embeddings...")
        self.generate_embeddings(offline_model_path=offline_model_path)
        print("Step 3: Building vector store...")
        self.build_vector_store(offline_model_path=offline_model_path)
        if queries:
            print("Step 4: Testing pipeline with QA...")
            self.test_pipeline(queries, offline_model_path=offline_model_path)

if __name__ == "__main__":
    pipeline = RAGPipelineKaggle(
        model_name='intfloat/multilingual-e5-large',
        llm_model_name='google/gemma-2b',
        data_dir='/kaggle/input/uet-rag-data',
        output_dir='/kaggle/working',
        batch_size=32,
        index_type='hnsw'
    )
    
    test_queries = [
        "Mã trường của Trường Đại học Giáo dục là gì?",
        "Triết lý giáo dục của ĐHQGHN là gì?",
        "Chỉ tiêu tuyển sinh năm 2024 của ngành Sư phạm Toán là bao nhiêu?",
        "Phương thức xét tuyển của ĐHQGHN năm 2024 là gì?"
    ]
    
    pipeline.run(
        run_chunking=True,
        queries=test_queries,
        offline_model_path='/kaggle/input/multilingual-e5-large-model/multilingual-e5-large'
    )