import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_embeddings(embedding_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_data, f, ensure_ascii=False, indent=2)

def check_embeddings(embeddings, contents, num_samples=5):
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

# Đọc chunks
output_chunks = load_chunks('data/chunks_output_text.json')
data_chunks = load_chunks('data/chunks_data.json')
all_chunks = output_chunks + data_chunks
all_contents = [chunk['content'] for chunk in all_chunks]

# Khởi tạo model
model = SentenceTransformer('intfloat/multilingual-e5-large')

# Thêm tiền tố "passage: "
passages = ["passage: " + content for content in all_contents]

# Tạo embedding
embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
embeddings = np.array(embeddings)

# Lưu embedding cùng thông tin chunk
embedding_data = [
    {
        'id': chunk['id'],
        'content': chunk['content'],
        'embedding': embedding.tolist(),
        'metadata': chunk['metadata']
    }
    for chunk, embedding in zip(all_chunks, embeddings)
]

# Lưu kết quả
save_embeddings(embedding_data, 'data/embeddings.json')
np.save('data/embeddings.npy', embeddings)

# Kiểm tra chất lượng
check_embeddings(embeddings, all_contents)