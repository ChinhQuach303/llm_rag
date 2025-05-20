import os
import json
import logging
import numpy as np
from typing import List, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def setup_logging():
    """Thiết lập logging"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def load_chunks(file_paths: List[str]) -> List[Dict]:
    """
    Đọc chunks từ danh sách các file JSON.
    
    Args:
        file_paths: Danh sách đường dẫn tới các file JSON.
    
    Returns:
        Danh sách các chunks từ tất cả file.
    """
    logger = setup_logging()
    all_chunks = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} không tồn tại.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
                logger.info(f"Đã tải {len(chunks)} chunks từ {file_path}")
        except Exception as e:
            logger.error(f"Lỗi khi tải file {file_path}: {e}")
    return all_chunks

def generate_chunks(data: str, source: str = "unknown", max_words_per_chunk: int = 200) -> List[Dict]:
    """
    Tạo chunks từ văn bản thô, phân loại theo ngữ nghĩa và gán metadata chi tiết.
    
    Args:
        data: Văn bản thô cần chia nhỏ.
        source: Nguồn dữ liệu (URL hoặc tên file).
        max_words_per_chunk: Số từ tối đa cho mỗi chunk.
    
    Returns:
        Danh sách các chunks với content và metadata.
    """
    logger = setup_logging()
    chunks = []
    lines = [line.strip() for line in data.split('\n') if line.strip()]
    
    current_chunk = []
    current_word_count = 0
    chunk_id = 0
    metadata = {
        "source": source,
        "timestamp": datetime.now().isoformat(),
        "category": "general"  # Mặc định, sẽ được cập nhật theo loại nội dung
    }
    
    for line in lines:
        # Xác định loại nội dung dựa trên từ khóa và cấu trúc
        words = line.split()
        line_length = len(words)
        
        # Phân loại: Header (tiêu đề ngắn, thường chứa từ khóa VNU hoặc tên trường)
        if line_length < 15 and any(keyword in line.lower() for keyword in ['giới thiệu', 'lịch sử', 'tuyển sinh', 'đào tạo', 'trường']):
            if current_chunk:
                chunk_content = " ".join(current_chunk).strip()
                if chunk_content:
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "content": chunk_content,
                        "metadata": {**metadata, "category": metadata["category"]}
                    })
                    chunk_id += 1
                current_chunk = []
                current_word_count = 0
            # Gán category dựa trên từ khóa
            category = "history" if "lịch sử" in line.lower() else \
                      "admission" if "tuyển sinh" in line.lower() else \
                      "training" if "đào tạo" in line.lower() else \
                      "member_school" if "trường" in line.lower() else "header"
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "content": line,
                "metadata": {**metadata, "type": "header", "category": category}
            })
            chunk_id += 1
            continue
        
        # Phân loại: Danh sách (ngành học, trường thành viên, phương án xét tuyển)
        if line.startswith(('Ngành ', 'Trường ', 'Phương án ', '- ', '• ', '* ')):
            if current_chunk:
                chunk_content = " ".join(current_chunk).strip()
                if chunk_content:
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "content": chunk_content,
                        "metadata": {**metadata, "category": metadata["category"]}
                    })
                    chunk_id += 1
                current_chunk = []
                current_word_count = 0
            category = "training" if "ngành" in line.lower() else \
                      "member_school" if "trường" in line.lower() else \
                      "admission" if "phương án" in line.lower() else "list"
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "content": line,
                "metadata": {**metadata, "type": "list", "category": category}
            })
            chunk_id += 1
            continue
        
        # Phân loại: Số liệu (năm, chỉ tiêu, số lượng sinh viên)
        if any(word.isdigit() for word in words) and line_length < 20:
            if current_chunk:
                chunk_content = " ".join(current_chunk).strip()
                if chunk_content:
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "content": chunk_content,
                        "metadata": {**metadata, "category": metadata["category"]}
                    })
                    chunk_id += 1
                current_chunk = []
                current_word_count = 0
            category = "history" if any(year in line for year in ["1945", "1956", "1995"]) else \
                      "admission" if "chỉ tiêu" in line.lower() else "general"
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "content": line,
                "metadata": {**metadata, "type": "number", "category": category}
            })
            chunk_id += 1
            continue
        
        # Thêm vào chunk văn bản
        current_chunk.append(line)
        current_word_count += line_length
        
        # Nếu vượt quá max_words_per_chunk, lưu chunk
        if current_word_count >= max_words_per_chunk:
            chunk_content = " ".join(current_chunk).strip()
            if chunk_content:
                # Xác định category dựa trên nội dung chunk
                category = "history" if any(keyword in chunk_content.lower() for keyword in ["lịch sử", "thành lập", "phát triển"]) else \
                          "admission" if any(keyword in chunk_content.lower() for keyword in ["tuyển sinh", "xét tuyển", "chỉ tiêu"]) else \
                          "training" if any(keyword in chunk_content.lower() for keyword in ["đào tạo", "ngành", "chương trình"]) else \
                          "member_school" if any(keyword in chunk_content.lower() for keyword in ["trường", "thành viên"]) else "general"
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "content": chunk_content,
                    "metadata": {**metadata, "type": "text", "category": category}
                })
                chunk_id += 1
            current_chunk = []
            current_word_count = 0
    
    # Lưu chunk cuối cùng
    if current_chunk:
        chunk_content = " ".join(current_chunk).strip()
        if chunk_content:
            category = "history" if any(keyword in chunk_content.lower() for keyword in ["lịch sử", "thành lập", "phát triển"]) else \
                      "admission" if any(keyword in chunk_content.lower() for keyword in ["tuyển sinh", "xét tuyển", "chỉ tiêu"]) else \
                      "training" if any(keyword in chunk_content.lower() for keyword in ["đào tạo", "ngành", "chương trình"]) else \
                      "member_school" if any(keyword in chunk_content.lower() for keyword in ["trường", "thành viên"]) else "general"
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "content": chunk_content,
                "metadata": {**metadata, "type": "text", "category": category}
            })
    
    logger.info(f"Tạo được {len(chunks)} chunks từ dữ liệu")
    return chunks

def save_embeddings(embedding_data: List[Dict], output_file: str):
    """
    Lưu embedding và thông tin chunk vào file JSON.
    
    Args:
        embedding_data: Danh sách các dictionary chứa id, content, embedding, metadata.
        output_file: Đường dẫn file JSON đầu ra.
    """
    logger = setup_logging()
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã lưu embeddings vào {output_file}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu embeddings: {e}")

def check_embeddings(embeddings: np.ndarray, chunks: List[Dict], num_samples: int = 5):
    """
    Kiểm tra embeddings và hiển thị các chunks tương tự.
    
    Args:
        embeddings: Ma trận embeddings.
        chunks: Danh sách chunks với content và metadata.
        num_samples: Số mẫu để kiểm tra.
    """
    logger = setup_logging()
    logger.info(f"Total embeddings: {len(embeddings)}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"Norms (should be ~1): min={norms.min():.4f}, max={norms.max():.4f}")
    
    sample_idx = np.random.choice(len(embeddings), num_samples, replace=False)
    for idx in sample_idx:
        similarities = cosine_similarity([embeddings[idx]], embeddings)[0]
        top_k = np.argsort(similarities)[-3:][::-1]
        logger.info(f"\nChunk: {chunks[idx]['content'][:100]}...")
        logger.info("Top 3 similar chunks:")
        for k in top_k:
            logger.info(f"- {chunks[k]['content'][:100]}... (similarity: {similarities[k]:.4f})")
            logger.info(f"  Metadata: {chunks[k]['metadata']}")

def process_vnu_data(input_files: List[str], output_dir: str, model_name: str = 'intfloat/multilingual-e5-large'):
    """
    Xử lý dữ liệu VNU, tạo chunks, embeddings và lưu kết quả.
    
    Args:
        input_files: Danh sách file văn bản thô hoặc JSON.
        output_dir: Thư mục lưu kết quả.
        model_name: Tên mô hình SentenceTransformer.
    """
    logger = setup_logging()
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo model
    logger.info(f"Khởi tạo mô hình {model_name}")
    model = SentenceTransformer(model_name)
    
    # Tải và xử lý dữ liệu
    all_chunks = []
    for input_file in input_files:
        if not os.path.exists(input_file):
            logger.warning(f"File {input_file} không tồn tại.")
            continue
        
        # Đọc văn bản thô
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            logger.warning(f"File {input_file} rỗng.")
            continue
        
        # Tạo chunks
        chunks = generate_chunks(content, source=input_file)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        logger.error("Không tạo được chunks từ dữ liệu.")
        return
    
    # Tạo embeddings
    passages = ["passage: " + chunk['content'] for chunk in all_chunks]
    logger.info(f"Tạo embeddings cho {len(passages)} chunks")
    embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings)
    
    # Kết hợp chunks và embeddings
    embedding_data = [
        {
            'id': chunk['id'],
            'content': chunk['content'],
            'embedding': embedding.tolist(),
            'metadata': chunk['metadata']
        }
        for chunk, embedding in zip(all_chunks, embeddings)
    ]
    
    # Lưu embeddings
    output_file = os.path.join(output_dir, 'embeddings.json')
    save_embeddings(embedding_data, output_file)
    
    # Kiểm tra embeddings
    check_embeddings(embeddings, all_chunks)

if __name__ == "__main__":
    input_files = [
        'data/output_text.txt',  
        'data/data.txt'# File chứa dữ liệu VNU
        # Thêm các file khác nếu có
    ]
    output_dir = 'output'
    process_vnu_data(input_files, output_dir)