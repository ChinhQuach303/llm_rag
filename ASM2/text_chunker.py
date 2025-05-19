import json
import os
import re
import uuid
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

class TextChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        Initialize the text chunker with specified parameters.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_by_sentence(self, text):
        """Split text into sentences using NLTK's sent_tokenize."""
        # Use NLTK's sent_tokenize which is based on the punkt tokenizer
        sentences = sent_tokenize(text)
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text, metadata=None):
        """
        Create chunks from text with specified overlap.
        
        Args:
            text: The text to chunk
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        # Split text into sentences
        sentences = self.split_by_sentence(text)
        
        # Initialize chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed chunk size and we already have content
            if current_size + sentence_len > self.chunk_size and current_chunk:
                # Create a chunk from the current sentences
                chunk_text = ' '.join(current_chunk)
                chunk_dict = {
                    'chunk_id': str(uuid.uuid4()),
                    'text': chunk_text,
                    'chunk_size': len(chunk_text)
                }
                
                # Add metadata if provided
                if metadata:
                    chunk_dict.update(metadata)
                
                chunks.append(chunk_dict)
                
                # Start a new chunk with overlap
                # To create overlap, keep some sentences from the end of the previous chunk
                overlap_size = 0
                overlap_chunk = []
                
                # Work backwards through current_chunk to create overlap
                for s in reversed(current_chunk):
                    s_len = len(s)
                    if overlap_size + s_len <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += s_len + 1  # +1 for space
                    else:
                        break
                
                # Reset current chunk with overlap content
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_dict = {
                'chunk_id': str(uuid.uuid4()),
                'text': chunk_text,
                'chunk_size': len(chunk_text)
            }
            
            # Add metadata if provided
            if metadata:
                chunk_dict.update(metadata)
            
            chunks.append(chunk_dict)
        
        return chunks

def process_documents(input_file, output_dir, chunk_size=500, chunk_overlap=50):
    """
    Process documents from cleaned data JSON file and create chunks.
    
    Args:
        input_file: Path to cleaned data JSON file
        output_dir: Directory to save chunked output
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Load cleaned data
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    all_chunks = []
    
    # Process each document
    for doc in tqdm(documents, desc="Chunking documents"):
        # Extract document content and metadata
        document_id = doc['id']
        content = doc['content']
        
        # Create metadata for chunks
        chunk_metadata = {
            'document_id': document_id,
            'title': doc.get('title', ''),
            'url': doc.get('url', ''),
            'category': doc.get('category', ''),
            'subcategory': doc.get('subcategory', '')
        }
        
        # Create chunks for the document
        doc_chunks = chunker.create_chunks(content, chunk_metadata)
        
        # Add document chunks to all chunks
        all_chunks.extend(doc_chunks)
    
    # Save all chunks to a JSON file
    output_file = os.path.join(output_dir, 'chunked_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents. Saved to {output_file}")
    return output_file

if __name__ == "__main__":
    input_file = "processed_data/cleaned_data.json"
    output_dir = "chunked_data"
    
    # You can adjust chunk size and overlap based on your needs
    process_documents(input_file, output_dir, chunk_size=500, chunk_overlap=50) 