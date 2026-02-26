import os
import logging
import base64
import httpx
import tiktoken
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from db.models import DocumentRecord
from core.config import DEEPSEEK_API_KEY
from pdf2image import convert_from_path
import re
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

class HierarchicalChunk:
    def __init__(self, doc_id: str, text: str, section_path: str, depth: int, start_page: int):
        self.doc_id = doc_id
        self.text = text
        self.section_path = section_path
        self.depth = depth
        self.start_page = start_page

class DocumentParserService:
    """
    Parses PDF using DeepSeek OCR.
    Includes robust retries (tenacity) and smart chunking via tiktoken.
    """
    def __init__(self, db_session: Session):
        self.db = db_session
        self.api_key = DEEPSEEK_API_KEY
        self.api_url = "https://api.deepseek.com/v1/chat/completions" # Generic endpoint placeholder
        try:
             self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
             self.encoder = None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(httpx.HTTPError)
    )
    def _call_deepseek_ocr(self, base64_image: str) -> str:
        """Call DeepSeek Vision API to get Markdown with exponential backoff."""
        if not self.api_key:
             return "Mock OCR Content: DeepSeek API Key not provided.\n\n# Header 1\nThis is mock section 1."
             
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-vl2", # Placeholder for the vision model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text and tables from this image and format strictly as Markdown. Do not add any conversational text."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.0
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def process_and_save(self, file_path: str, original_filename: str, tenant_id: str) -> Optional[str]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None

        logger.info(f"Processing '{original_filename}' for tenant '{tenant_id}' using DeepSeek OCR...")

        new_doc = DocumentRecord(
            tenant_id=tenant_id,
            filename=original_filename,
            status="PROCESSING"
        )
        self.db.add(new_doc)
        self.db.commit()
        self.db.refresh(new_doc)
        doc_id = str(new_doc.id)

        try:
            # 1. Convert PDF to images
            if file_path.lower().endswith(".pdf"):
                images = convert_from_path(file_path, dpi=200)
            else:
                 from PIL import Image
                 images = [Image.open(file_path)]

            full_markdown = []
            
            # 2. Extract OCR for each page
            for i, img in enumerate(images):
                 from io import BytesIO
                 img_byte_arr = BytesIO()
                 img.save(img_byte_arr, format='JPEG')
                 base64_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                 
                 page_md = self._call_deepseek_ocr(base64_img)
                 full_markdown.append({"page": i+1, "markdown": page_md})
            
            new_doc.raw_content = {"pages": full_markdown}
            new_doc.status = "COMPLETED"
            self.db.commit()
            
            logger.info(f"Successfully processed and saved {original_filename}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to process {original_filename}: {e}")
            new_doc.status = "FAILED"
            self.db.commit()
            return None

    def _split_text_by_tokens(self, text: str, max_tokens: int = 512) -> List[str]:
        """Fall-back chunking strategy for massive unformatted OCR text blocks."""
        if not self.encoder or not text:
            return [text]
            
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
            
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(self.encoder.decode(chunk_tokens))
        return chunks

    def chunk_document(self, doc_id: str) -> List[HierarchicalChunk]:
        """Performs hierarchical text splitting and enforces token limit via tiktoken."""
        doc = self.db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
        
        if not doc or doc.status != "COMPLETED" or not doc.raw_content:
            return []
            
        chunks = []
        pages = doc.raw_content.get("pages", [])
        
        current_path = []
        
        for page_data in pages:
            page_no = page_data.get("page", 1)
            markdown_content = page_data.get("markdown", "")
            
            lines = markdown_content.split("\n")
            current_buffer = []
            
            def save_buffer(buffer, path, page):
                text_block = "\n".join(buffer).strip()
                if not text_block: return
                path_str = " > ".join(path) if path else "Root"
                
                # Smart Token Split
                sub_chunks = self._split_text_by_tokens(text_block, 512)
                for sub_chunk in sub_chunks:
                     chunks.append(HierarchicalChunk(
                         doc_id=doc_id, text=sub_chunk,
                         section_path=path_str, depth=len(path), start_page=page
                     ))
            
            for line in lines:
                header_match = re.match(r'^(#{1,6})\s+(.*)', line)
                if header_match:
                    save_buffer(current_buffer, current_path, page_no)
                    current_buffer = []
                        
                    level = len(header_match.group(1))
                    header_text = header_match.group(2).strip()
                    
                    current_path = current_path[:level-1]
                    current_path.append(header_text)
                else:
                    if line.strip() or current_buffer: # Keep some empty lines inside blocks
                        current_buffer.append(line)
                        
            if current_buffer:
                save_buffer(current_buffer, current_path, page_no)

        logger.info(f"Split document into {len(chunks)} Markdown limits chunks.")
        return chunks
