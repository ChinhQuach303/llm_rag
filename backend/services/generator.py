import logging
from typing import List, Dict, Any, Generator as TypeGenerator
import time
from openai import OpenAI
from core.config import LOCAL_LLM_URL, LOCAL_LLM_KEY, LOCAL_MODEL_NAME

logger = logging.getLogger(__name__)

class GeneratorService:
    """Manages context injection and strictly grounded LLM generation via vLLM."""
    def __init__(self):
        logger.info(f"Generator connecting to {LOCAL_LLM_URL} (Model: {LOCAL_MODEL_NAME})")
        try:
            self.client = OpenAI(base_url=LOCAL_LLM_URL, api_key=LOCAL_LLM_KEY)
            self.model_name = LOCAL_MODEL_NAME
        except Exception as e:
            logger.error(f"Failed to initialize Local LLM Client: {e}")
            self.client = None

    def evaluate_groundedness(self, top_chunk_score: float, threshold: float = 0.1) -> bool:
        is_grounded = top_chunk_score >= threshold
        if not is_grounded:
            logger.warning(f"Groundedness check FAILED. Top score {top_chunk_score:.4f} < {threshold}.")
        return is_grounded

    def create_strict_system_prompt(self, context_str: str) -> str:
        return (
            "You are a high-precision RAG assistant. Follow these rules strictly:\n"
            "1. Only answer based on the provided CONTEXT.\n"
            "2. If the answer is not in the context, output exactly: 'NOT FOUND'.\n"
            "3. Do not use external knowledge.\n\n"
            "--- FEW-SHOT EXAMPLES ---\n"
            "Query: What is the revenue in Q3?\n"
            "Context: [Source: Report] Q3 revenue reached $40M.\n"
            "Answer: The revenue in Q3 was $40M.\n\n"
            "Query: Who is the CEO?\n"
            "Context: [Source: Profile] Our company was founded in 1990.\n"
            "Answer: NOT FOUND\n"
            "--------------------------\n\n"
            "CONTEXT:\n"
            f"{context_str}"
        )

    def generate_stream(self, query: str, context_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = []) -> TypeGenerator[str, None, None]:
        context_blocks = [f"[Document {i+1} | Source: {c.get('section_path', 'Root')}]\n{c.get('text', '')}" for i, c in enumerate(context_chunks)]
        context_str = "\n\n".join(context_blocks)
        
        messages = [{"role": "system", "content": self.create_strict_system_prompt(context_str)}]
        
        for msg in chat_history[-3:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
        messages.append({"role": "user", "content": query})

        if not self.client:
            yield "NOT FOUND (Local LLM Client not initialized)"
            return

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=0.0
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Local LLM Execution Error: {e}")
            yield "NOT FOUND (Connection error to inference server)"
