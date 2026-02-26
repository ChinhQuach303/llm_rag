from pydantic import BaseModel
from typing import List, Dict, Any

class QueryResponse(BaseModel):
    answer: str
    context_used: List[Dict[str, Any]] = []

class IngestResponse(BaseModel):
    task_id: str
    status: str
    message: str
