from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    limit: int = 3
    alpha: float = 0.5  # For hybrid search weighting

class DocumentSource(BaseModel):
    content: str
    source: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentSource]
    query_time_ms: float

class HealthResponse(BaseModel):
    status: str
    message: str

class MetricsResponse(BaseModel):
    document_count: int
    total_requests: int
    average_latency_ms: float
    vector_db_status: str

class DocumentInfo(BaseModel):
    id: str
    source: str
    page_content: str  # Preview
    metadata: dict

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
