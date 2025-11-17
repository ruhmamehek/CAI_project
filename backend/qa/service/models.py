"""Data models for RAG service."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .utils import clean_image_text


@dataclass
class Source:
    """Source information for a retrieved chunk."""
    ticker: str
    year: str
    score: float
    chunk_id: str
    chunk_type: str = "Text"
    page: int = 0
    image_path: Optional[str] = None
    text: Optional[str] = None


@dataclass
class QueryRequest:
    """Request model for RAG query."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryRequest':
        """Create QueryRequest from dictionary."""
        return cls(
            query=data.get('query', ''),
            filters=data.get('filters'),
            top_k=data.get('top_k')
        )
    
    def validate(self) -> None:
        """Validate request data."""
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")


@dataclass
class QueryResponse:
    """Response model for RAG query."""
    answer: str
    sources: List[Source]
    num_chunks_retrieved: int
    verification: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "answer": self.answer,
            "sources": [
                {
                    "ticker": source.ticker,
                    "year": source.year,
                    "score": source.score,
                    "chunk_id": source.chunk_id,
                    "chunk_type": source.chunk_type,
                    "page": source.page,
                    "image_path": source.image_path,
                    "text": source.text
                }
                for source in self.sources
            ],
            "num_chunks_retrieved": self.num_chunks_retrieved
        }
        if self.verification:
            result["verification"] = self.verification
        return result


@dataclass
class Chunk:
    """Retrieved chunk from vector database."""
    text: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
    
    def to_source(self) -> Source:
        """Convert chunk to Source."""
        metadata = self.metadata or {}
        cleaned_text = clean_image_text(self.text)
        
        return Source(
            ticker=metadata.get('ticker', 'Unknown'),
            year=metadata.get('year', 'Unknown'),
            score=self.score,
            chunk_id=self.chunk_id,
            chunk_type=metadata.get('type', 'Text'),
            page=int(metadata.get('page', 0)) if metadata.get('page') else 0,
            image_path=metadata.get('image_path') or metadata.get('filename'),
            text=cleaned_text
        )
