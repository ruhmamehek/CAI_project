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
    item_number: Optional[str] = None  # SEC Item number (e.g., "1", "1A", "7")


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
    reasoning_steps: Optional[str] = None  # Chain-of-thought reasoning
    filter_reasoning: Optional[str] = None  # Reasoning for filter selection
    applied_filters: Optional[Dict[str, Any]] = None  # Filters that were applied
    
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
                    "text": source.text,
                    "item_number": source.item_number
                }
                for source in self.sources
            ],
            "num_chunks_retrieved": self.num_chunks_retrieved
        }
        if self.verification:
            result["verification"] = self.verification
        if self.reasoning_steps:
            result["reasoning_steps"] = self.reasoning_steps
        if self.filter_reasoning:
            result["filter_reasoning"] = self.filter_reasoning
        if self.applied_filters:
            result["applied_filters"] = self.applied_filters
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
        item_number = metadata.get('item_number')
        # Only include item_number if it's not empty
        item_number = item_number if item_number else None
        cleaned_text = clean_image_text(self.text)
        
        return Source(
            ticker=metadata.get('ticker', 'Unknown'),
            year=metadata.get('year', 'Unknown'),
            score=self.score,
            chunk_id=self.chunk_id,
            chunk_type=metadata.get('type', 'Text'),
            page=int(metadata.get('page', 0)) if metadata.get('page') else 0,
            image_path=metadata.get('image_path') or metadata.get('filename'),
            text=cleaned_text,
            item_number=item_number
        )
