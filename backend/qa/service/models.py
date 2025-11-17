"""Data models for RAG service."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Source:
    """Source information for a retrieved chunk."""
    ticker: str
    # filing_type: str
    year: str
    accession_number: str
    score: float
    chunk_id: str
    text: Optional[str] = None  # Optional chunk text content


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
                    # "filing_type": source.filing_type,
                    "year": source.year,
                    "accession_number": source.accession_number,
                    "score": source.score,
                    "chunk_id": source.chunk_id,
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
        return Source(
            ticker=metadata.get('ticker', 'Unknown'),
            # filing_type=metadata.get('filing_type', 'Unknown'),
            year=metadata.get('year', 'Unknown'),
            accession_number=metadata.get('accession_number', 'Unknown'),
            score=self.score,
            chunk_id=self.chunk_id,
            text=self.text
        )

