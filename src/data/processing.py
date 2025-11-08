"""Document processing and chunking."""

import re
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process raw documents (SEC filings, FOMC texts) into text."""
    
    def parse_sec_filing(self, file_path: Path) -> str:
        """
        Parse SEC filing (HTML/XML) to extract text.
        
        Args:
            file_path: Path to SEC filing file
            
        Returns:
            Extracted text content
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove HTML tags and normalize whitespace
        text = re.sub(r'<[^>]+>', '', content)
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def parse_fomc_text(self, file_path: Path) -> str:
        """
        Parse FOMC text (PDF or HTML) to extract text.
        
        Args:
            file_path: Path to FOMC text file
            
        Returns:
            Extracted text content
        """
        # TODO: Implement FOMC text parsing (may need PyPDF2 for PDFs)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


class Chunker:
    """Chunk documents into overlapping segments."""
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: float = 0.3,
        tokenizer_name: str = "gpt2"
    ):
        """
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap ratio (0.0 to 1.0)
            tokenizer_name: Tokenizer to use for counting tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, any]]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Input text
            doc_id: Document identifier
            
        Returns:
            List of chunks with metadata
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        overlap_tokens = int(self.chunk_size * self.chunk_overlap)
        step_size = self.chunk_size - overlap_tokens
        
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "start_token": i,
                "end_token": min(i + self.chunk_size, len(tokens))
            })
            
        return chunks
