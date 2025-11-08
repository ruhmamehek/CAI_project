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
        # Split text into sentences first to avoid tokenizing huge texts at once
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        overlap_tokens = int(self.chunk_size * self.chunk_overlap)
        
        current_chunk_sentences = []
        token_count = 0
        
        for sentence in sentences:
            # Tokenize sentence individually (with truncation to avoid warnings)
            sentence_tokens = self.tokenizer.encode(
                sentence, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=1024
            )
            sentence_token_count = len(sentence_tokens)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if token_count + sentence_token_count > self.chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append({
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                    "start_token": 0,  # Simplified - not tracking exact positions
                    "end_token": token_count
                })
                
                # Start new chunk with overlap (keep last few sentences)
                overlap_sentences = []
                overlap_count = 0
                
                # Keep sentences that fit in overlap
                for sent in reversed(current_chunk_sentences):
                    sent_tokens = self.tokenizer.encode(
                        sent, 
                        add_special_tokens=False, 
                        truncation=True, 
                        max_length=1024
                    )
                    if overlap_count + len(sent_tokens) <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                        overlap_count += len(sent_tokens)
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                token_count = overlap_count
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            token_count += sentence_token_count
        
        # Add final chunk if any remaining
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "start_token": 0,
                "end_token": token_count
            })
            
        return chunks
