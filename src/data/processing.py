"""Document processing and chunking."""

import re
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
import logging

from .cleaning import SECFilingCleaner

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process raw documents (SEC filings, FOMC texts) into text."""
    
    def __init__(
        self,
        min_chunk_length: int = 50,
        remove_xbrl: bool = True,
        remove_metadata: bool = True,
        remove_boilerplate: bool = True
    ):
        """
        Args:
            min_chunk_length: Minimum character length for chunks to keep
            remove_xbrl: Whether to remove XBRL/XML structured data
            remove_metadata: Whether to remove filing metadata sections
            remove_boilerplate: Whether to remove common boilerplate text
        """
        self.cleaner = SECFilingCleaner(
            min_chunk_length=min_chunk_length,
            remove_xbrl=remove_xbrl,
            remove_metadata=remove_metadata,
            remove_boilerplate=remove_boilerplate
        )
    
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
        
        text = self.cleaner.clean_text(content)
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
        
    def chunk_text(self, text: str, doc_id: str, filter_chunks: bool = True) -> List[Dict[str, any]]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Input text
            doc_id: Document identifier
            filter_chunks: Whether to filter out low-quality chunks
            
        Returns:
            List of chunks with metadata
        """
        if not text or len(text.strip()) < 50:
            logger.warning(f"Skipping {doc_id}: text too short after cleaning")
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        overlap_tokens = int(self.chunk_size * self.chunk_overlap)
        
        current_chunk_sentences = []
        token_count = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_tokens = self.tokenizer.encode(
                sentence, 
                add_special_tokens=False, 
                truncation=True, 
                max_length=1024
            )
            sentence_token_count = len(sentence_tokens)
            
            if token_count + sentence_token_count > self.chunk_size and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append({
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                    "start_token": 0,
                    "end_token": token_count
                })
                
                overlap_sentences = []
                overlap_count = 0
                
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
            
            current_chunk_sentences.append(sentence)
            token_count += sentence_token_count
        
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "start_token": 0,
                "end_token": token_count
            })
        
        if filter_chunks and chunks:
            from .cleaning import SECFilingCleaner
            cleaner = SECFilingCleaner(min_chunk_length=50)
            chunks = cleaner.filter_chunks(chunks)
            
        return chunks
