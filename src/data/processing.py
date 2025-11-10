"""Document processing and chunking."""

import re
import json
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
        
        return self.cleaner.clean_text(content)
    
    def parse_fomc_text(self, file_path: Path) -> str:
        """
        Parse FOMC text (PDF or HTML) to extract text.
        
        Args:
            file_path: Path to FOMC text file
            
        Returns:
            Extracted text content
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()


class Chunker:
    """Chunk documents into overlapping segments."""
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: float = 0.3,
        tokenizer_name: str = "gpt2",
        max_chunk_bytes: int = 16384  # ChromaDB Cloud free tier limit: 16 KB
    ):
        """
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap ratio (0.0 to 1.0)
            tokenizer_name: Tokenizer to use for counting tokens
            max_chunk_bytes: Maximum byte size for a chunk (text + metadata)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_bytes = max_chunk_bytes
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def _extract_metadata_for_storage(self, chunk_data: Dict) -> Dict:
        """Extract only the metadata fields that will be stored in ChromaDB."""
        # These are the fields that will be stored in ChromaDB metadata
        # (excluding 'text' and 'chunk_id' which are stored separately)
        metadata_fields = [
            "doc_id", "ticker", "filing_type", "accession_number", 
            "year", "start_token", "end_token"
        ]
        metadata = {}
        for field in metadata_fields:
            if field in chunk_data:
                value = chunk_data[field]
                # Convert year to string (as ChromaDB does)
                if field == "year" and isinstance(value, int):
                    value = str(value)
                metadata[field] = value
        return metadata
    
    def _calculate_chunk_bytes(self, text: str, chunk_data: Dict) -> int:
        """
        Calculate the byte size of a chunk (text + metadata as stored in ChromaDB).
        
        Args:
            text: The text content of the chunk
            chunk_data: The full chunk data dictionary
            
        Returns:
            Total byte size including text and metadata
        """
        text_bytes = len(text.encode('utf-8'))
        # Extract only metadata fields that will be stored in ChromaDB
        metadata = self._extract_metadata_for_storage(chunk_data)
        # Estimate metadata size (serialized as JSON, matching ChromaDB storage)
        metadata_bytes = len(json.dumps(metadata, default=str).encode('utf-8'))
        return text_bytes + metadata_bytes
    
    def _split_oversized_chunk(
        self, 
        chunk_text: str, 
        doc_id: str, 
        base_chunk_data: Dict,
        chunk_id_prefix: str
    ) -> List[Dict]:
        """
        Split a chunk that exceeds max_chunk_bytes into smaller chunks.
        
        Args:
            chunk_text: Text content of the oversized chunk
            doc_id: Document identifier
            base_chunk_data: Base chunk data dictionary (contains metadata fields)
            chunk_id_prefix: Prefix for chunk IDs (e.g., "{doc_id}_chunk_{n}")
            
        Returns:
            List of smaller chunks that fit within the byte limit
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
        if not sentences:
            sentences = [chunk_text]
        
        split_chunks = []
        current_chunk_sentences = []
        current_chunk_id = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Try adding sentence to current chunk
            test_sentences = current_chunk_sentences + [sentence]
            test_text = ' '.join(test_sentences)
            test_chunk_data = base_chunk_data.copy()
            test_chunk_data["text"] = test_text  # Temporary, for size calculation
            test_bytes = self._calculate_chunk_bytes(test_text, test_chunk_data)
            
            if test_bytes <= self.max_chunk_bytes:
                # Fits, add to current chunk
                current_chunk_sentences.append(sentence)
            else:
                # Doesn't fit, save current chunk and start new one
                if current_chunk_sentences:
                    chunk_text_final = ' '.join(current_chunk_sentences)
                    chunk_data = base_chunk_data.copy()
                    chunk_data["text"] = chunk_text_final
                    chunk_data["chunk_id"] = f"{chunk_id_prefix}_split_{current_chunk_id}"
                    chunk_data["start_token"] = 0  # Will be recalculated if needed
                    chunk_data["end_token"] = 0
                    split_chunks.append(chunk_data)
                    current_chunk_id += 1
                
                # Check if single sentence is too large
                test_chunk_data = base_chunk_data.copy()
                test_chunk_data["text"] = sentence  # Temporary, for size calculation
                sentence_bytes = self._calculate_chunk_bytes(sentence, test_chunk_data)
                if sentence_bytes > self.max_chunk_bytes:
                    # Sentence itself is too large, split by words
                    words = sentence.split()
                    word_chunks = []
                    current_words = []
                    
                    for word in words:
                        test_words = current_words + [word]
                        test_text = ' '.join(test_words)
                        test_chunk_data = base_chunk_data.copy()
                        test_chunk_data["text"] = test_text  # Temporary, for size calculation
                        test_bytes = self._calculate_chunk_bytes(test_text, test_chunk_data)
                        
                        if test_bytes <= self.max_chunk_bytes:
                            current_words.append(word)
                        else:
                            if current_words:
                                word_chunks.append(' '.join(current_words))
                            current_words = [word]
                    
                    if current_words:
                        word_chunks.append(' '.join(current_words))
                    
                    # Add word-level chunks
                    for word_chunk in word_chunks:
                        chunk_data = base_chunk_data.copy()
                        chunk_data["text"] = word_chunk
                        chunk_data["chunk_id"] = f"{chunk_id_prefix}_split_{current_chunk_id}"
                        chunk_data["start_token"] = 0
                        chunk_data["end_token"] = 0
                        split_chunks.append(chunk_data)
                        current_chunk_id += 1
                else:
                    # Single sentence fits, start new chunk with it
                    current_chunk_sentences = [sentence]
        
        # Add remaining chunk
        if current_chunk_sentences:
            chunk_text_final = ' '.join(current_chunk_sentences)
            chunk_data = base_chunk_data.copy()
            chunk_data["text"] = chunk_text_final
            chunk_data["chunk_id"] = f"{chunk_id_prefix}_split_{current_chunk_id}"
            chunk_data["start_token"] = 0
            chunk_data["end_token"] = 0
            split_chunks.append(chunk_data)
        
        return split_chunks
        
    def chunk_text(
        self, 
        text: str, 
        doc_id: str, 
        metadata: Dict = None,
        filter_chunks: bool = True
    ) -> List[Dict[str, any]]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Input text
            doc_id: Document identifier
            metadata: Additional metadata (ticker, filing_type, year, etc.)
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
                chunk_data = {
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                    "start_token": 0,
                    "end_token": token_count
                }
                if metadata:
                    chunk_data.update(metadata)
                
                # Check if chunk exceeds byte size limit
                chunk_bytes = self._calculate_chunk_bytes(chunk_text, chunk_data)
                if chunk_bytes > self.max_chunk_bytes:
                    # Split oversized chunk
                    logger.debug(
                        f"Chunk {chunk_data['chunk_id']} exceeds byte limit "
                        f"({chunk_bytes} bytes > {self.max_chunk_bytes} bytes). Splitting..."
                    )
                    split_chunks = self._split_oversized_chunk(
                        chunk_text, 
                        doc_id, 
                        chunk_data,
                        f"{doc_id}_chunk_{len(chunks)}"
                    )
                    chunks.extend(split_chunks)
                else:
                    chunks.append(chunk_data)
                
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
            chunk_data = {
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "start_token": 0,
                "end_token": token_count
            }
            if metadata:
                chunk_data.update(metadata)
            
            # Check if final chunk exceeds byte size limit
            chunk_bytes = self._calculate_chunk_bytes(chunk_text, chunk_data)
            if chunk_bytes > self.max_chunk_bytes:
                # Split oversized chunk
                logger.debug(
                    f"Final chunk {chunk_data['chunk_id']} exceeds byte limit "
                    f"({chunk_bytes} bytes > {self.max_chunk_bytes} bytes). Splitting..."
                )
                split_chunks = self._split_oversized_chunk(
                    chunk_text, 
                    doc_id, 
                    chunk_data,
                    f"{doc_id}_chunk_{len(chunks)}"
                )
                chunks.extend(split_chunks)
            else:
                chunks.append(chunk_data)
        
        if filter_chunks and chunks:
            from .cleaning import SECFilingCleaner
            cleaner = SECFilingCleaner(min_chunk_length=50)
            chunks = cleaner.filter_chunks(chunks)
            
        return chunks
