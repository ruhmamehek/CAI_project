"""Prompt building for RAG queries."""

from typing import List
from .models import Chunk
import logging

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builder for RAG prompts."""
    
    SYSTEM_PROMPT = "You are a helpful financial analyst assistant."
    
    @staticmethod
    def build_context(chunks: List[Chunk], max_length: int = 4000) -> str:
        """
        Build context string from chunks.
        
        Args:
            chunks: List of retrieved chunks
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        separator = "\n\n"
        separator_length = len(separator)
        current_length = 0
        
        for chunk in chunks:
            chunk_text = chunk.text
            metadata = chunk.metadata or {}
            ticker = metadata.get('ticker', 'Unknown')
            year = metadata.get('year', 'Unknown')
            filing_type = metadata.get('filing_type', 'Unknown')
            
            # Format chunk with metadata
            header = f"[Source: {ticker} {filing_type} {year}]\n"
            chunk_with_meta = header + chunk_text
            chunk_length = len(chunk_with_meta)
            
            # Calculate total length if we add this chunk
            if context_parts:
                # Need separator between chunks
                total_length = current_length + separator_length + chunk_length
            else:
                # First chunk, no separator needed
                total_length = chunk_length
            
            # Check if adding this chunk would exceed max length
            if total_length > max_length:
                # If this is the first chunk and it's too long, truncate it
                if not context_parts:
                    # Truncate the chunk text to fit within max_length
                    available_length = max_length - len(header) - 3  # -3 for "..."
                    if available_length > 0:
                        truncated_text = chunk_text[:available_length] + "..."
                        chunk_with_meta = header + truncated_text
                        context_parts.append(chunk_with_meta)
                        current_length = len(chunk_with_meta)
                # Otherwise, stop adding chunks
                break
            
            context_parts.append(chunk_with_meta)
            # Update current length (including separators)
            if len(context_parts) == 1:
                current_length = chunk_length
            else:
                current_length = total_length
        
        context = separator.join(context_parts)
        logger.info(f"Built context: {context}")
        return context
    
    @staticmethod
    def build_prompt(query: str, context: str) -> str:
        """
        Build prompt for LLM with query and context.
        
        Args:
            query: User query
            context: Context from retrieved chunks
            
        Returns:
            Formatted prompt
        """
        return f"""You are a financial analyst assistant. Answer the user's question based on the provided context from SEC filings.

Context from SEC filings:
{context}

Question: {query}

Instructions:
- Answer the question based solely on the provided context
- If the context doesn't contain enough information to answer the question, say so
- Cite specific sources (ticker, filing type, year) when referencing information
- Be concise and accurate
- Use professional financial terminology

Answer:"""
    
    @staticmethod
    def build_empty_response() -> str:
        """Build response when no chunks are retrieved."""
        return "I couldn't find any relevant information in the SEC filings to answer your question."

