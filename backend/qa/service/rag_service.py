"""RAG service for querying SEC filings."""

import logging
from typing import Optional, Dict, Any, List
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .retriever import ChromaDBRetriever
from .llm_client import LLMClient, create_llm_client
from .prompt_builder import PromptBuilder
from .models import QueryRequest, QueryResponse, Chunk
from .config import RAGConfig

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for querying SEC filings using ChromaDB and LLM."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize RAG service.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.retriever = ChromaDBRetriever(config.chroma)
        self.llm_client = create_llm_client(config.llm)
        self.prompt_builder = PromptBuilder()
        
        # Initialize reranking models (optional, lazy loading)
        self.bi_encoder = None
        self.cross_encoder = None
        self._initialize_reranking()
    
    def _initialize_reranking(self):
        """Initialize reranking models if available and enabled."""
        if not self.config.enable_reranking:
            logger.info("Reranking disabled: not enabled in configuration")
            return
        
        try:
            cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            max_length = self.config.rerank_max_length
            
            logger.info(f"Initializing reranking models (max_length={max_length})...")
            self.cross_encoder = CrossEncoder(
                cross_encoder_model,
                max_length=max_length,
                device="cpu"
            )
            logger.info("Reranking models initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize reranking models: {e}. Reranking disabled.")
            self.cross_encoder = None
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Retrieve relevant chunks from ChromaDB.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve (defaults to config.top_k)
            filters: Metadata filters (e.g., {"ticker": "AAPL", "year": "2023"})
            
        Returns:
            List of retrieved chunks
        """
        top_k = top_k or self.config.top_k
        return self.retriever.retrieve(query, top_k=top_k, filters=filters)
    
    def generate_response(
        self,
        query: str,
        chunks: List[Chunk],
        max_context_length: Optional[int] = None
    ) -> str:
        """
        Generate response using LLM with retrieved chunks as context.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            max_context_length: Maximum context length (defaults to config.max_context_length)
            
        Returns:
            Generated response text
        """
        if not chunks:
            return self.prompt_builder.build_empty_response()
        
        logger.info(f"Generating response for query: {query}")
        max_length = max_context_length or self.config.max_context_length
        
        # Build context from chunks
        logger.info(f"Building context from {len(chunks)} chunks")
        context = self.prompt_builder.build_context(chunks, max_length=max_length)
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(query, context)
        logger.info(f"Prompt: {prompt}")

        # Generate response
        system_prompt = self.prompt_builder.SYSTEM_PROMPT
        response = self.llm_client.generate(prompt, system_prompt=system_prompt)
        
        return response
    
    def rerank_chunks(self, query: str, chunks: List[Chunk], top_k: Optional[int] = None) -> List[Chunk]:
        """
        Rerank chunks based on their relevance to the query using cross-encoder.
        
        Args:
            query: Query text
            chunks: List of chunks to rerank
            top_k: Return only top K chunks after reranking (optional)
            
        Returns:
            Reranked list of chunks
        """
        if not self.cross_encoder:
            logger.warning("Reranking requested but cross-encoder not available. Returning original chunks.")
            return chunks
        
        if not chunks:
            return chunks
        
        try:
            pairs = [[query, chunk.text] for chunk in chunks]
            
            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)
            
            # Sort by score (descending)
            sorted_indices = np.argsort(scores)[::-1]
            logger.info(f"Reranked indices: {sorted_indices}")
            reranked_chunks = [chunks[i] for i in sorted_indices]
            
            # Update scores in chunks (this may or may not be useful)
            # for i, idx in enumerate(sorted_indices):
            #     reranked_chunks[i].score = float(scores[idx])
            
            if top_k:
                reranked_chunks = reranked_chunks[:top_k]
            
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            return chunks
    
    def query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> QueryResponse:
        """
        Complete RAG pipeline: retrieve and generate response.
        
        Args:
            query: User query
            filters: Metadata filters for retrieval
            top_k: Number of chunks to retrieve
            
        Returns:
            QueryResponse with answer, sources, and metadata
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(query, top_k=top_k, filters=filters)
        logger.info(f"Retrieved {len(chunks)} chunks")

        # Rerank chunks
        chunks = self.rerank_chunks(query, chunks, top_k=10)
        logger.info(f"Reranked {len(chunks)} chunks")

        # Generate response
        answer = self.generate_response(query, chunks)
        
        # Extract source information
        sources = [chunk.to_source() for chunk in chunks]
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            num_chunks_retrieved=len(chunks)
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB collection.
        
        Returns:
            Dictionary with collection information
        """
        info = self.retriever.get_collection_info()
        info.update({
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model
        })
        return info

