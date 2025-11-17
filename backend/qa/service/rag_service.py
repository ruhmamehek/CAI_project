"""RAG service for querying SEC filings."""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .retriever import ChromaDBRetriever
from .llm_client import LLMClient, create_llm_client
from .prompt_builder import PromptBuilder
from .models import QueryRequest, QueryResponse, Chunk
from .config import RAGConfig
from .verification import RAGVerifier
from .utils import get_project_root, is_path_safe, get_image_mime_type

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for querying SEC filings using ChromaDB and LLM."""
    
    def __init__(self, config: RAGConfig):
        """Initialize RAG service."""
        self.config = config
        self.retriever = ChromaDBRetriever(config.chroma)
        self.llm_client = create_llm_client(config.llm)
        self.prompt_builder = PromptBuilder()
        self.verifier = RAGVerifier(self.llm_client)
        
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
        """Retrieve relevant chunks from ChromaDB."""
        top_k = top_k or self.config.top_k
        return self.retriever.retrieve(query, top_k=top_k, filters=filters)
    
    def generate_response(
        self,
        query: str,
        chunks: List[Chunk],
        max_context_length: Optional[int] = None
    ) -> str:
        """Generate response using LLM with retrieved chunks as context."""
        if not chunks:
            return self.prompt_builder.build_empty_response()
        
        logger.info(f"Generating response for query: {query}")
        max_length = max_context_length or self.config.max_context_length
        
        logger.info(f"Building context from {len(chunks)} chunks")
        context = self.prompt_builder.build_context(chunks, max_length=max_length)
        prompt = self.prompt_builder.build_prompt(query, context)

        images = []
        image_mime_types = []
        project_root = get_project_root()
        
        for chunk in chunks:
            metadata = chunk.metadata or {}
            image_path = metadata.get('image_path') or metadata.get('filename')
            if image_path:
                try:
                    full_path = project_root / image_path
                    
                    if not is_path_safe(full_path, project_root):
                        logger.warning(f"Image path outside project root: {image_path}")
                        continue
                    
                    if full_path.exists():
                        with open(full_path, 'rb') as f:
                            image_data = f.read()
                        images.append(image_data)
                        
                        mime_type = get_image_mime_type(full_path.suffix) or 'image/png'
                        image_mime_types.append(mime_type)
                        logger.info(f"Loaded image: {image_path} ({len(image_data)} bytes)")
                    else:
                        logger.warning(f"Image not found: {full_path}")
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {e}", exc_info=True)
                    continue
        
        system_prompt = self.prompt_builder.SYSTEM_PROMPT
        if images:
            logger.info(f"Sending {len(images)} image(s) to LLM along with text")
            response = self.llm_client.generate(
                prompt, 
                system_prompt=system_prompt,
                images=images,
                image_mime_types=image_mime_types
            )
        else:
            response = self.llm_client.generate(prompt, system_prompt=system_prompt)
        
        return response
    
    def rerank_chunks(self, query: str, chunks: List[Chunk], top_k: Optional[int] = None) -> List[Chunk]:
        """Rerank chunks based on their relevance to the query using cross-encoder."""
        if not self.cross_encoder:
            logger.warning("Reranking requested but cross-encoder not available. Returning original chunks.")
            return chunks
        
        if not chunks:
            return chunks
        
        try:
            pairs = [[query, chunk.text] for chunk in chunks]
            scores = self.cross_encoder.predict(pairs)
            sorted_indices = np.argsort(scores)[::-1]
            logger.info(f"Reranked indices: {sorted_indices}")
            reranked_chunks = [chunks[i] for i in sorted_indices]
            
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
        top_k: Optional[int] = None,
        enable_verification: bool = True
    ) -> QueryResponse:
        """Complete RAG pipeline: retrieve and generate response."""
        chunks = self.retrieve(query, top_k=top_k, filters=filters)
        logger.info(f"Retrieved {len(chunks)} chunks")

        chunks = self.rerank_chunks(query, chunks, top_k=18)
        logger.info(f"Reranked {len(chunks)} chunks")

        answer = self.generate_response(query, chunks)
        sources = [chunk.to_source() for chunk in chunks]
        
        verification_result = None
        if enable_verification and self.config.enable_verification:
            try:
                verification = self.verifier.verify(answer, chunks, query)
                verification_result = verification.to_dict()
                logger.info(f"Verification completed. Overall score: {verification.overall_score:.2f}")
            except Exception as e:
                logger.warning(f"Verification failed: {e}", exc_info=True)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            num_chunks_retrieved=len(chunks),
            verification=verification_result
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""
        info = self.retriever.get_collection_info()
        info.update({
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model
        })
        return info

