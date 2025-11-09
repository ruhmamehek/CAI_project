"""RAG pipeline: retrieval + generation with citations."""

from typing import Dict, Optional, List
from pathlib import Path
import logging

from ..retrieval.dense_retriever import DenseRetriever
from ..retrieval.multi_index_retriever import MultiIndexRetriever
from ..generation.generator import AnswerGenerator
from ..data.company_detector import MetadataDetector

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline: retrieval + generation."""
    
    def __init__(
        self,
        index_dir: str,
        retriever: Optional[DenseRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        retrieval_config: Optional[Dict] = None,
        generation_config: Optional[Dict] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            index_dir: Directory containing FAISS index
            retriever: Pre-initialized retriever (optional)
            generator: Pre-initialized generator (optional)
            retrieval_config: Config dict for retriever
            generation_config: Config dict for generator
        """
        self.index_dir = Path(index_dir)
        
        if retriever:
            self.retriever = retriever
        else:
            retrieval_config = retrieval_config or {}
            manifest_path = self.index_dir / "index_manifest.json"
            
            if manifest_path.exists():
                logger.info("Using multi-index retriever (company/year organized)")
                self.retriever = MultiIndexRetriever(
                    index_dir=str(self.index_dir),
                    model_name=retrieval_config.get("model_name", "BAAI/bge-base-en-v1.5"),
                    device=retrieval_config.get("device", "auto")
                )
            else:
                logger.info("Using single index retriever")
                self.retriever = DenseRetriever(
                    model_name=retrieval_config.get("model_name", "BAAI/bge-base-en-v1.5"),
                    device=retrieval_config.get("device", "auto")
                )
                self.retriever.load_index(str(self.index_dir))
        
        if generator:
            self.generator = generator
        else:
            generation_config = generation_config or {}
            self.generator = AnswerGenerator(
                model_name=generation_config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                device=generation_config.get("device", "auto"),
                max_new_tokens=generation_config.get("max_new_tokens", 256),
                temperature=generation_config.get("temperature", 0.7)
            )
        
        self.metadata_detector = MetadataDetector()
        self.available_tickers = None
        self.available_years = None
    
    def query(
        self, 
        question: str, 
        top_k: int = 5, 
        return_chunks: bool = True,
        filter_metadata: Dict = None,
        auto_detect_company: bool = True
    ) -> Dict:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            return_chunks: Whether to return retrieved chunks in response
            filter_metadata: Dict of metadata filters (e.g., {"ticker": "AAPL", "year": 2023})
            auto_detect_company: Automatically detect company from query if not specified
            
        Returns:
            Dict with answer, citations, and optionally chunks
        """
        logger.info(f"Processing query: {question}")
        
        if filter_metadata is None and auto_detect_company:
            if self.available_tickers is None:
                self.available_tickers, self.available_years = self._get_available_metadata()
            
            detected = self.metadata_detector.detect_from_query(
                question, 
                available_tickers=self.available_tickers,
                available_years=self.available_years
            )
            if detected:
                filter_metadata = detected
                logger.info(f"Auto-detected metadata from query: {detected}")
        
        if filter_metadata:
            logger.info(f"Filtering by metadata: {filter_metadata}")
        
        chunks = self.retriever.retrieve(question, top_k=top_k, filter_metadata=filter_metadata)
        logger.info(f"Retrieved {len(chunks)} chunks")
        
        result = self.generator.generate(question, chunks)
        
        response = {
            "question": question,
            "answer": result["answer"],
            "citations": result["citations"],
            "num_chunks_used": len(result["chunks_used"])
        }
        
        if return_chunks:
            response["retrieved_chunks"] = [
                {
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "doc_id": chunk.get("doc_id", "unknown"),
                    "score": chunk.get("score", 0.0)
                }
                for chunk in chunks
            ]
        
        return response
    
    def _get_available_metadata(self):
        """Get list of available tickers and years."""
        if hasattr(self.retriever, 'get_available_metadata'):
            metadata = self.retriever.get_available_metadata()
            return metadata["tickers"], metadata["years"]
        else:
            tickers = set()
            years = set()
            
            if hasattr(self.retriever, 'chunks') and self.retriever.chunks:
                for chunk in self.retriever.chunks:
                    if "ticker" in chunk and chunk["ticker"]:
                        tickers.add(chunk["ticker"])
                    if "year" in chunk and chunk["year"]:
                        years.add(chunk["year"])
            
            return list(tickers), list(years)
