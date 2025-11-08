"""Dense retrieval using sentence transformers and FAISS."""

import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch
import json
import logging

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> str:
    """Get the best available device."""
    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_preference


class DenseRetriever:
    """Dense retriever using embeddings and FAISS index."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "auto"
    ):
        """
        Args:
            model_name: Sentence transformer model name
            device: Device to run model on ("auto", "cuda", "mps", or "cpu")
        """
        self.model_name = model_name
        self.device = get_device(device)
        logger.info(f"Using device: {self.device}")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.chunks = []
        
    def build_index(self, chunks: List[Dict[str, any]]):
        """
        Build FAISS index from chunks.
        
        Args:
            chunks: List of chunk dicts with 'text' and metadata
        """
        logger.info(f"Building index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Encode chunks
        logger.info("Encoding chunks...")
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, any]]:
        """
        Retrieve top-k chunks for query.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)
        
        return results
    
    def save_index(self, index_dir: str):
        """Save index and metadata to disk."""
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path / "index.faiss"))
        
        # Save chunks metadata
        with open(index_path / "chunks.json", "w") as f:
            json.dump(self.chunks, f, indent=2)
        
        logger.info(f"Index saved to {index_path}")
    
    def load_index(self, index_dir: str):
        """Load index and metadata from disk."""
        index_path = Path(index_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path / "index.faiss"))
        
        # Load chunks metadata
        with open(index_path / "chunks.json", "r") as f:
            self.chunks = json.load(f)
        
        logger.info(f"Index loaded from {index_path} with {self.index.ntotal} vectors")

