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

try:
    _ = faiss.StandardGpuResources
    FAISS_GPU_AVAILABLE = True
except AttributeError:
    FAISS_GPU_AVAILABLE = False


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
        device: str = "auto",
        use_faiss_gpu: bool = True
    ):
        """
        Args:
            model_name: Sentence transformer model name
            device: Device to run model on ("auto", "cuda", "mps", or "cpu")
            use_faiss_gpu: Whether to use GPU for FAISS operations (if available)
        """
        self.model_name = model_name
        self.device = get_device(device)
        self.use_faiss_gpu = use_faiss_gpu and (self.device == "cuda") and FAISS_GPU_AVAILABLE
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
        
        logger.info("Encoding chunks...")
        batch_size = 128 if self.device == "cuda" else 32
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        n_vectors = len(embeddings)
        
        use_ivf = n_vectors > 10000
        if use_ivf:
            n_clusters = min(256, int(n_vectors ** 0.5))
            quantizer = faiss.IndexFlatIP(dimension)
            cpu_index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            cpu_index.nprobe = 10
            logger.info(f"Using approximate index (IVF) with {n_clusters} clusters")
            logger.info("Training IVF index...")
            cpu_index.train(embeddings.astype('float32'))
        else:
            cpu_index = faiss.IndexFlatIP(dimension)
            logger.info("Using exact index (IndexFlatIP)")
        
        if self.use_faiss_gpu:
            try:
                gpu_resource = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
            except Exception:
                self.index = cpu_index
        else:
            self.index = cpu_index
            
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 20,
        filter_metadata: Dict = None
    ) -> List[Dict[str, any]]:
        """
        Retrieve top-k chunks for query with optional metadata filtering.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            filter_metadata: Dict of metadata filters (e.g., {"ticker": "AAPL", "year": 2023})
            
        Returns:
            List of retrieved chunks with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        search_k = top_k * 5 if filter_metadata else top_k
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx].copy()
            
            if filter_metadata:
                matches = all(
                    chunk.get(key) == value 
                    for key, value in filter_metadata.items()
                )
                if not matches:
                    continue
            
            chunk["score"] = float(score)
            results.append(chunk)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save_index(self, index_dir: str):
        """Save index and metadata to disk."""
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        try:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        except (AttributeError, RuntimeError):
            cpu_index = self.index
            
        faiss.write_index(cpu_index, str(index_path / "index.faiss"))
        
        with open(index_path / "chunks.json", "w") as f:
            json.dump(self.chunks, f, indent=2)
        
        logger.info(f"Index saved to {index_path}")
    
    def load_index(self, index_dir: str):
        """Load index and metadata from disk."""
        index_path = Path(index_dir)
        
        cpu_index = faiss.read_index(str(index_path / "index.faiss"))
        
        if self.use_faiss_gpu:
            try:
                gpu_resource = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
            except Exception:
                self.index = cpu_index
        else:
            self.index = cpu_index
        
        with open(index_path / "chunks.json", "r") as f:
            self.chunks = json.load(f)
        
        logger.info(f"Index loaded from {index_path} with {self.index.ntotal} vectors")
