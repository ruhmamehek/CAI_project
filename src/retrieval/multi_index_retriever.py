"""Multi-index retriever for company/year organized indices."""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

from .dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


class MultiIndexRetriever:
    """Retriever that manages multiple FAISS indices organized by company/year."""
    
    def __init__(
        self,
        index_dir: str,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "auto",
        use_faiss_gpu: bool = True
    ):
        """
        Args:
            index_dir: Directory containing organized indices
            model_name: Sentence transformer model name
            device: Device to run model on
            use_faiss_gpu: Whether to use GPU for FAISS operations
        """
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.device = device
        self.use_faiss_gpu = use_faiss_gpu
        
        self.retrievers = {}
        self.index_manifest = {}
        self._load_manifest()
        self._load_indices()
    
    def _load_manifest(self):
        """Load index manifest."""
        manifest_path = self.index_dir / "index_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.index_manifest = json.load(f)
            logger.info(f"Loaded manifest with {len(self.index_manifest)} indices")
        else:
            logger.warning(f"Manifest not found at {manifest_path}")
    
    def _load_indices(self):
        """Load all available indices."""
        if not self.index_manifest:
            logger.warning("No manifest found, attempting to discover indices...")
            self._discover_indices()
        
        for key, info in self.index_manifest.items():
            index_path = Path(info["path"])
            if index_path.exists():
                retriever = DenseRetriever(
                    model_name=self.model_name,
                    device=self.device,
                    use_faiss_gpu=self.use_faiss_gpu
                )
                retriever.load_index(str(index_path))
                self.retrievers[key] = retriever
                logger.info(f"Loaded index: {key} ({info['num_chunks']} chunks)")
            else:
                logger.warning(f"Index path not found: {index_path}")
    
    def _discover_indices(self):
        """Discover indices by scanning directory structure."""
        for ticker_dir in self.index_dir.iterdir():
            if not ticker_dir.is_dir() or ticker_dir.name == "__pycache__":
                continue
            
            for year_dir in ticker_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                index_path = year_dir / "index.faiss"
                if index_path.exists():
                    key = f"{ticker_dir.name}/{year_dir.name}"
                    self.index_manifest[key] = {
                        "ticker": ticker_dir.name,
                        "year": int(year_dir.name) if year_dir.name.isdigit() else year_dir.name,
                        "path": str(year_dir)
                    }
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_metadata: Dict = None
    ) -> List[Dict]:
        """
        Retrieve chunks from relevant indices based on metadata filters.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            filter_metadata: Dict of metadata filters (e.g., {"ticker": "AAPL", "year": 2023})
            
        Returns:
            List of retrieved chunks with scores
        """
        if filter_metadata:
            ticker = filter_metadata.get("ticker")
            year = filter_metadata.get("year")
            
            if ticker and year:
                key = f"{ticker}/{year}"
                if key in self.retrievers:
                    return self.retrievers[key].retrieve(query, top_k=top_k, filter_metadata=None)
            elif ticker:
                results = []
                for key, retriever in self.retrievers.items():
                    if key.startswith(f"{ticker}/"):
                        results.extend(retriever.retrieve(query, top_k=top_k, filter_metadata=None))
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                return results[:top_k]
            elif year:
                results = []
                for key, retriever in self.retrievers.items():
                    if key.endswith(f"/{year}"):
                        results.extend(retriever.retrieve(query, top_k=top_k, filter_metadata=None))
                results.sort(key=lambda x: x.get("score", 0), reverse=True)
                return results[:top_k]
        
        results = []
        for retriever in self.retrievers.values():
            results.extend(retriever.retrieve(query, top_k=top_k, filter_metadata=None))
        
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]
    
    def get_available_metadata(self) -> Dict:
        """Get available tickers and years."""
        tickers = set()
        years = set()
        
        for key in self.index_manifest.keys():
            parts = key.split("/")
            if len(parts) == 2:
                ticker, year = parts
                tickers.add(ticker)
                try:
                    years.add(int(year))
                except ValueError:
                    pass
        
        return {
            "tickers": sorted(list(tickers)),
            "years": sorted(list(years))
        }

