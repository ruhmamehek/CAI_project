"""ChromaDB retriever for SEC filings."""

import logging
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

from .models import Chunk
from .config import ChromaDBConfig

logger = logging.getLogger(__name__)


class ChromaDBRetriever:
    """Retriever for querying ChromaDB vector database."""
    
    def __init__(self, config: ChromaDBConfig):
        """
        Initialize ChromaDB retriever.
        
        Args:
            config: ChromaDB configuration
        """
        self.config = config
        self.collection = self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize ChromaDB client and collection."""
        logger.info("Connecting to ChromaDB Cloud...")
        client = chromadb.CloudClient(
            api_key=self.config.api_key,
            tenant=self.config.tenant,
            database=self.config.database
        )
        
        # Create embedding function
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        
        # Get collection
        collection = client.get_collection(
            name=self.config.collection_name,
            embedding_function=embedding_fn
        )
        
        doc_count = collection.count()
        logger.info(
            f"Connected to collection '{self.config.collection_name}' "
            f"with {doc_count} documents"
        )
        
        return collection
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Retrieve relevant chunks from ChromaDB.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            filters: Metadata filters (e.g., {"ticker": "AAPL", "year": "2023"})
            
        Returns:
            List of retrieved chunks
        """
        # Build where clause for filtering
        # ChromaDB requires $and operator when multiple filters are present
        where_clause = None
        if filters:
            filter_items = list(filters.items())
            if len(filter_items) == 0:
                where_clause = None
            elif len(filter_items) == 1:
                # Single filter: use directly
                key, value = filter_items[0]
                where_clause = {key: value}
            else:
                # Multiple filters: use $and operator
                where_clause = {
                    "$and": [
                        {key: value}
                        for key, value in filter_items
                    ]
                }
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        
        # Format results into Chunk objects
        chunks = []
        if results.get('documents') and len(results['documents']) > 0:
            documents = results['documents'][0]
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else [{}] * len(documents)
            distances = results.get('distances', [[0.0]])[0] if results.get('distances') else [0.0] * len(documents)
            ids = results.get('ids', [[]])[0] if results.get('ids') else []
            
            for doc, metadata, distance, chunk_id in zip(documents, metadatas, distances, ids):
                chunks.append(Chunk(
                    text=doc,
                    metadata=metadata or {},
                    score=1.0 - distance,  # Convert distance to similarity score
                    chunk_id=chunk_id
                ))
        
        logger.info(f"Retrieved {len(chunks)} chunks for query")
        return chunks
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the ChromaDB collection.
        
        Returns:
            Dictionary with collection information
        """
        count = self.collection.count()
        return {
            "collection_name": self.config.collection_name,
            "num_documents": count,
            "embedding_model": self.config.embedding_model
        }

