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
        """Initialize ChromaDB retriever."""
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
        
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model
        )
        
        try:
            collection = client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=embedding_fn,
                metadata={"embedding_model": self.config.embedding_model}
            )
        except Exception as e:
            if "soft deleted" in str(e).lower() or "not found" in str(e).lower():
                logger.warning(f"Collection '{self.config.collection_name}' appears to be soft-deleted. Attempting to recreate...")
                try:
                    client.delete_collection(name=self.config.collection_name)
                    logger.info(f"Deleted soft-deleted collection '{self.config.collection_name}'")
                except Exception as delete_error:
                    logger.warning(f"Could not delete soft-deleted collection: {delete_error}")
                
                collection = client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=embedding_fn,
                    metadata={"embedding_model": self.config.embedding_model}
                )
                logger.info(f"Created new collection '{self.config.collection_name}'")
            else:
                raise
        
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
        """Retrieve relevant chunks from ChromaDB."""
        where_clause = None
        if filters:
            filtered_filters = {k: v for k, v in filters.items() if k != 'filing_type'}
            filter_items = list(filtered_filters.items())
            if len(filter_items) == 0:
                where_clause = None
            elif len(filter_items) == 1:
                key, value = filter_items[0]
                where_clause = {key: value}
            else:
                where_clause = {
                    "$and": [
                        {key: value}
                        for key, value in filter_items
                    ]
                }
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
        except Exception as e:
            error_str = str(e).lower()
            if "soft deleted" in error_str or "not found" in error_str:
                logger.warning(f"Collection appears to be soft-deleted during query. Reinitializing...")
                self.collection = self._initialize_collection()
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where_clause if where_clause else None
                )
            else:
                raise
        
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
                    score=1.0 - distance,
                    chunk_id=chunk_id
                ))
        
        logger.info(f"Retrieved {len(chunks)} chunks for query")
        return chunks
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""
        count = self.collection.count()
        return {
            "collection_name": self.config.collection_name,
            "num_documents": count,
            "embedding_model": self.config.embedding_model
        }

