"""Upload processed chunks to ChromaDB."""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('CHROMA_API_KEY')
tenant = os.getenv('CHROMA_TENANT')
database = os.getenv('CHROMA_DATABASE')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chunks(chunks_file: str) -> List[Dict]:
    """Load chunks from JSON file."""
    logger.info(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks

def upload_to_chromadb(
    chunks: List[Dict],
    collection_name: str = "sec_filings",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 100
):
    """
    Upload chunks to ChromaDB.
    
    Args:
        chunks: List of chunk dictionaries with 'text' and metadata
        collection_name: Name of the ChromaDB collection
        embedding_model: Name of the embedding model to use
        batch_size: Number of chunks to upload per batch
    """
    
    logger.info("Connecting to ChromaDB Cloud...")
    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database
    )
    
    # Create embedding function using the same model as the retriever
    logger.info(f"Using embedding model: {embedding_model}")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    
    # Get or create collection
    logger.info(f"Getting/creating collection: {collection_name}")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"embedding_model": embedding_model}
    )
    
    # Check if collection already has data
    existing_count = collection.count()
    if existing_count > 0:
        logger.warning(
            f"Collection already contains {existing_count} documents. "
            "New chunks will be added. Consider deleting the collection first if you want to start fresh."
        )
    
    # Prepare data for upload
    logger.info("Preparing chunks for upload...")
    texts = []
    ids = []
    metadatas = []
    
    for chunk in chunks:
        texts.append(chunk["text"])
        ids.append(chunk["chunk_id"])
        
        # Extract metadata (exclude 'text' and 'chunk_id' as they're handled separately)
        metadata = {
            "doc_id": chunk.get("doc_id", ""),
            "ticker": chunk.get("ticker", ""),
            "filing_type": chunk.get("filing_type", ""),
            "accession_number": chunk.get("accession_number", ""),
            "year": chunk.get("year", ""),
            "start_token": chunk.get("start_token", 0),
            "end_token": chunk.get("end_token", 0),
        }
        # Convert year to string for ChromaDB (metadata values must be strings, numbers, or bools)
        if isinstance(metadata["year"], int):
            metadata["year"] = str(metadata["year"])
        metadatas.append(metadata)
    
    # Upload in batches
    logger.info(f"Uploading {len(chunks)} chunks in batches of {batch_size}...")
    total_uploaded = 0
    
    for i in range(0, len(chunks), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        
        try:
            collection.add(
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
            total_uploaded += len(batch_texts)
            logger.info(
                f"Uploaded batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} "
                f"({total_uploaded}/{len(chunks)} chunks)"
            )
        except Exception as e:
            logger.error(f"Error uploading batch {i // batch_size + 1}: {e}")
            raise
    
    logger.info(f"Successfully uploaded {total_uploaded} chunks to ChromaDB")
    logger.info(f"Collection '{collection_name}' now contains {collection.count()} documents")

def main():
    """Main function to upload chunks to ChromaDB."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload SEC filing chunks to ChromaDB")
    parser.add_argument(
        "--chunks-file",
        type=str,
        default="data/processed/sec_chunks.json",
        help="Path to chunks JSON file"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="sec_filings",
        help="Name of ChromaDB collection"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for uploads"
    )
    
    args = parser.parse_args()
    
    # Load chunks
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    chunks = load_chunks(chunks_file)
    
    if not chunks:
        logger.warning("No chunks to upload")
        return
    
    # Upload to ChromaDB
    upload_to_chromadb(
        chunks=chunks,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
