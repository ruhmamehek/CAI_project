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
    batch_size: int = 100,
    replace: bool = True
):
    """
    Upload chunks to ChromaDB.
    
    Args:
        chunks: List of chunk dictionaries with 'text' and metadata
        collection_name: Name of the ChromaDB collection
        embedding_model: Name of the embedding model to use
        batch_size: Number of chunks to upload per batch
        replace: If True, replace all existing data in the collection (default: True)
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
    
    # Delete collection if it exists and replace is True
    if replace:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            logger.debug(f"Collection {collection_name} does not exist or could not be deleted: {e}")
    
    # Create collection (will be created if it doesn't exist)
    logger.info(f"Creating collection: {collection_name}")
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"embedding_model": embedding_model}
    )
    
    # Prepare data for upload
    logger.info("Preparing chunks for upload...")
    texts = []
    ids = []
    metadatas = []
    
    for chunk in chunks:
        if "text" not in chunk:
            logger.warning(f"Chunk missing 'text' field, skipping: {chunk.get('chunk_id', 'unknown')}")
            continue
        
        texts.append(chunk["text"])
        
        # Generate chunk_id if not present
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            ticker_str = chunk.get("ticker", "unknown")
            year_str = chunk.get("year", "unknown")
            chunk_id = f"{ticker_str}_{year_str}_chunk_{len(ids)}"
        ids.append(chunk_id)
        
        # Extract metadata (exclude 'text' and 'chunk_id' as they're handled separately)
        metadata = {
            "ticker": chunk.get("ticker", ""),
            "year": str(chunk.get("year", "")),
            "item_number": chunk.get("item_number") if chunk.get("item_number") is not None else "",
        }
        
        # Add optional fields if present
        if "page_number" in chunk:
            metadata["page_number"] = chunk.get("page_number")
        if "doc_id" in chunk:
            metadata["doc_id"] = chunk.get("doc_id", "")
        if "accession_number" in chunk:
            metadata["accession_number"] = chunk.get("accession_number", "")
        if "start_token" in chunk:
            metadata["start_token"] = chunk.get("start_token", 0)
        if "end_token" in chunk:
            metadata["end_token"] = chunk.get("end_token", 0)
        
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
