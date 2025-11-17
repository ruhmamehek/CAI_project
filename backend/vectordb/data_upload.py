"""Upload processed chunks to ChromaDB."""

import os
import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

env_paths = [
    Path(__file__).parent.parent.parent / '.env',
    Path(__file__).parent.parent / '.env',
    Path(__file__).parent / '.env',
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

api_key = os.getenv('CHROMA_API_KEY')
tenant = os.getenv('CHROMA_TENANT')
database = os.getenv('CHROMA_DATABASE')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_credentials():
    """Validate ChromaDB credentials are set."""
    missing = []
    if not api_key:
        missing.append('CHROMA_API_KEY')
    if not tenant:
        missing.append('CHROMA_TENANT')
    if not database:
        missing.append('CHROMA_DATABASE')
    
    if missing:
        raise ValueError(
            f"Missing ChromaDB credentials: {', '.join(missing)}\n"
            f"Please set these environment variables or create a .env file.\n"
            f"Checked locations: {[str(p) for p in env_paths]}"
        )
    
    return True

def extract_ticker_year_from_filename(chunks_file: str) -> Tuple[str, str]:
    """
    Extract ticker and year from filename.
    
    Expected format: {TICKER}_{YEAR}_unified_chunks.json
    Examples: TSLA_2023_unified_chunks.json, AAPL_2024_unified_chunks.json
    
    Args:
        chunks_file: Path to unified chunks JSON file
        
    Returns:
        Tuple of (ticker, year) as strings
        
    Raises:
        ValueError: If filename doesn't match expected format
    """
    import re
    from pathlib import Path
    
    filename = Path(chunks_file).stem
    match = re.match(r'^([A-Z]+)_(\d{4})_unified_chunks$', filename)
    if not match:
        raise ValueError(
            f"Filename '{chunks_file}' doesn't match expected format: "
            "{{TICKER}}_{{YEAR}}_unified_chunks.json (e.g., TSLA_2023_unified_chunks.json)"
        )
    ticker, year = match.groups()
    return ticker, year


def add_ticker_year_to_chunks(chunks: List[Dict], ticker: str, year: str) -> List[Dict]:
    """
    Add ticker and year to each chunk's metadata, overwriting existing values.
    
    Args:
        chunks: List of chunk dictionaries
        ticker: Ticker symbol to add
        year: Year to add
        
    Returns:
        List of chunks with ticker and year set in metadata
    """
    for chunk in chunks:
        chunk["ticker"] = ticker
        chunk["year"] = year
    return chunks


def load_chunks(chunks_file: str, auto_add_ticker_year: bool = True) -> List[Dict]:
    """
    Load chunks from JSON file and optionally add ticker/year from filename.
    
    Args:
        chunks_file: Path to unified chunks JSON file
        auto_add_ticker_year: If True, extract ticker/year from filename and add to chunks
        
    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    if auto_add_ticker_year:
        try:
            ticker, year = extract_ticker_year_from_filename(chunks_file)
            chunks = add_ticker_year_to_chunks(chunks, ticker, year)
            logger.info(f"Added ticker={ticker}, year={year} to chunks from filename")
        except ValueError as e:
            logger.warning(f"Could not extract ticker/year from filename: {e}")
    
    return chunks

def generate_chunk_id(chunk: Dict, index: int) -> str:
    """
    Generate a unique chunk ID if one doesn't exist.
    
    Args:
        chunk: Chunk dictionary
        index: Index of the chunk in the list
        
    Returns:
        Unique chunk ID string
    """
    existing_id = chunk.get("chunk_id", "").strip()
    if existing_id:
        return existing_id
    
    ticker = chunk.get("ticker", "unknown")
    year = chunk.get("year", "unknown")
    chunk_type = chunk.get("type", "text").lower().replace(" ", "_")
    page = chunk.get("page", 0)
    
    if ticker != "unknown" and year != "unknown":
        return f"{chunk_type}_{ticker}_{year}_{page:03d}_{index:04d}"
    elif ticker != "unknown":
        return f"{chunk_type}_{ticker}_{page:03d}_{index:04d}"
    else:
        return f"chunk_{uuid.uuid4().hex[:8]}_{index:04d}"

def delete_collection(collection_name: str = "sec_filings"):
    """
    Delete a ChromaDB collection.
    
    Args:
        collection_name: Name of the collection to delete
    """
    validate_credentials()
    
    logger.info("Connecting to ChromaDB Cloud...")
    try:
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise
    
    try:
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            logger.warning(f"Collection '{collection_name}' does not exist. Nothing to delete.")
            return
        
        logger.info(f"Deleting collection '{collection_name}'...")
        client.delete_collection(name=collection_name)
        logger.info(f"Successfully deleted collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise


def delete_chunks_by_ticker(collection_name: str = "sec_filings", ticker: str = "TSLA"):
    """
    Delete chunks from ChromaDB collection by ticker.
    
    Args:
        collection_name: Name of the ChromaDB collection
        ticker: Ticker symbol to filter chunks for deletion
    """
    validate_credentials()
    
    logger.info("Connecting to ChromaDB Cloud...")
    try:
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise
    
    try:
        collection = client.get_collection(name=collection_name)
        
        before_count = collection.count()
        logger.info(f"Collection '{collection_name}' contains {before_count} documents before deletion")
        
        results = collection.get(where={"ticker": ticker})
        
        if not results['ids']:
            logger.info(f"No chunks found with ticker '{ticker}'. Nothing to delete.")
            return
        
        num_chunks = len(results['ids'])
        logger.info(f"Found {num_chunks} chunks with ticker '{ticker}'")
        
        collection.delete(ids=results['ids'])
        
        after_count = collection.count()
        deleted_count = before_count - after_count
        
        logger.info(f"Successfully deleted {deleted_count} chunks with ticker '{ticker}'")
        logger.info(f"Collection '{collection_name}' now contains {after_count} documents")
        
    except Exception as e:
        logger.error(f"Error deleting chunks by ticker: {e}")
        raise

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
    
    validate_credentials()
    
    logger.info("Connecting to ChromaDB Cloud...")
    try:
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
    except Exception as e:
        error_msg = str(e)
        if "Permission denied" in error_msg or "401" in error_msg or "Unauthorized" in error_msg:
            logger.error(
                "ChromaDB authentication failed. Please check your credentials:\n"
                f"  - CHROMA_API_KEY: {'Set' if api_key else 'MISSING'} "
                f"({'***' + api_key[-4:] if api_key and len(api_key) > 4 else 'Invalid'})\n"
                f"  - CHROMA_TENANT: {'Set' if tenant else 'MISSING'} "
                f"({tenant if tenant else 'Invalid'})\n"
                f"  - CHROMA_DATABASE: {'Set' if database else 'MISSING'} "
                f"({database if database else 'Invalid'})\n"
                "\nMake sure your credentials are correct and have proper permissions."
            )
        raise
    
    logger.info(f"Using embedding model: {embedding_model}")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    
    logger.info(f"Getting/creating collection: {collection_name}")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"embedding_model": embedding_model}
    )
    
    existing_count = collection.count()
    if existing_count > 0:
        logger.warning(
            f"Collection already contains {existing_count} documents. "
            "New chunks will be added. Consider deleting the collection first if you want to start fresh."
        )
    
    logger.info("Preparing chunks for upload...")
    texts = []
    ids = []
    metadatas = []
    
    for index, chunk in enumerate(chunks):
        chunk_id = generate_chunk_id(chunk, index)
        
        text_for_embedding = chunk["text"]
        chunk_type = chunk.get("type", "Text")
        
        if chunk_type in ["Table", "Figure"] and "image_path" in chunk:
            image_path = chunk["image_path"]
            text_for_embedding = f"{text_for_embedding}\n\nImage: {image_path}"
        
        texts.append(text_for_embedding)
        ids.append(chunk_id)
        
        metadata = {
            "chunk_id": chunk_id,
            "ticker": chunk.get("ticker", ""),
            "year": chunk.get("year", ""),
            "type": chunk_type,
            "page": chunk.get("page", 0),
        }
        
        if chunk_type in ["Table", "Figure"]:
            if "image_path" in chunk:
                metadata["image_path"] = chunk["image_path"]
            if "filename" in chunk:
                metadata["filename"] = chunk["filename"]
            if "size" in chunk and isinstance(chunk["size"], dict):
                if "width" in chunk["size"]:
                    metadata["image_width"] = chunk["size"]["width"]
                if "height" in chunk["size"]:
                    metadata["image_height"] = chunk["size"]["height"]
        
        if "coordinates" in chunk and isinstance(chunk["coordinates"], dict):
            coords = chunk["coordinates"]
            if "x_1" in coords:
                metadata["x_1"] = coords["x_1"]
            if "y_1" in coords:
                metadata["y_1"] = coords["y_1"]
            if "x_2" in coords:
                metadata["x_2"] = coords["x_2"]
            if "y_2" in coords:
                metadata["y_2"] = coords["y_2"]
        
        if isinstance(metadata.get("year"), int):
            metadata["year"] = str(metadata["year"])
        if isinstance(metadata.get("page"), (int, float)):
            metadata["page"] = str(int(metadata["page"]))
        for coord_key in ["x_1", "y_1", "x_2", "y_2"]:
            if coord_key in metadata and isinstance(metadata[coord_key], (int, float)):
                metadata[coord_key] = str(metadata[coord_key])
        for size_key in ["image_width", "image_height"]:
            if size_key in metadata and isinstance(metadata[size_key], (int, float)):
                metadata[size_key] = str(metadata[size_key])
        
        metadatas.append(metadata)
    
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
        required=False,
        help="Path to unified chunks JSON file. Expected format: {TICKER}_{YEAR}_unified_chunks.json (e.g., data/processed/TSLA_2023_unified_chunks.json)"
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
    parser.add_argument(
        "--clear-collection",
        action="store_true",
        help="Delete the collection and all its documents before uploading new chunks"
    )
    parser.add_argument(
        "--delete-by-ticker",
        type=str,
        default=None,
        help="Delete chunks by ticker symbol (e.g., TSLA). When used, --chunks-file is not required."
    )
    
    args = parser.parse_args()
    
    if args.delete_by_ticker:
        logger.info(f"Deleting chunks with ticker '{args.delete_by_ticker}'...")
        delete_chunks_by_ticker(
            collection_name=args.collection_name,
            ticker=args.delete_by_ticker
        )
        return
    
    if not args.chunks_file:
        parser.error("--chunks-file is required unless using --delete-by-ticker")
    
    chunks_file = Path(args.chunks_file)
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    chunks = load_chunks(str(chunks_file))
    
    if not chunks:
        logger.warning("No chunks to upload")
        return
    
    if args.clear_collection:
        logger.info("Deleting collection before upload...")
        delete_collection(collection_name=args.collection_name)
    
    upload_to_chromadb(
        chunks=chunks,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
