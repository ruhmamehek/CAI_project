"""Process layout-parser JSON files to add item number fields."""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Pattern to match SEC filing item numbers like "Item 1.", "Item 1A.", "Item 2.", etc.
# Only matches specific items: 1, 1A, 1B, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 10, 11, 12, 13, 14, 15
ITEM_PATTERN = re.compile(r'Item\s+(1[0-5]|[1-9][A-Z]?|7A)\b', re.IGNORECASE)

# Allowed types for chunking
ALLOWED_TYPES = ["Text", "List item", "Table"]

# Types to delete
DELETE_TYPES = ["Page header", "Page footer"]


def extract_ticker_and_year(filename: str) -> Dict[str, Optional[str]]:
    """
    Extract ticker and year from filename.
    
    Supports formats like:
    - jnj_2023.json -> ticker: "JNJ", year: 2023
    - tsla-2024.json -> ticker: "TSLA", year: 2024
    
    Args:
        filename: Filename (with or without path)
        
    Returns:
        Dictionary with 'ticker' and 'year' keys
    """
    # Get just the filename without path and extension
    stem = Path(filename).stem
    
    # Pattern to match: TICKER_separator_YEAR or TICKER-separator-YEAR
    # Separator can be underscore, dash, or space
    pattern = r'^([A-Z]+)[_\-](\d{4})$'
    match = re.match(pattern, stem, re.IGNORECASE)
    
    if match:
        ticker = match.group(1).upper()
        year = match.group(2)
        return {'ticker': ticker, 'year': year}
    
    # Try alternative pattern: TICKER followed by year anywhere
    pattern2 = r'([A-Z]+).*?(\d{4})'
    match2 = re.search(pattern2, stem, re.IGNORECASE)
    
    if match2:
        ticker = match2.group(1).upper()
        year = match2.group(2)
        return {'ticker': ticker, 'year': year}
    
    logger.warning(f"Could not extract ticker and year from filename: {filename}")
    return {'ticker': None, 'year': None}


def extract_item_number_from_section_header(text: str) -> Optional[str]:
    """
    Extract item number from section header text if it contains a valid Item header.
    
    Only extracts if the text is a section header with one of the valid Item numbers.
    
    Args:
        text: Text to search for item number
        
    Returns:
        Item number string (e.g., "1", "1A", "2") or None
    """
    if not text:
        return None
    
    # Look for Item pattern in the text
    match = ITEM_PATTERN.search(text.strip())
    if match:
        return match.group(1).upper()  # Normalize to uppercase (e.g., "1a" -> "1A")
    
    return None


def add_item_numbers(
    data: List[Dict], 
    chunk_size: int = 512, 
    tokenizer_name: str = "gpt2",
    ticker: Optional[str] = None,
    year: Optional[str] = None
) -> List[Dict]:
    """
    Process layout-parser data to:
    1. Filter to only include ["Text", "List item", "Table"] types
    2. Delete ["Page header", "Page footer"] types
    3. Detect section headers with Item numbers and assign to subsequent chunks
    4. Aggregate text blocks into chunks of size 512 tokens
    
    Args:
        data: List of objects with 'text' and 'type' fields
        chunk_size: Target chunk size in tokens (default: 512)
        tokenizer_name: Tokenizer model name for counting tokens
        ticker: Ticker symbol to add to each chunk
        year: Year to add to each chunk
        
    Returns:
        List of chunk objects with added 'item_number', 'ticker', and 'year' fields
    """
    # Initialize tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except ImportError:
        logger.warning(f"transformers library not available. Using simple word count approximation.")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}. Using simple word count approximation.")
    
    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        if tokenizer:
            return len(tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=10000))
        else:
            # Fallback: approximate tokens as words * 1.3 (rough estimate)
            return int(len(text.split()) * 1.3)
    
    # Step 1: Filter and process objects
    filtered_objects = []
    for obj in data:
        obj_type = obj.get('type', '')
        text = obj.get('text', '')
        
        # Delete page headers and footers
        if obj_type in DELETE_TYPES:
            continue
        
        # Only keep allowed types
        if obj_type not in ALLOWED_TYPES:
            # Check if it's a section header - we need to process these separately
            if obj_type == "Section header":
                # Check if it contains a valid Item number
                item_num = extract_item_number_from_section_header(text)
                if item_num:
                    # Add as a special marker object
                    marker_obj = obj.copy()
                    marker_obj['_is_item_header'] = True
                    marker_obj['_item_number'] = item_num
                    filtered_objects.append(marker_obj)
            continue
        
        # Add allowed types
        if text.strip():  # Only add if there's actual text
            filtered_objects.append(obj.copy())
    
    # Step 2: Process objects and create chunks with item numbers
    chunks = []
    current_item = None
    current_chunk_texts = []
    current_chunk_token_count = 0
    current_chunk_first_obj = None
    
    def finalize_chunk():
        """Finalize current chunk and add to chunks list."""
        nonlocal current_chunk_texts, current_chunk_token_count, current_chunk_first_obj
        if not current_chunk_texts:
            return
        
        chunk_text = ' '.join(current_chunk_texts)
        chunk_obj = {
            'text': chunk_text,
            'item_number': current_item
        }
        
        # Add ticker and year if provided
        if ticker:
            chunk_obj['ticker'] = ticker
        if year:
            chunk_obj['year'] = year
        
        # Preserve metadata from first object in chunk
        if current_chunk_first_obj:
            if 'page_number' in current_chunk_first_obj:
                chunk_obj['page_number'] = current_chunk_first_obj.get('page_number')
            # Preserve other relevant metadata if needed
            for key in ['left', 'top', 'width', 'height', 'page_width', 'page_height']:
                if key in current_chunk_first_obj:
                    chunk_obj[key] = current_chunk_first_obj.get(key)
        
        chunks.append(chunk_obj)
        current_chunk_texts = []
        current_chunk_token_count = 0
        current_chunk_first_obj = None
    
    for obj in filtered_objects:
        # Check if this is an item header marker
        if obj.get('_is_item_header'):
            # First, finalize current chunk if any
            finalize_chunk()
            
            # Update current item number
            current_item = obj.get('_item_number')
            logger.debug(f"Found item header: Item {current_item}")
            continue
        
        # Regular text block - add to current chunk
        text = obj.get('text', '').strip()
        if not text:
            continue
        
        text_tokens = count_tokens(text)
        
        # If adding this text would exceed chunk size, finalize current chunk
        if current_chunk_token_count + text_tokens > chunk_size and current_chunk_texts:
            finalize_chunk()
        
        # Add text to current chunk
        current_chunk_texts.append(text)
        current_chunk_token_count += text_tokens
        
        # Store first object in chunk for metadata
        if current_chunk_first_obj is None:
            current_chunk_first_obj = obj
    
    # Finalize last chunk if any
    finalize_chunk()
    
    logger.info(f"Created {len(chunks)} chunks from {len(filtered_objects)} filtered objects")
    
    return chunks


def process_json_file(
    input_path: Path, 
    output_path: Optional[Path] = None,
    upload_to_chromadb_flag: bool = False,
    collection_name: str = "sec_filings",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 100
) -> Dict:
    """
    Process a single JSON file to add item numbers.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (defaults to data/processed/{filename})
        
    Returns:
        Dictionary with processing statistics
    """
    if output_path is None:
        # Default to processed folder
        processed_dir = input_path.parent.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / input_path.name
    
    logger.info(f"Processing {input_path}")
    
    # Load JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    
    if not data:
        logger.warning(f"File {input_path} is empty")
        return {'total_objects': 0, 'objects_with_item_number': 0, 'unique_item_numbers': []}
    
    # Check if file is already processed (has chunks but no 'type' field)
    sample_obj = data[0]
    if 'type' not in sample_obj and 'item_number' in sample_obj:
        logger.warning(
            f"File {input_path} appears to already be processed (has 'item_number' field but no 'type' field). "
            f"Skipping. The original file structure with 'type' fields is required for processing."
        )
        return {'total_objects': len(data), 'objects_with_item_number': 0, 'unique_item_numbers': []}
    
    # Extract ticker and year from filename
    file_metadata = extract_ticker_and_year(input_path.name)
    ticker = file_metadata.get('ticker')
    year = file_metadata.get('year')
    
    if ticker and year:
        logger.info(f"Extracted ticker: {ticker}, year: {year} from filename")
    else:
        logger.warning(f"Could not extract ticker and year from filename: {input_path.name}")
    
    # Add item numbers
    processed_data = add_item_numbers(data, ticker=ticker, year=year)
    
    # Count items with item numbers
    items_with_numbers = sum(1 for obj in processed_data if obj.get('item_number') is not None)
    unique_items = set(obj.get('item_number') for obj in processed_data if obj.get('item_number') is not None)
    
    # Add chunk_id to each chunk if not present
    for idx, chunk in enumerate(processed_data):
        if 'chunk_id' not in chunk:
            ticker_str = chunk.get('ticker', 'unknown')
            year_str = chunk.get('year', 'unknown')
            chunk['chunk_id'] = f"{ticker_str}_{year_str}_chunk_{idx}"
    
    # Save processed JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    stats = {
        'total_objects': len(processed_data),
        'objects_with_item_number': items_with_numbers,
        'unique_item_numbers': sorted(unique_items) if unique_items else [],
        'output_path': str(output_path)
    }
    
    logger.info(f"Processed {len(processed_data)} objects, {items_with_numbers} have item numbers")
    logger.info(f"Found item numbers: {stats['unique_item_numbers']}")
    logger.info(f"Saved processed data to {output_path}")
    
    # Upload to ChromaDB if requested
    if upload_to_chromadb_flag:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from data_upload import upload_to_chromadb
            logger.info(f"Uploading {len(processed_data)} chunks to ChromaDB collection '{collection_name}'...")
            upload_to_chromadb(
                chunks=processed_data,
                collection_name=collection_name,
                embedding_model=embedding_model,
                batch_size=batch_size,
                replace=False  # Will replace when processing all files together
            )
            logger.info(f"Successfully uploaded chunks to ChromaDB")
        except ImportError as e:
            logger.error(f"Could not import upload_to_chromadb: {e}. Make sure data_upload.py is available.")
        except Exception as e:
            logger.error(f"Error uploading to ChromaDB: {e}")
            raise
    
    return stats


def process_all_json_files(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    file_pattern: str = "*.json",
    upload_to_chromadb_flag: bool = False,
    collection_name: str = "sec_filings",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 100
):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Output directory (defaults to data/processed/)
        file_pattern: Glob pattern for JSON files
    """
    if output_dir is None:
        # Default to processed folder relative to input_dir
        output_dir = input_dir.parent.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob(file_pattern))
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return
    
    logger.info(f"Processing {len(json_files)} JSON files from {input_dir}")
    
    # If uploading to ChromaDB, collect all chunks first, then upload together
    all_chunks = [] if upload_to_chromadb_flag else None
    
    all_stats = []
    for json_file in json_files:
        try:
            output_path = output_dir / json_file.name
            # Don't upload individual files, we'll upload all at once
            stats = process_json_file(
                json_file, 
                output_path,
                upload_to_chromadb_flag=False
            )
            stats['filename'] = json_file.name
            all_stats.append(stats)
            
            # Collect chunks for batch upload
            if upload_to_chromadb_flag and stats.get('output_path'):
                try:
                    with open(stats['output_path'], 'r', encoding='utf-8') as f:
                        file_chunks = json.load(f)
                        all_chunks.extend(file_chunks)
                except Exception as e:
                    logger.warning(f"Could not load chunks from {stats['output_path']}: {e}")
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    
    # Upload all chunks to ChromaDB at once (replacing existing data)
    if upload_to_chromadb_flag and all_chunks:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from data_upload import upload_to_chromadb
            logger.info(f"Uploading {len(all_chunks)} total chunks to ChromaDB collection '{collection_name}' (replacing existing data)...")
            upload_to_chromadb(
                chunks=all_chunks,
                collection_name=collection_name,
                embedding_model=embedding_model,
                batch_size=batch_size,
                replace=True  # Replace all existing data
            )
            logger.info(f"Successfully uploaded {len(all_chunks)} chunks to ChromaDB")
        except ImportError as e:
            logger.error(f"Could not import upload_to_chromadb: {e}. Make sure data_upload.py is available.")
        except Exception as e:
            logger.error(f"Error uploading to ChromaDB: {e}")
            raise
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    for stats in all_stats:
        print(f"\n{stats['filename']}:")
        print(f"  Total objects: {stats['total_objects']}")
        print(f"  Objects with item number: {stats['objects_with_item_number']}")
        print(f"  Item numbers found: {', '.join(stats['unique_item_numbers'])}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add item numbers to layout-parser JSON files")
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing JSON files to process'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output directory (defaults to input directory)'
    )
    parser.add_argument(
        '--pattern',
        default='*.json',
        help='File pattern for JSON files (default: *.json)'
    )
    parser.add_argument(
        '--file',
        type=Path,
        default=None,
        help='Process a single file instead of directory'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--upload-chromadb',
        action='store_true',
        help='Upload processed chunks to ChromaDB'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='sec_filings',
        help='ChromaDB collection name (default: sec_filings)'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='BAAI/bge-base-en-v1.5',
        help='Embedding model name (default: BAAI/bge-base-en-v1.5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for ChromaDB uploads (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.file:
        # Process single file
        process_json_file(
            args.file, 
            args.output,
            upload_to_chromadb_flag=args.upload_chromadb,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size
        )
    else:
        # Process directory
        process_all_json_files(
            args.input_dir, 
            args.output, 
            args.pattern,
            upload_to_chromadb_flag=args.upload_chromadb,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()

