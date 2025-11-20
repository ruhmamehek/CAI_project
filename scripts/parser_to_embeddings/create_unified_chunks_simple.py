"""Create unified chunks from parsed JSON and enriched clippings.

This script combines:
1. Text chunks from parsed JSON (tsla-20231-60-64.json format)
2. Table/Figure chunks from enriched_metadata.json (with Gemini captions)

Into a unified format ready for ChromaDB upload.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_parsed_json(parsed_path: Path) -> List[Dict]:
    """Load parsed JSON with left/top/width/height format."""
    logger.info(f"Loading parsed JSON from: {parsed_path}")
    with open(parsed_path, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    logger.info(f"Loaded {len(elements)} elements")
    return elements


def load_enriched_clippings(enriched_path: Path) -> List[Dict]:
    """Load enriched clippings with captions."""
    if not enriched_path.exists() or str(enriched_path) == '/dev/null':
        logger.warning(f"Enriched clippings file not found: {enriched_path}")
        return []
    logger.info(f"Loading enriched clippings from: {enriched_path}")
    with open(enriched_path, 'r', encoding='utf-8') as f:
        clippings = json.load(f)
    logger.info(f"Loaded {len(clippings)} enriched clippings")
    return clippings


def create_chunk_id(prefix: str, index: int, page: int) -> str:
    """Create a unique chunk ID."""
    return f"{prefix}_{page:03d}_{index:03d}"


def create_enriched_chunk_from_clipping(
    clipping: Dict,
    doc_metadata: Dict,
    clippings_base_dir: Path,
    chunk_index: int
) -> Optional[Dict]:
    """
    Create an enriched chunk from a clipping with caption.
    
    Args:
        clipping: Enriched clipping with caption
        doc_metadata: Document metadata
        clippings_base_dir: Base directory for resolving image paths
        chunk_index: Index for chunk ID
        
    Returns:
        Prepared chunk dict, or None if invalid
    """
    caption = clipping.get('caption', '').strip()
    if not caption:
        return None
    
    page = clipping.get('page', 1)
    chunk_type = clipping.get('type', 'Figure')
    block_id = clipping.get('block_id', chunk_index)
    
    # Resolve image path
    image_path = clipping.get('image_path', clipping.get('filename', ''))
    if not Path(image_path).is_absolute():
        if '/' in image_path:
            if not image_path.startswith('data/'):
                image_path = f"data/sec_pdfs/clippings/{image_path}"
        else:
            pdf_name = clippings_base_dir.parent.name if clippings_base_dir.name == 'clippings' else clippings_base_dir.name
            image_path = f"data/sec_pdfs/clippings/{pdf_name}/clippings/{clipping.get('filename', image_path)}"
    
    chunk = {
        'chunk_id': create_chunk_id(chunk_type.lower(), block_id, page),
        'text': caption,  # Caption is what gets embedded
        'type': chunk_type,
        'page': page,
        'image_path': image_path,
        'filename': clipping.get('filename', ''),
        'ticker': doc_metadata.get('ticker', ''),
        'year': doc_metadata.get('year', ''),
    }
    
    # Add coordinates and size if available
    if 'coordinates' in clipping:
        chunk['coordinates'] = clipping['coordinates']
    if 'size' in clipping:
        chunk['size'] = clipping['size']
    
    return chunk


def prepare_chunks_from_parsed_with_enrichment(
    elements: List[Dict],
    enriched_clippings: List[Dict],
    doc_metadata: Dict,
    clippings_base_dir: Path
) -> List[Dict]:
    """
    Convert parsed JSON elements to chunks, replacing Table/Picture elements
    with enriched versions from clippings.
    
    Args:
        elements: List of elements from parsed JSON
        enriched_clippings: List of enriched clippings with captions
        doc_metadata: Document metadata
        clippings_base_dir: Base directory for resolving image paths
        
    Returns:
        List of prepared chunks in sequential order
    """
    # Create a lookup map for enriched clippings by page and type
    enriched_map = {}
    for clipping in enriched_clippings:
        page = clipping.get('page', 1)
        chunk_type = clipping.get('type', 'Figure')
        # Use block_id or index as key
        key = (page, chunk_type, clipping.get('block_id', 0))
        enriched_map[key] = clipping
    
    prepared = []
    chunk_index = 0
    clipping_usage_count = {}  # Track which clippings we've used
    
    for element in elements:
        element_type = element.get('type', '')
        page = element.get('page_number', 1)
        text = element.get('text', '').strip()
        
        # Check if this is a Table or Picture that we have an enriched version for
        if element_type in ['Table', 'Picture']:
            # Try to find matching enriched clipping
            # For Picture, look for Figure type
            lookup_type = 'Figure' if element_type == 'Picture' else element_type
            
            # Try to match by page and type
            matched_clipping = None
            for key, clipping in enriched_map.items():
                if key[0] == page and key[1] == lookup_type:
                    # Check if we haven't used this clipping yet
                    if key not in clipping_usage_count:
                        matched_clipping = clipping
                        clipping_usage_count[key] = True
                        break
            
            if matched_clipping:
                # Replace with enriched version
                enriched_chunk = create_enriched_chunk_from_clipping(
                    matched_clipping,
                    doc_metadata,
                    clippings_base_dir,
                    chunk_index
                )
                if enriched_chunk:
                    prepared.append(enriched_chunk)
                    chunk_index += 1
                    continue
            else:
                # No enriched version found, skip the original (or could keep it without image)
                logger.debug(f"No enriched clipping found for {element_type} on page {page}, skipping")
                continue
        
        # Process text-related elements
        if element_type in ['Text', 'Section header', 'Title', 'List']:
            if not text:
                continue
            
            # Convert coordinates
            left = element['left']
            top = element['top']
            width = element['width']
            height = element['height']
            
            chunk = {
                'chunk_id': create_chunk_id('text', chunk_index, page),
                'text': text,
                'type': element_type,
                'page': page,
                'ticker': doc_metadata.get('ticker', ''),
                'year': doc_metadata.get('year', ''),
                'coordinates': {
                    'x_1': float(left),
                    'y_1': float(top),
                    'x_2': float(left + width),
                    'y_2': float(top + height)
                }
            }
            prepared.append(chunk)
            chunk_index += 1
    
    return prepared


def prepare_table_figure_chunks_from_enriched(
    clippings: List[Dict],
    doc_metadata: Dict,
    clippings_base_dir: Path
) -> List[Dict]:
    """
    Convert enriched clippings to table/figure chunks.
    
    Args:
        clippings: List of enriched clippings with captions
        doc_metadata: Document metadata
        clippings_base_dir: Base directory for clippings (for resolving image paths)
        
    Returns:
        List of prepared table/figure chunks
    """
    prepared = []
    
    for i, clipping in enumerate(clippings):
        caption = clipping.get('caption', '').strip()
        if not caption:
            logger.warning(f"No caption found for {clipping.get('filename')}, skipping")
            continue
        
        page = clipping.get('page', 1)
        chunk_type = clipping.get('type', 'Figure')
        block_id = clipping.get('block_id', i)
        
        # Resolve image path relative to clippings base directory
        image_path = clipping.get('image_path', clipping.get('filename', ''))
        if not Path(image_path).is_absolute():
            # If image_path is already a relative path, use it directly
            # Otherwise construct from filename
            if '/' in image_path:
                # Already has path structure, use as-is but ensure it's relative to project root
                if not image_path.startswith('data/'):
                    image_path = f"data/sec_pdfs/clippings/{image_path}"
            else:
                # Just filename, construct full path
                # clippings_base_dir should be something like: .../tsla-20231231-60-64/clippings
                # We want: data/sec_pdfs/clippings/tsla-20231231-60-64/clippings/filename
                pdf_name = clippings_base_dir.parent.name if clippings_base_dir.name == 'clippings' else clippings_base_dir.name
                image_path = f"data/sec_pdfs/clippings/{pdf_name}/clippings/{clipping.get('filename', image_path)}"
        
        chunk = {
            'chunk_id': create_chunk_id(chunk_type.lower(), block_id, page),
            'text': caption,  # Caption is what gets embedded
            'type': chunk_type,
            'page': page,
            'image_path': image_path,
            'filename': clipping.get('filename', ''),
            'doc_id': doc_metadata.get('doc_id', ''),
            'ticker': doc_metadata.get('ticker', ''),
            'filing_type': doc_metadata.get('filing_type', ''),
            'accession_number': doc_metadata.get('accession_number', ''),
            'year': doc_metadata.get('year', ''),
            'start_token': 0,
            'end_token': 0,
        }
        
        # Add coordinates and size if available
        if 'coordinates' in clipping:
            chunk['coordinates'] = clipping['coordinates']
        if 'size' in clipping:
            chunk['size'] = clipping['size']
        
        prepared.append(chunk)
    
    return prepared


def sort_chunks_sequentially(chunks: List[Dict]) -> List[Dict]:
    """
    Sort chunks sequentially by page and vertical position (top coordinate).
    
    Args:
        chunks: List of chunks
        
    Returns:
        Sorted list of chunks
    """
    def sort_key(chunk):
        page = chunk.get('page', 0)
        # Get top coordinate for sorting within page
        coords = chunk.get('coordinates', {})
        top = coords.get('y_1', 0) if isinstance(coords, dict) else 0
        return (page, top)
    
    sorted_chunks = sorted(chunks, key=sort_key)
    return sorted_chunks


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create unified chunks from parsed JSON and enriched clippings")
    parser.add_argument(
        "--parsed-json",
        type=str,
        default="data/sec_pdfs/parsed/tsla-20231-60-64.json",
        help="Path to parsed JSON file (left/top/width/height format)"
    )
    parser.add_argument(
        "--enriched-clippings",
        type=str,
        default="data/sec_pdfs/clippings/tsla-20231231-60-64/clippings/enriched_metadata.json",
        help="Path to enriched clippings metadata JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/unified_chunks.json",
        help="Output path for unified chunks JSON"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="TSLA",
        help="Ticker symbol"
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year"
    )
    
    args = parser.parse_args()
    
    # Prepare document metadata - only include what's available
    doc_metadata = {
        'ticker': args.ticker,
        'year': args.year
    }
    
    # Load data
    parsed_elements = load_parsed_json(Path(args.parsed_json))
    
    # Get clippings base directory for resolving paths
    clippings_base_dir = Path(args.enriched_clippings).parent
    
    # Load enriched clippings if available
    enriched_clippings = []
    if Path(args.enriched_clippings).exists():
        enriched_clippings = load_enriched_clippings(Path(args.enriched_clippings))
    else:
        logger.warning(f"Enriched clippings not found: {args.enriched_clippings}")
        logger.info("Creating chunks with text only (no images)")
    
    # Prepare chunks - this will replace Table/Picture elements with enriched versions
    unified_chunks = prepare_chunks_from_parsed_with_enrichment(
        parsed_elements,
        enriched_clippings,
        doc_metadata,
        clippings_base_dir
    )
    
    # Sort sequentially by page and position
    unified_chunks = sort_chunks_sequentially(unified_chunks)
    
    # Save unified chunks
    logger.info(f"Saving unified chunks to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_chunks, f, indent=2, ensure_ascii=False)
    
    # Print summary
    type_counts = {}
    for chunk in unified_chunks:
        chunk_type = chunk.get('type', 'Unknown')
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total chunks: {len(unified_chunks)}")
    logger.info(f"\nBy type:")
    for chunk_type, count in sorted(type_counts.items()):
        logger.info(f"  {chunk_type}: {count}")
    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

