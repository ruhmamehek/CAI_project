"""End-to-end pipeline to process all SEC filings JSON files.

This script:
1. Reads all JSON files from sec_filings_json/ folder
2. For each file (ticker_year.json):
   - Extracts table/figure clippings from PDF
   - Generates captions using Gemini
   - Creates unified chunks (text + table/figure)
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse filename like 'tsla_2023.json' to extract ticker and year.
    
    Args:
        filename: Filename (e.g., 'tsla_2023.json')
        
    Returns:
        Dict with 'ticker' and 'year', or None if parsing fails
    """
    try:
        name = Path(filename).stem  # Remove .json extension
        parts = name.split('_')
        if len(parts) >= 2:
            ticker = parts[0].upper()
            year = parts[1]
            return {'ticker': ticker, 'year': year}
        else:
            logger.warning(f"Could not parse filename: {filename}")
            return None
    except Exception as e:
        logger.error(f"Error parsing filename {filename}: {e}")
        return None


def find_pdf_for_json(json_path: Path, pdf_base_dir: Path) -> Optional[Path]:
    """
    Find the corresponding PDF file for a JSON file.
    
    Looks for PDFs matching patterns like:
    - {ticker}-{year}*.pdf
    - {ticker}-{year}-*.pdf
    
    Args:
        json_path: Path to JSON file
        pdf_base_dir: Base directory to search for PDFs
        
    Returns:
        Path to PDF file, or None if not found
    """
    filename = json_path.stem
    parts = filename.split('_')
    if len(parts) < 2:
        return None
    
    ticker = parts[0].lower()
    year = parts[1]
    
    # Try different patterns
    patterns = [
        f"{ticker}-{year}*.pdf",
        f"{ticker}-{year}-*.pdf",
        f"{ticker}*{year}*.pdf"
    ]
    
    for pattern in patterns:
        matches = list(pdf_base_dir.glob(pattern))
        if matches:
            logger.info(f"Found PDF: {matches[0]}")
            return matches[0]
    
    logger.warning(f"Could not find PDF for {json_path}")
    return None


def extract_clippings(
    json_path: Path,
    pdf_path: Path,
    output_base_dir: Path,
    page_offset: int = 0
) -> Optional[Path]:
    """
    Extract clippings from PDF using parsed JSON.
    
    Args:
        json_path: Path to parsed JSON file
        pdf_path: Path to PDF file
        output_base_dir: Base directory for output
        page_offset: Page offset (JSON page 1 = PDF page offset+1)
        
    Returns:
        Path to clippings_metadata.json, or None if failed
    """
    logger.info(f"Extracting clippings for {json_path.name}...")
    
    pdf_name = pdf_path.stem
    clippings_dir = output_base_dir / pdf_name / "clippings"
    clippings_metadata = clippings_dir / "clippings_metadata.json"
    
    # Check if already exists
    if clippings_metadata.exists():
        logger.info(f"  Clippings already exist, skipping extraction")
        return clippings_metadata
    
    try:
        cmd = [
            sys.executable,
            "scripts/parser_to_embeddings/extract_clippings_from_parsed.py",
            "--pdf", str(pdf_path),
            "--parsed", str(json_path),
            "--output", str(output_base_dir),
            "--page-offset", str(page_offset)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"  Failed to extract clippings: {result.stderr}")
            return None
        
        if clippings_metadata.exists():
            logger.info(f"  ✓ Clippings extracted to {clippings_metadata}")
            return clippings_metadata
        else:
            logger.error(f"  Clippings metadata not created")
            return None
            
    except Exception as e:
        logger.error(f"  Error extracting clippings: {e}")
        return None


def generate_captions(
    clippings_metadata_path: Path,
    clippings_dir: Path,
    enriched_output_path: Path,
    parsed_json_path: Optional[Path] = None,
    model: str = "gemini-2.5-flash"
) -> bool:
    """
    Generate captions for clippings using Gemini.
    
    Args:
        clippings_metadata_path: Path to clippings_metadata.json
        clippings_dir: Directory containing clipping images
        enriched_output_path: Path to save enriched_metadata.json
        parsed_json_path: Optional path to parsed JSON for context
        model: Gemini model name
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Generating captions for {clippings_metadata_path.name}...")
    
    # Check if already exists
    if enriched_output_path.exists():
        logger.info(f"  Enriched metadata already exists, skipping caption generation")
        return True
    
    try:
        cmd = [
            sys.executable,
            "scripts/parser_to_embeddings/generate_captions.py",
            "--clippings-metadata", str(clippings_metadata_path),
            "--clippings-dir", str(clippings_dir),
            "--output", str(enriched_output_path),
            "--model", model
        ]
        
        if parsed_json_path and parsed_json_path.exists():
            cmd.extend(["--chunks-json", str(parsed_json_path)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"  Failed to generate captions: {result.stderr}")
            return False
        
        if enriched_output_path.exists():
            logger.info(f"  ✓ Captions generated: {enriched_output_path}")
            return True
        else:
            logger.error(f"  Enriched metadata not created")
            return False
            
    except Exception as e:
        logger.error(f"  Error generating captions: {e}")
        return False


def create_unified_chunks(
    parsed_json_path: Path,
    enriched_clippings_path: Optional[Path],
    output_path: Path,
    ticker: str,
    year: str
) -> bool:
    """
    Create unified chunks from parsed JSON and enriched clippings.
    
    Args:
        parsed_json_path: Path to parsed JSON file
        enriched_clippings_path: Path to enriched_metadata.json
        output_path: Path to save unified_chunks.json
        ticker: Ticker symbol
        year: Year
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating unified chunks for {ticker} {year}...")
    
    try:
        cmd = [
            sys.executable,
            "scripts/parser_to_embeddings/create_unified_chunks_simple.py",
            "--parsed-json", str(parsed_json_path),
            "--output", str(output_path),
            "--ticker", ticker,
            "--year", year
        ]
        
        # Add enriched clippings path if available
        if enriched_clippings_path and enriched_clippings_path.exists():
            cmd.extend(["--enriched-clippings", str(enriched_clippings_path)])
        else:
            # Use a dummy path - script will handle it gracefully
            cmd.extend(["--enriched-clippings", "/dev/null"])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"  Failed to create unified chunks: {result.stderr}")
            return False
        
        if output_path.exists():
            logger.info(f"  ✓ Unified chunks created: {output_path}")
            return True
        else:
            logger.error(f"  Unified chunks not created")
            return False
            
    except Exception as e:
        logger.error(f"  Error creating unified chunks: {e}")
        return False


def process_single_filing(
    json_path: Path,
    pdf_base_dir: Path,
    clippings_base_dir: Path,
    unified_chunks_dir: Path,
    model: str = "gemini-2.5-flash",
    page_offset: int = 0
) -> bool:
    """
    Process a single filing JSON file through the complete pipeline.
    
    Args:
        json_path: Path to parsed JSON file (ticker_year.json)
        pdf_base_dir: Base directory to search for PDFs
        clippings_base_dir: Base directory for clippings output
        unified_chunks_dir: Directory for unified chunks output
        model: Gemini model name
        page_offset: Page offset for PDF pages
        
    Returns:
        True if all steps succeeded, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {json_path.name}")
    logger.info(f"{'='*60}")
    
    # Parse filename
    file_info = parse_filename(json_path.name)
    if not file_info:
        logger.error(f"Could not parse filename: {json_path.name}")
        return False
    
    ticker = file_info['ticker']
    year = file_info['year']
    
    # Find PDF
    pdf_path = find_pdf_for_json(json_path, pdf_base_dir)
    if not pdf_path:
        logger.warning(f"PDF not found for {json_path.name}, skipping clipping extraction")
        # Continue anyway - might be able to generate captions if clippings exist
    
    # Step 1: Extract clippings
    clippings_metadata_path = None
    if pdf_path:
        clippings_metadata_path = extract_clippings(
            json_path, pdf_path, clippings_base_dir, page_offset
        )
    else:
        # Try to find existing clippings
        pdf_name = json_path.stem.replace('_', '-')
        potential_clippings = clippings_base_dir.glob(f"*/clippings/clippings_metadata.json")
        for p in potential_clippings:
            if pdf_name in str(p):
                clippings_metadata_path = p
                logger.info(f"  Found existing clippings: {clippings_metadata_path}")
                break
    
    if not clippings_metadata_path or not clippings_metadata_path.exists():
        logger.warning(f"  No clippings found, skipping caption generation")
        # Still create unified chunks with just text
        clippings_dir = None
        enriched_path = None
    else:
        clippings_dir = clippings_metadata_path.parent
        
        # Step 2: Generate captions
        enriched_path = clippings_dir / "enriched_metadata.json"
        if not generate_captions(
            clippings_metadata_path,
            clippings_dir,
            enriched_path,
            parsed_json_path=json_path,
            model=model
        ):
            logger.warning(f"  Caption generation failed, continuing without images")
            enriched_path = None
    
    # Step 3: Create unified chunks
    unified_output = unified_chunks_dir / f"{ticker}_{year}_unified_chunks.json"
    
    # Always call create_unified_chunks - it handles missing enriched clippings gracefully
    success = create_unified_chunks(
        json_path,
        enriched_path if (enriched_path and enriched_path.exists()) else None,
        unified_output,
        ticker,
        year
    )
    
    if success:
        logger.info(f"✓ Successfully processed {json_path.name}")
        return True
    else:
        logger.error(f"✗ Failed to process {json_path.name}")
        return False


def main():
    """Main function to process all filings."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process all SEC filings through complete pipeline")
    parser.add_argument(
        "--json-dir",
        type=str,
        default="sec_filings_json",
        help="Directory containing parsed JSON files (ticker_year.json format)"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/sec_pdfs",
        help="Base directory to search for PDF files"
    )
    parser.add_argument(
        "--clippings-dir",
        type=str,
        default="data/sec_pdfs/clippings",
        help="Base directory for clippings output"
    )
    parser.add_argument(
        "--unified-dir",
        type=str,
        default="data/processed",
        help="Directory for unified chunks output"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name"
    )
    parser.add_argument(
        "--page-offset",
        type=int,
        default=0,
        help="Page offset (JSON page 1 = PDF page offset+1)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process only a specific file (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    json_dir = Path(args.json_dir)
    pdf_base_dir = Path(args.pdf_dir)
    clippings_base_dir = Path(args.clippings_dir)
    unified_chunks_dir = Path(args.unified_dir)
    
    # Create output directories
    clippings_base_dir.mkdir(parents=True, exist_ok=True)
    unified_chunks_dir.mkdir(parents=True, exist_ok=True)
    
    if not json_dir.exists():
        logger.error(f"JSON directory not found: {json_dir}")
        return
    
    # Find all JSON files
    if args.file:
        json_files = [json_dir / args.file]
    else:
        json_files = sorted(json_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {json_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON file(s) to process")
    
    # Process each file
    success_count = 0
    failed_files = []
    
    for json_file in json_files:
        success = process_single_filing(
            json_file,
            pdf_base_dir,
            clippings_base_dir,
            unified_chunks_dir,
            model=args.model,
            page_offset=args.page_offset
        )
        
        if success:
            success_count += 1
        else:
            failed_files.append(json_file.name)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {len(json_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_files)}")
    
    if failed_files:
        logger.info(f"\nFailed files:")
        for f in failed_files:
            logger.info(f"  - {f}")
    
    logger.info(f"\nUnified chunks saved to: {unified_chunks_dir}")


if __name__ == "__main__":
    main()

