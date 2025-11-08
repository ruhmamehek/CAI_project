"""Process raw documents: parse and chunk."""

import argparse
import sys
import json
import re
from pathlib import Path
import yaml
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.acquisition import SECFilingDownloader
from src.data.processing import DocumentProcessor, Chunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_metadata_from_path(file_path: Path, sec_dir: Path) -> dict:
    """Extract metadata from SEC filing file path.
    
    Path structure: sec-edgar-filings/{TICKER}/{FILING_TYPE}/{ACCESSION}/full-submission.txt
    """
    try:
        relative_path = file_path.relative_to(sec_dir)
        parts = relative_path.parts
        
        ticker = parts[0] if len(parts) > 0 else None
        filing_type = parts[1] if len(parts) > 1 else None
        accession = parts[2] if len(parts) > 2 else None
        
        year = None
        if accession:
            year_match = re.search(r'-(\d{2})-', accession)
            if year_match:
                year_2digit = int(year_match.group(1))
                year = 2000 + year_2digit if year_2digit < 50 else 1900 + year_2digit
        
        return {
            "ticker": ticker,
            "filing_type": filing_type,
            "accession_number": accession,
            "year": year
        }
    except Exception as e:
        logger.warning(f"Could not extract metadata from path {file_path}: {e}")
        return {}


def process_sec_filings(raw_dir: Path, processed_dir: Path, config: dict):
    """Process SEC filings."""
    logger.info("Processing SEC filings...")
    
    processor = DocumentProcessor()
    chunker = Chunker(
        chunk_size=config["data"]["chunk_size"],
        chunk_overlap=config["data"]["chunk_overlap"]
    )
    
    sec_dir = raw_dir.parent / "sec-edgar-filings"
    
    if not sec_dir.exists():
        logger.warning(f"SEC filings directory not found: {sec_dir}")
        logger.info("Expected location: data/sec-edgar-filings")
        return
    
    all_chunks = []
    
    for filing_file in sec_dir.rglob("full-submission.txt"):
        logger.info(f"Processing {filing_file.relative_to(sec_dir)}")
        text = processor.parse_sec_filing(filing_file)
        doc_id = filing_file.parent.name
        
        metadata = extract_metadata_from_path(filing_file, sec_dir)
        chunks = chunker.chunk_text(text, doc_id, metadata=metadata)
        all_chunks.extend(chunks)
    
    output_file = processed_dir / "sec_chunks.json"
    with open(output_file, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    logger.info(f"Saved {len(all_chunks)} chunks to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process raw documents")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--download", action="store_true", help="Download data first")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if args.download:
        logger.info("Downloading data...")
        
        if config["data"]["sec"]["companies"]:
            sec_downloader = SECFilingDownloader(
                str(raw_dir / "sec"),
                config["data"]["sec"]["email"]
            )
            sec_downloader.download_filings(
                config["data"]["sec"]["companies"],
                config["data"]["sec"]["filing_types"],
                config["data"]["sec"]["years"]
            )
    
    process_sec_filings(raw_dir, processed_dir, config)
    logger.info("Data processing complete!")


if __name__ == "__main__":
    main()
