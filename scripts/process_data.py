"""Process raw documents: parse and chunk."""

import argparse
import sys
import json
from pathlib import Path
import yaml
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.acquisition import SECFilingDownloader, FOMCDownloader
from src.data.processing import DocumentProcessor, Chunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_sec_filings(raw_dir: Path, processed_dir: Path, config: dict):
    """Process SEC filings."""
    logger.info("Processing SEC filings...")
    
    processor = DocumentProcessor()
    chunker = Chunker(
        chunk_size=config["data"]["chunk_size"],
        chunk_overlap=config["data"]["chunk_overlap"]
    )
    
    sec_dir = raw_dir / "sec"
    if not sec_dir.exists():
        logger.warning(f"SEC directory not found: {sec_dir}")
        return
    
    all_chunks = []
    
    # Process each filing
    for filing_file in sec_dir.rglob("*.txt"):
        logger.info(f"Processing {filing_file.name}")
        text = processor.parse_sec_filing(filing_file)
        doc_id = filing_file.stem
        chunks = chunker.chunk_text(text, doc_id)
        all_chunks.extend(chunks)
    
    # Save processed chunks
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
    
    # Download data if requested
    if args.download:
        logger.info("Downloading data...")
        
        # Download SEC filings
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
        
        # Download FOMC texts
        fomc_downloader = FOMCDownloader(str(raw_dir / "fomc"))
        fomc_downloader.download_fomc_texts(config["data"]["fomc"]["years"])
    
    # Process documents
    process_sec_filings(raw_dir, processed_dir, config)
    
    logger.info("Data processing complete!")


if __name__ == "__main__":
    main()

