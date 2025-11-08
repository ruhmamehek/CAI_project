"""Build FAISS index from processed chunks."""

import argparse
import sys
import json
from pathlib import Path
import yaml
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.dense_retriever import DenseRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def group_chunks_by_metadata(chunks):
    """Group chunks by ticker and year."""
    grouped = {}
    for chunk in chunks:
        ticker = chunk.get("ticker")
        year = chunk.get("year")
        
        if not ticker:
            ticker = "unknown"
        if not year:
            year = "unknown"
        
        key = (ticker, year)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(chunk)
    
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--chunks_file", type=str, help="Path to chunks JSON file")
    parser.add_argument("--output_dir", type=str, help="Output directory for index")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    chunks_file = args.chunks_file or Path(config["data"]["processed_dir"]) / "sec_chunks.json"
    output_dir = args.output_dir or config.get("data", {}).get("index_dir", "data/indices")
    
    chunks_file = Path(chunks_file)
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return
    
    logger.info(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    grouped_chunks = group_chunks_by_metadata(chunks)
    logger.info(f"Grouped into {len(grouped_chunks)} company/year combinations")
    
    model_name = config.get("retrieval", {}).get("dense", {}).get("model_name", "BAAI/bge-base-en-v1.5")
    device = config.get("device", "cuda")
    
    retriever = DenseRetriever(model_name=model_name, device=device)
    
    index_info = {}
    
    for (ticker, year), group_chunks in grouped_chunks.items():
        logger.info(f"Building index for {ticker}/{year} ({len(group_chunks)} chunks)...")
        
        retriever.build_index(group_chunks)
        
        index_path = Path(output_dir) / str(ticker) / str(year)
        retriever.save_index(str(index_path))
        
        index_info[f"{ticker}/{year}"] = {
            "ticker": ticker,
            "year": year,
            "num_chunks": len(group_chunks),
            "path": str(index_path)
        }
    
    index_manifest = Path(output_dir) / "index_manifest.json"
    with open(index_manifest, 'w') as f:
        json.dump(index_info, f, indent=2)
    
    logger.info(f"Index building complete! Created {len(grouped_chunks)} indices")
    logger.info(f"Index manifest saved to {index_manifest}")


if __name__ == "__main__":
    main()
