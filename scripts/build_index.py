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


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--chunks_file", type=str, help="Path to chunks JSON file")
    parser.add_argument("--output_dir", type=str, help="Output directory for index")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Get paths from config if not provided
    chunks_file = args.chunks_file or Path(config["data"]["processed_dir"]) / "sec_chunks.json"
    output_dir = args.output_dir or config.get("data", {}).get("index_dir", "data/indices")
    
    chunks_file = Path(chunks_file)
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return
    
    # Load chunks
    logger.info(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize retriever
    model_name = config.get("retrieval", {}).get("dense", {}).get("model_name", "BAAI/bge-base-en-v1.5")
    device = config.get("device", "cuda")
    
    retriever = DenseRetriever(model_name=model_name, device=device)
    
    # Build index
    retriever.build_index(chunks)
    
    # Save index
    retriever.save_index(output_dir)
    
    logger.info("Index building complete!")


if __name__ == "__main__":
    main()

