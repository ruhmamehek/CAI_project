#!/usr/bin/env python
"""CLI utility to build vector indices for the RAG system."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rag.indexing import CorpusConfig, IndexBuildConfig, IndexBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for SEC/FOMC corpora.")
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Directory containing SEC filings (e.g. sec-edgar-filings/).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Destination directory to store the FAISS index artefacts.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="sec_filings",
        help="Corpus name stored in the manifest.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=380,
        help="Chunk size in tokens (default: 380).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.3,
        help="Chunk overlap ratio (0-1).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum character length for cleaned text segments.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer model to produce embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--no-faiss-gpu",
        action="store_true",
        help="Disable GPU acceleration for FAISS even if CUDA is available.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    source_dir = Path(args.source_dir).expanduser()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    index_dir = Path(args.index_dir).expanduser()
    index_dir.mkdir(parents=True, exist_ok=True)

    corpus = CorpusConfig(
        name=args.name,
        source_dir=str(source_dir),
        doc_glob="**/full-submission.txt",
    )

    config = IndexBuildConfig(
        index_dir=str(index_dir),
        corpus=corpus,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_char_length=args.min_chars,
        embedding_model=args.embedding_model,
        device=args.device,
        use_faiss_gpu=not args.no_faiss_gpu,
    )

    builder = IndexBuilder(config)
    manifest = builder.build()

    print(f"Index built successfully at {manifest['index_dir']}")
    print(f"Chunks indexed: {manifest['num_chunks']}")
    manifest_path = Path(manifest["index_dir"]) / "index_manifest.json"
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
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
