"""Query the RAG pipeline."""

import argparse
import sys
from pathlib import Path
import yaml
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Query RAG pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--ticker", type=str, help="Filter by company ticker (e.g., AAPL)")
    parser.add_argument("--filing_type", type=str, help="Filter by filing type (e.g., 10-K)")
    parser.add_argument("--year", type=int, help="Filter by year (e.g., 2023)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    index_dir = config.get("data", {}).get("index_dir", "data/indices")
    
    logger.info("Initializing RAG pipeline...")
    retrieval_config = config.get("retrieval", {}).get("dense", {})
    generation_config = config.get("generation", {})
    device = config.get("device", "auto")
    
    rag = RAGPipeline(
        index_dir=index_dir,
        retrieval_config={
            "model_name": retrieval_config.get("model_name", "BAAI/bge-base-en-v1.5"),
            "device": device
        },
        generation_config={
            "model_name": generation_config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            "device": device,
            "max_new_tokens": generation_config.get("max_new_tokens", 256),
            "temperature": generation_config.get("temperature", 0.7)
        }
    )
    logger.info("RAG pipeline ready!")
    
    filter_metadata = {}
    if args.ticker:
        filter_metadata["ticker"] = args.ticker
    if args.filing_type:
        filter_metadata["filing_type"] = args.filing_type
    if args.year:
        filter_metadata["year"] = args.year
    
    if args.interactive:
        print("\n" + "="*60)
        print("RAG Pipeline - Interactive Mode")
        print("Type 'quit' or 'exit' to stop")
        if filter_metadata:
            print(f"Metadata filter: {filter_metadata}")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                
                result = rag.query(question, top_k=args.top_k, filter_metadata=filter_metadata if filter_metadata else None)
                
                print("\n" + "-"*60)
                print(f"Answer: {result['answer']}")
                if result['citations']:
                    print(f"Citations: {result['citations']}")
                print("-"*60 + "\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}\n")
    
    elif args.query:
        result = rag.query(args.query, top_k=args.top_k, filter_metadata=filter_metadata if filter_metadata else None)
        
        print("\n" + "="*60)
        print(f"Question: {result['question']}")
        if filter_metadata:
            print(f"Filter: {filter_metadata}")
        print("="*60)
        print(f"\nAnswer: {result['answer']}\n")
        if result['citations']:
            print(f"Citations: {result['citations']}\n")
        print("="*60)
    
    else:
        parser.print_help()
        print("\nUse --query 'your question' or --interactive for interactive mode")


if __name__ == "__main__":
    main()
