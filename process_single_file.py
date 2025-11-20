#!/usr/bin/env python3
"""Simple script to process a single JSON file."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from vectordb.data_processing import process_json_file
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_single_file.py <input_json_file> [output_json_file]")
        print("Example: python process_single_file.py data/layout-parser/tsla-2023.json data/processed/tsla-2023.json")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed") / input_file.name
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {input_file}")
    print(f"Output: {output_file}")
    
    try:
        stats = process_json_file(
            input_path=input_file,
            output_path=output_file,
            upload_to_chromadb_flag=False
        )
        
        print(f"\n✅ Processing complete!")
        print(f"   Total chunks: {stats['total_objects']}")
        print(f"   Chunks with item numbers: {stats['objects_with_item_number']}")
        print(f"   Output saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

