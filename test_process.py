#!/usr/bin/env python3
"""Test script to process tsla-2023.json and verify chunks are working correctly."""

import sys
import json
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

logger = logging.getLogger(__name__)

def main():
    """Process tsla-2023.json and save to test folder."""
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Input file
    input_file = project_root / "data" / "layout-parser" / "tsla-2023.json"
    
    # Output folder
    test_output_dir = project_root / "test_output"
    test_output_dir.mkdir(exist_ok=True)
    
    # Output file
    output_file = test_output_dir / "tsla-2023_processed.json"
    
    print(f"=" * 60)
    print(f"Testing Data Processing Pipeline")
    print(f"=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Output directory: {test_output_dir}")
    print(f"=" * 60)
    print()
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return 1
    
    try:
        # Process the file
        print("Processing file...")
        stats = process_json_file(
            input_path=input_file,
            output_path=output_file,
            upload_to_chromadb_flag=False  # Don't upload to ChromaDB for test
        )
        
        print()
        print(f"=" * 60)
        print(f"Processing Complete!")
        print(f"=" * 60)
        print(f"Total chunks created: {stats['total_objects']}")
        print(f"Chunks with item numbers: {stats['objects_with_item_number']}")
        print(f"Unique item numbers: {stats['unique_item_numbers']}")
        print(f"Output saved to: {stats['output_path']}")
        print()
        
        # Load and analyze the output
        with open(output_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Analyze chunks
        print(f"=" * 60)
        print(f"Chunk Analysis")
        print(f"=" * 60)
        
        # Count by type
        type_counts = {}
        picture_chunks = []
        chunks_with_images = []
        
        for chunk in chunks:
            chunk_type = chunk.get('type', 'Unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            if chunk_type == 'Picture':
                picture_chunks.append(chunk)
            
            if chunk.get('image_path'):
                chunks_with_images.append(chunk)
        
        print(f"\nChunks by type:")
        for chunk_type, count in sorted(type_counts.items()):
            print(f"  {chunk_type}: {count}")
        
        print(f"\nPicture chunks: {len(picture_chunks)}")
        print(f"Chunks with image_path: {len(chunks_with_images)}")
        
        # Show sample chunks
        print(f"\n" + "=" * 60)
        print(f"Sample Chunks")
        print(f"=" * 60)
        
        # Sample text chunk
        text_chunks = [c for c in chunks if c.get('type') == 'Text']
        if text_chunks:
            print(f"\nSample Text Chunk:")
            sample = text_chunks[0]
            print(f"  Chunk ID: {sample.get('chunk_id')}")
            print(f"  Ticker: {sample.get('ticker')}")
            print(f"  Year: {sample.get('year')}")
            print(f"  Item Number: {sample.get('item_number')}")
            print(f"  Page: {sample.get('page_number')}")
            print(f"  Text preview: {sample.get('text', '')[:200]}...")
        
        # Sample picture chunk
        if picture_chunks:
            print(f"\nSample Picture Chunk:")
            sample = picture_chunks[0]
            print(f"  Chunk ID: {sample.get('chunk_id')}")
            print(f"  Ticker: {sample.get('ticker')}")
            print(f"  Year: {sample.get('year')}")
            print(f"  Item Number: {sample.get('item_number')}")
            print(f"  Page: {sample.get('page_number')}")
            print(f"  Image Path: {sample.get('image_path')}")
            print(f"  Description preview: {sample.get('text', '')[:200]}...")
            
            # Check if image file exists
            if sample.get('image_path'):
                img_path = project_root / sample['image_path']
                if img_path.exists():
                    print(f"  ✓ Image file exists: {img_path}")
                else:
                    print(f"  ✗ Image file NOT found: {img_path}")
        
        # Show item number distribution
        item_distribution = {}
        for chunk in chunks:
            item_num = chunk.get('item_number')
            if item_num:
                item_distribution[item_num] = item_distribution.get(item_num, 0) + 1
        
        if item_distribution:
            print(f"\n" + "=" * 60)
            print(f"Item Number Distribution")
            print(f"=" * 60)
            for item_num in sorted(item_distribution.keys()):
                print(f"  Item {item_num}: {item_distribution[item_num]} chunks")
        
        print(f"\n" + "=" * 60)
        print(f"Test Complete!")
        print(f"=" * 60)
        print(f"Check the output file for full details: {output_file}")
        print(f"Check test_output/images/ for clipped images")
        print(f"=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

