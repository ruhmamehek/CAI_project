#!/usr/bin/env python3
"""Test script to verify chunking logic: 15% overlap and max 512 tokens."""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from vectordb.data_processing import add_item_numbers, extract_ticker_and_year
from transformers import AutoTokenizer

def count_tokens_exact(text: str, tokenizer) -> int:
    """Count tokens using tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=10000))

def verify_chunks(chunks, chunk_size=512, overlap_ratio=0.15):
    """Verify chunks meet requirements: max 512 tokens, 15% overlap."""
    print("\n" + "="*60)
    print("Chunk Verification")
    print("="*60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        print("Warning: Could not load tokenizer, using approximate token counting")
        tokenizer = None
    
    def count_tokens(text):
        if tokenizer:
            return count_tokens_exact(text, tokenizer)
        else:
            return int(len(text.split()) * 1.3)
    
    issues = []
    overlap_target = int(chunk_size * overlap_ratio)  # ~77 tokens for 512
    
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get('text', '')
        chunk_tokens = count_tokens(chunk_text)
        
        # Check 1: Chunk size doesn't exceed max
        if chunk_tokens > chunk_size:
            issues.append(f"Chunk {i}: {chunk_tokens} tokens exceeds max {chunk_size}")
        
        # Check 2: Verify overlap with next chunk
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            next_chunk_text = next_chunk.get('text', '')
            
            # Check if there's overlap (simple check: last part of current chunk should appear in next chunk)
            # Calculate what should be in overlap (~15% of current chunk or 15% of chunk_size)
            current_chunk_words = chunk_text.split()
            next_chunk_words = next_chunk_text.split()
            
            # Find overlap by checking how many words from end of current chunk appear at start of next chunk
            overlap_found = 0
            max_check = min(len(current_chunk_words), len(next_chunk_words), int(chunk_size * overlap_ratio / 1.3))
            
            for check_len in range(1, max_check + 1):
                current_end = ' '.join(current_chunk_words[-check_len:])
                next_start = ' '.join(next_chunk_words[:check_len])
                if current_end == next_start:
                    overlap_found = check_len
            
            if overlap_found == 0 and chunk_tokens > 100:
                # Only warn if chunk is substantial (small chunks might not have overlap)
                issues.append(f"Chunk {i} -> {i+1}: No overlap detected (chunk {i} has {chunk_tokens} tokens)")
            elif overlap_found > 0:
                overlap_tokens = count_tokens(' '.join(current_chunk_words[-overlap_found:]))
                expected_overlap = int(chunk_size * overlap_ratio)
                if abs(overlap_tokens - expected_overlap) > expected_overlap * 0.5:  # Allow 50% variance
                    issues.append(f"Chunk {i} -> {i+1}: Overlap is {overlap_tokens} tokens (expected ~{expected_overlap})")
    
    # Print statistics
    chunk_sizes = [count_tokens(chunk.get('text', '')) for chunk in chunks]
    text_chunks = [c for c in chunks if c.get('type') == 'Text']
    
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Text chunks: {len(text_chunks)}")
    print(f"\nChunk size statistics:")
    print(f"  Min: {min(chunk_sizes) if chunk_sizes else 0} tokens")
    print(f"  Max: {max(chunk_sizes) if chunk_sizes else 0} tokens")
    print(f"  Average: {sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0:.1f} tokens")
    print(f"  Chunks exceeding {chunk_size}: {sum(1 for s in chunk_sizes if s > chunk_size)}")
    
    # Check overlap
    text_chunk_sizes = [count_tokens(c.get('text', '')) for c in text_chunks]
    print(f"\nText chunk sizes:")
    if text_chunk_sizes:
        print(f"  Min: {min(text_chunk_sizes)} tokens")
        print(f"  Max: {max(text_chunk_sizes)} tokens")
        print(f"  Average: {sum(text_chunk_sizes) / len(text_chunk_sizes):.1f} tokens")
    
    # Show sample consecutive chunks with overlap
    print(f"\nSample consecutive chunks (showing overlap):")
    for i in range(min(3, len(text_chunks) - 1)):
        chunk1 = text_chunks[i]
        chunk2 = text_chunks[i + 1]
        text1 = chunk1.get('text', '')
        text2 = chunk2.get('text', '')
        tokens1 = count_tokens(text1)
        tokens2 = count_tokens(text2)
        
        # Find overlap
        words1 = text1.split()
        words2 = text2.split()
        overlap_words = 0
        for check_len in range(1, min(50, len(words1), len(words2))):
            if words1[-check_len:] == words2[:check_len]:
                overlap_words = check_len
        
        print(f"\n  Chunk {i}:")
        print(f"    Size: {tokens1} tokens")
        print(f"    Preview: {text1[:150]}...")
        print(f"  Chunk {i+1}:")
        print(f"    Size: {tokens2} tokens")
        print(f"    Overlap: {overlap_words} words ({count_tokens(' '.join(words1[-overlap_words:])) if overlap_words > 0 else 0} tokens)")
        print(f"    Preview: {text2[:150]}...")
    
    # Report issues
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print(f"\n✅ All checks passed!")
    
    return len(issues) == 0

if __name__ == "__main__":
    # Load processed file
    processed_file = Path("test_output/tsla-2023_processed.json")
    
    if not processed_file.exists():
        print(f"Error: Processed file not found: {processed_file}")
        print("Run test_process.py first to generate processed file.")
        sys.exit(1)
    
    with open(processed_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Verifying {len(chunks)} chunks...")
    success = verify_chunks(chunks, chunk_size=512, overlap_ratio=0.15)
    
    sys.exit(0 if success else 1)

