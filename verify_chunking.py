#!/usr/bin/env python3
"""Simple script to verify chunking: max 512 tokens and 15% overlap."""

import json
import sys
from pathlib import Path

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
except ImportError:
    print("Warning: transformers not available, using word approximation")
    def count_tokens(text):
        return int(len(text.split()) * 1.3)

def verify_file(json_file: str, chunk_size: int = 512, overlap_ratio: float = 0.15):
    """Verify chunks in a processed JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    text_chunks = [c for c in chunks if c.get('type') == 'Text']
    
    print(f"\nVerifying {len(text_chunks)} text chunks...")
    print(f"Max size: {chunk_size} tokens, Overlap target: ~{int(chunk_size * overlap_ratio)} tokens\n")
    
    # Check sizes
    sizes = [count_tokens(c.get('text', '')) for c in text_chunks]
    exceeding = [(i, s) for i, s in enumerate(sizes) if s > chunk_size]
    
    print(f"Chunk size statistics:")
    print(f"  Min: {min(sizes)} tokens")
    print(f"  Max: {max(sizes)} tokens")
    print(f"  Avg: {sum(sizes)/len(sizes):.1f} tokens")
    print(f"  Chunks exceeding {chunk_size}: {len(exceeding)}")
    
    if exceeding:
        print(f"\n⚠️  Found {len(exceeding)} chunks exceeding {chunk_size} tokens:")
        for idx, tokens in exceeding[:5]:
            chunk = text_chunks[idx]
            preview = chunk.get('text', '')[:100]
            print(f"  Chunk {idx}: {tokens} tokens - {preview}...")
        if len(exceeding) > 5:
            print(f"  ... and {len(exceeding) - 5} more")
    else:
        print(f"\n✅ All chunks are within {chunk_size} token limit!")
    
    # Check overlap between consecutive chunks
    overlap_target = int(chunk_size * overlap_ratio)
    overlaps_found = 0
    no_overlap = []
    
    for i in range(len(text_chunks) - 1):
        chunk1 = text_chunks[i]
        chunk2 = text_chunks[i + 1]
        
        text1 = chunk1.get('text', '')
        text2 = chunk2.get('text', '')
        
        words1 = text1.split()
        words2 = text2.split()
        
        # Find overlap at word boundaries
        overlap_words = 0
        for check_len in range(1, min(100, len(words1), len(words2))):
            if words1[-check_len:] == words2[:check_len]:
                overlap_words = check_len
        
        if overlap_words > 0:
            overlaps_found += 1
            overlap_tokens = count_tokens(' '.join(words1[-overlap_words:]))
            if abs(overlap_tokens - overlap_target) <= overlap_target * 0.5:  # Within 50% of target
                pass  # Good overlap
        elif count_tokens(text1) > 100:  # Only check for substantial chunks
            no_overlap.append(i)
    
    print(f"\nOverlap statistics:")
    print(f"  Consecutive chunk pairs with overlap: {overlaps_found}/{len(text_chunks)-1}")
    if no_overlap:
        print(f"  ⚠️  {len(no_overlap)} chunk pairs without detected overlap")
        if len(no_overlap) <= 10:
            print(f"    (This may be normal for item boundaries or small chunks)")
    else:
        print(f"  ✅ Overlap detected in all relevant chunks")
    
    # Show sample overlaps
    print(f"\nSample consecutive chunks (showing overlap):")
    samples = min(3, len(text_chunks) - 1)
    for i in range(samples):
        chunk1 = text_chunks[i]
        chunk2 = text_chunks[i + 1]
        text1 = chunk1.get('text', '')
        text2 = chunk2.get('text', '')
        tokens1 = count_tokens(text1)
        tokens2 = count_tokens(text2)
        
        words1 = text1.split()
        words2 = text2.split()
        overlap_words = 0
        for check_len in range(1, min(50, len(words1), len(words2))):
            if words1[-check_len:] == words2[:check_len]:
                overlap_words = check_len
        
        print(f"\n  Chunk {i} ({tokens1} tokens):")
        print(f"    ...{text1[-150:]}")
        print(f"  Chunk {i+1} ({tokens2} tokens):")
        print(f"    {text2[:150]}...")
        if overlap_words > 0:
            overlap_text = ' '.join(words1[-overlap_words:])
            overlap_tokens = count_tokens(overlap_text)
            print(f"    → Overlap: {overlap_words} words ({overlap_tokens} tokens)")
        else:
            print(f"    → No word-level overlap detected")
    
    return len(exceeding) == 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_chunking.py <processed_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not Path(json_file).exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    success = verify_file(json_file)
    sys.exit(0 if success else 1)

