"""Extract layout chunks from PDF using deep learning layout detection.

This script extracts semantic chunks (Text, Title, List, Table, Figure) from PDFs
using layout-parser's deep learning models. The chunks include coordinates
for later image extraction.
"""

import layoutparser as lp
import json
from pathlib import Path
from typing import List, Dict


def extract_chunks_from_pdf(
    pdf_path: Path,
    output_path: Path,
    model_name: str = 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
    start_page: int = None,
    end_page: int = None
) -> List[Dict]:
    """
    Extract layout chunks from PDF.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Path to save chunks JSON
        model_name: Layout detection model name
        start_page: First page to process (1-indexed, None for start)
        end_page: Last page to process (1-indexed, None for end)
        
    Returns:
        List of chunk dictionaries with text, coordinates, and metadata
    """
    print(f"Loading PDF: {pdf_path}")
    
    # Load PDF with images (required for DL model)
    pdf_layouts, pdf_images = lp.load_pdf(str(pdf_path), load_images=True)
    total_pages = len(pdf_layouts)
    
    # Select page range if specified
    if start_page is not None or end_page is not None:
        start_idx = (start_page - 1) if start_page else 0
        end_idx = end_page if end_page else total_pages
        pdf_layouts = pdf_layouts[start_idx:end_idx]
        pdf_images = pdf_images[start_idx:end_idx]
        page_offset = start_idx  # For correct page numbering in output
        print(f"Loaded pages {start_page or 1}-{end_page or total_pages} ({len(pdf_layouts)} pages from {total_pages} total)\n")
    else:
        page_offset = 0
        print(f"Loaded {len(pdf_layouts)} pages\n")
    
    # Check backend availability
    from layoutparser.file_utils import (
        is_detectron2_available,
        is_effdet_available,
        is_paddle_available
    )
    
    has_detectron2 = is_detectron2_available()
    has_effdet = is_effdet_available()
    has_paddle = is_paddle_available()
    
    print(f"Backend availability:")
    print(f"  Detectron2: {has_detectron2}")
    print(f"  EfficientDet: {has_effdet}")
    print(f"  PaddleDetection: {has_paddle}")
    
    # If no backends available, use basic token extraction
    if not (has_detectron2 or has_effdet or has_paddle):
        print(f"\n⚠ No DL backends available - using basic token extraction")
        print(f"  For better results, install a backend:")
        print(f"    pip install 'layoutparser[layoutmodels]'  # installs all")
        print(f"    OR")
        print(f"    pip install detectron2  # for Detectron2")
        print(f"    pip install effdet  # for EfficientDet")
        print(f"    pip install paddlepaddle  # for PaddleDetection")
        print()
        model = None
    else:
        # Try to load DL model
        print(f"\nLoading model: {model_name}")
        try:
            # Try AutoLayoutModel first
            print(f"  Attempting AutoLayoutModel...")
            model = lp.AutoLayoutModel(model_name)
            
            # If that returns None, try backends directly
            if model is None:
                print("  ⚠ AutoLayoutModel returned None, trying backends directly...")
                
                if has_detectron2 and "PubLayNet" in model_name:
                    print("  Trying Detectron2LayoutModel...")
                    from layoutparser.models import Detectron2LayoutModel
                    model = Detectron2LayoutModel(
                        model_name,
                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                    )
                elif has_effdet:
                    print("  Trying EfficientDetLayoutModel...")
                    from layoutparser.models import EfficientDetLayoutModel
                    model = EfficientDetLayoutModel(model_name)
                elif has_paddle:
                    print("  Trying PaddleDetectionLayoutModel...")
                    from layoutparser.models import PaddleDetectionLayoutModel
                    model = PaddleDetectionLayoutModel(model_name)
            
            if model is None:
                raise ValueError(f"Model returned None for: {model_name}")
            if not hasattr(model, 'detect'):
                raise AttributeError(f"Model does not have 'detect' method. Model type: {type(model)}")
            print(f"✓ Model loaded successfully (type: {type(model).__name__})\n")
        except Exception as e:
            print(f"✗ Error loading DL model: {e}")
            print(f"  Falling back to basic token extraction...\n")
            model = None
    
    all_chunks = []
    
    # Process each page
    for page_idx, (page_layout, page_image) in enumerate(zip(pdf_layouts, pdf_images)):
        print(f"Page {page_idx + 1}/{len(pdf_layouts)}", end=" ... ")
        
        if model is None:
            # Fallback: Basic token extraction (no DL model)
            # Group tokens into simple chunks by proximity
            tokens = list(page_layout)
            if not tokens:
                print("0 chunks (no tokens)")
                continue
            
            # Simple grouping: tokens on same line (similar y-coordinate)
            current_chunk_tokens = []
            current_y = None
            y_tolerance = 5.0
            
            for token in tokens:
                if not hasattr(token, 'text') or not token.text:
                    continue
                
                token_y = token.block.y_1 if hasattr(token.block, 'y_1') else None
                
                if current_y is None or (token_y and abs(token_y - current_y) <= y_tolerance):
                    current_chunk_tokens.append(token)
                    if token_y:
                        current_y = token_y
                else:
                    # Save current chunk and start new one
                    if current_chunk_tokens:
                        chunk_text = " ".join([t.text for t in current_chunk_tokens])
                        if chunk_text.strip():
                            actual_page_num = page_idx + 1 + page_offset
                            chunk = {
                                "page": actual_page_num,
                                "block_id": len(all_chunks),
                                "type": "Text",
                                "text": chunk_text,
                                "num_tokens": len(current_chunk_tokens),
                                "coordinates": {
                                    "x_1": min(t.block.x_1 for t in current_chunk_tokens if hasattr(t.block, 'x_1')),
                                    "x_2": max(t.block.x_2 for t in current_chunk_tokens if hasattr(t.block, 'x_2')),
                                    "y_1": min(t.block.y_1 for t in current_chunk_tokens if hasattr(t.block, 'y_1')),
                                    "y_2": max(t.block.y_2 for t in current_chunk_tokens if hasattr(t.block, 'y_2')),
                                },
                                "is_image": False
                            }
                            all_chunks.append(chunk)
                    current_chunk_tokens = [token]
                    current_y = token_y
            
            # Add final chunk
            if current_chunk_tokens:
                chunk_text = " ".join([t.text for t in current_chunk_tokens])
                if chunk_text.strip():
                    actual_page_num = page_idx + 1 + page_offset
                    chunk = {
                        "page": actual_page_num,
                        "block_id": len(all_chunks),
                        "type": "Text",
                        "text": chunk_text,
                        "num_tokens": len(current_chunk_tokens),
                        "coordinates": {
                            "x_1": min(t.block.x_1 for t in current_chunk_tokens if hasattr(t.block, 'x_1')),
                            "x_2": max(t.block.x_2 for t in current_chunk_tokens if hasattr(t.block, 'x_2')),
                            "y_1": min(t.block.y_1 for t in current_chunk_tokens if hasattr(t.block, 'y_1')),
                            "y_2": max(t.block.y_2 for t in current_chunk_tokens if hasattr(t.block, 'y_2')),
                        },
                        "is_image": False
                    }
                    all_chunks.append(chunk)
            
            print(f"{len([c for c in all_chunks if c['page'] == page_idx + 1 + page_offset])} chunks (basic)")
            continue
        
        # DL model available - use layout detection
        # Detect layout blocks
        try:
            detected_layout = model.detect(page_image)
        except Exception as e:
            print(f"Error detecting layout: {e}")
            continue
        
        # Separate by type
        text_blocks = lp.Layout([b for b in detected_layout if b.type in ['Text', 'Title', 'List']])
        table_blocks = lp.Layout([b for b in detected_layout if b.type == 'Table'])
        figure_blocks = lp.Layout([b for b in detected_layout if b.type == 'Figure'])
        
        # Remove text blocks inside figures
        text_blocks = lp.Layout([b for b in text_blocks 
                               if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
        
        # Process all blocks
        page_blocks = []
        
        # Text blocks (Text, Title, List)
        for block in text_blocks:
            tokens = page_layout.filter_by(block.block, center=True)
            text = " ".join([t.text for t in tokens if hasattr(t, 'text')])
            if text.strip():
                page_blocks.append({
                    "block": block,
                    "type": block.type,
                    "text": text,
                    "num_tokens": len(tokens)
                })
        
        # Table blocks
        for block in table_blocks:
            tokens = page_layout.filter_by(block.block, center=True)
            text = " ".join([t.text for t in tokens if hasattr(t, 'text')])
            page_blocks.append({
                "block": block,
                "type": "Table",
                "text": text,
                "num_tokens": len(tokens)
            })
        
        # Figure blocks (images)
        for block in figure_blocks:
            page_blocks.append({
                "block": block,
                "type": "Figure",
                "text": "",
                "num_tokens": 0
            })
        
        # Sort by reading order
        page_blocks.sort(key=lambda b: (b["block"].coordinates[1], b["block"].coordinates[0]))
        
        # Create chunks
        actual_page_num = page_idx + 1 + page_offset  # Correct page number
        for block_idx, block_data in enumerate(page_blocks):
            block = block_data["block"]
            chunk = {
                "page": actual_page_num,
                "block_id": block_idx,
                "type": block_data["type"],
                "text": block_data["text"],
                "num_tokens": block_data["num_tokens"],
                "coordinates": {
                    "x_1": float(block.block.x_1),
                    "x_2": float(block.block.x_2),
                    "y_1": float(block.block.y_1),
                    "y_2": float(block.block.y_2),
                },
                "is_image": block_data["type"] == "Figure"
            }
            all_chunks.append(chunk)
        
        print(f"{len(page_blocks)} chunks")
    
    # Save chunks
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total chunks: {len(all_chunks):,}")
    
    # Count by type
    type_counts = {}
    for chunk in all_chunks:
        chunk_type = chunk["type"]
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
    
    print(f"\nBy type:")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  {chunk_type}: {count:,}")
    
    print(f"\n✓ Saved to: {output_path}")
    
    return all_chunks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract layout chunks from PDF")
    parser.add_argument("--pdf", type=str, default="data/sec_pdfs/tsla-20231231.pdf",
                       help="Path to PDF file")
    parser.add_argument("--output", type=str, default="data/sec_pdfs/parsed/tsla-20231231_chunks.json",
                       help="Output path for chunks JSON")
    parser.add_argument("--start-page", type=int, default=None,
                       help="First page to process (1-indexed)")
    parser.add_argument("--end-page", type=int, default=None,
                       help="Last page to process (1-indexed, inclusive)")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    output_path = Path(args.output)
    
    chunks = extract_chunks_from_pdf(
        pdf_path, 
        output_path, 
        start_page=args.start_page,
        end_page=args.end_page
    )

