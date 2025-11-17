"""Extract table and figure clippings from PDF based on parsed JSON coordinates.

This script reads a parsed JSON file (with left/top/width/height format) and 
extracts image clippings for all Table and Picture/Figure elements.
"""

import json
import layoutparser as lp
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Optional


def convert_coordinates(element: Dict) -> Dict:
    """
    Convert coordinates from left/top/width/height to x_1/y_1/x_2/y_2 format.
    
    Args:
        element: Element dict with left, top, width, height
        
    Returns:
        Dict with x_1, y_1, x_2, y_2
    """
    left = element["left"]
    top = element["top"]
    width = element["width"]
    height = element["height"]
    
    return {
        "x_1": float(left),
        "y_1": float(top),
        "x_2": float(left + width),
        "y_2": float(top + height)
    }


def extract_clippings_from_parsed(
    pdf_path: Path,
    parsed_json_path: Path,
    output_dir: Path,
    chunk_types: List[str] = ["Table", "Picture", "Figure"],
    page_offset: int = 0
) -> List[Dict]:
    """
    Extract clippings for specified chunk types from PDF using parsed JSON.
    
    Args:
        pdf_path: Path to PDF file
        parsed_json_path: Path to parsed JSON file (with left/top/width/height format)
        output_dir: Directory to save clippings
        chunk_types: List of chunk types to extract (default: ["Table", "Picture", "Figure"])
        page_offset: Offset to add to page numbers (e.g., if JSON page 1 = PDF page 60, use 59)
        
    Returns:
        List of dictionaries with clipping metadata
    """
    print(f"Loading parsed JSON from: {parsed_json_path}")
    with open(parsed_json_path, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    
    # Filter elements by type and convert format
    target_elements = []
    all_elements_by_page = {}
    
    for element in elements:
        element_type = element["type"]
        # Convert "Picture" to "Figure" for consistency
        if element_type == "Picture":
            element_type = "Figure"
        
        page_num = element["page_number"] + page_offset  # Add offset to get actual PDF page
        
        # Store all elements for header detection
        if page_num not in all_elements_by_page:
            all_elements_by_page[page_num] = []
        all_elements_by_page[page_num].append(element)
        
        # Check if this is a target type
        if element_type in chunk_types or element["type"] in chunk_types:
            # Convert coordinates
            coords = convert_coordinates(element)
            
            # Create chunk in expected format
            chunk = {
                "page": page_num,
                "block_id": len(target_elements),
                "type": element_type,
                "text": element.get("text", ""),
                "coordinates": coords,
                "original_type": element["type"]  # Keep original for reference
            }
            target_elements.append(chunk)
    
    print(f"Found {len(target_elements)} elements of types: {chunk_types}")
    
    if not target_elements:
        print("No matching elements found!")
        return []
    
    # Group by page
    chunks_by_page = {}
    for chunk in target_elements:
        page = chunk["page"]
        if page not in chunks_by_page:
            chunks_by_page[page] = []
        chunks_by_page[page].append(chunk)
    
    print(f"Processing {len(chunks_by_page)} pages...\n")
    
    # Load PDF pages as images
    print(f"Loading PDF: {pdf_path}")
    pdf_layouts, pdf_images = lp.load_pdf(str(pdf_path), load_images=True)
    print(f"Loaded {len(pdf_images)} pages\n")
    
    # Create output directory structure: output_dir/pdf_name/clippings
    pdf_name = pdf_path.stem
    clippings_dir = output_dir / pdf_name / "clippings"
    clippings_dir.mkdir(parents=True, exist_ok=True)
    
    clipping_metadata = []
    clipping_count = 0
    
    # Process each page that has target chunks
    for page_num in sorted(chunks_by_page.keys()):
        page_idx = page_num - 1  # Convert to 0-indexed
        
        if page_idx < 0 or page_idx >= len(pdf_images):
            print(f"⚠ Page {page_num} not found in PDF (max pages: {len(pdf_images)})")
            continue
        
        page_image = pdf_images[page_idx]
        page_chunks = chunks_by_page[page_num]
        
        print(f"Page {page_num}: Processing {len(page_chunks)} chunks...")
        
        # Convert PIL Image to numpy array if needed
        if isinstance(page_image, Image.Image):
            page_array = np.array(page_image)
        else:
            page_array = page_image
        
        # Get page dimensions for coordinate scaling if needed
        img_height, img_width = page_array.shape[:2]
        
        # Process each chunk on this page
        for chunk_idx, chunk in enumerate(page_chunks):
            coords = chunk["coordinates"]
            x1, y1 = int(coords["x_1"]), int(coords["y_1"])
            x2, y2 = int(coords["x_2"]), int(coords["y_2"])
            
            # Check if coordinates need scaling (if JSON uses different DPI than PDF images)
            # Get original element to check page dimensions
            original_elements = [e for e in all_elements_by_page[page_num] 
                               if e.get("page_number") == page_num - page_offset]
            if original_elements:
                orig_element = original_elements[0]
                json_page_width = orig_element.get("page_width", img_width)
                json_page_height = orig_element.get("page_height", img_height)
                
                # Scale coordinates if page dimensions differ
                if json_page_width != img_width or json_page_height != img_height:
                    scale_x = img_width / json_page_width
                    scale_y = img_height / json_page_height
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
            
            # Try to include nearby header text blocks (for tables/figures)
            if chunk["type"] in ["Table", "Figure"]:
                header_margin = 10  # pixels above to search for headers
                all_page_elements = all_elements_by_page.get(page_num, [])
                
                # Collect potential header blocks
                header_blocks = []
                for other_element in all_page_elements:
                    if other_element["type"] in ["Text", "Section header"]:
                        other_left = other_element["left"]
                        other_top = other_element["top"]
                        other_width = other_element["width"]
                        other_height = other_element["height"]
                        other_right = other_left + other_width
                        other_bottom = other_top + other_height
                        
                        # Check if this text block is directly above (within small margin)
                        # and overlaps horizontally with the table/figure
                        if (other_bottom <= y1 + header_margin and  # Above or just above
                            other_bottom >= y1 - 5 and  # Very close (within 5px below or above)
                            not (other_right < x1 - 50 or other_left > x2 + 50)):  # Horizontally aligned
                            header_blocks.append((other_element, other_top, other_bottom, other_left, other_right))
                
                # If we found header blocks, expand coordinates to include them
                if header_blocks:
                    # Sort by y position (top to bottom)
                    header_blocks.sort(key=lambda x: x[1])
                    # Take the topmost header block(s) that are on the same row
                    top_header_y = header_blocks[0][1]
                    for header_element, h_y1, h_y2, h_x1, h_x2 in header_blocks:
                        # Include headers that are on the same row (within 5px vertically)
                        if abs(h_y1 - top_header_y) <= 5:
                            # Scale header coordinates if needed
                            if original_elements:
                                orig_element = original_elements[0]
                                json_page_width = orig_element.get("page_width", img_width)
                                json_page_height = orig_element.get("page_height", img_height)
                                if json_page_width != img_width or json_page_height != img_height:
                                    scale_x = img_width / json_page_width
                                    scale_y = img_height / json_page_height
                                    h_x1 = int(h_x1 * scale_x)
                                    h_x2 = int(h_x2 * scale_x)
                                    h_y1 = int(h_y1 * scale_y)
                            
                            y1 = min(y1, int(h_y1))
                            x1 = min(x1, int(h_x1))
                            x2 = max(x2, int(h_x2))
                            header_text = header_element['text'].strip()
                            if len(header_text) < 100:  # Only log short text (likely headers)
                                print(f"  ↳ Including header: '{header_text}'")
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            # Ensure valid crop region
            if x2 <= x1 or y2 <= y1:
                print(f"  ⚠ Skipping chunk {chunk_idx} (invalid coordinates)")
                continue
            
            # Crop the region
            try:
                # For numpy array: [y1:y2, x1:x2] (note: y comes first in numpy)
                cropped = page_array[y1:y2, x1:x2]
                
                # Convert back to PIL Image for saving
                if len(cropped.shape) == 3:
                    # RGB image
                    cropped_image = Image.fromarray(cropped)
                else:
                    # Grayscale
                    cropped_image = Image.fromarray(cropped, mode='L')
                
                # Generate output filename
                chunk_type = chunk["type"].lower()
                block_id = chunk.get("block_id", chunk_idx)
                filename = f"page_{page_num:03d}_{chunk_type}_{block_id:03d}.png"
                output_path = clippings_dir / filename
                
                # Save the clipping
                cropped_image.save(output_path, "PNG")
                
                clipping_count += 1
                print(f"  ✓ Saved: {filename} ({x2-x1}x{y2-y1} pixels)")
                
                # Store metadata
                # Store relative page number (from parsed JSON) not PDF page number
                # page_num is the PDF page (with offset), so subtract offset to get relative page
                relative_page = page_num - page_offset
                clipping_metadata.append({
                    "page": relative_page,  # Use relative page to match parsed JSON
                    "pdf_page": page_num,  # Also store PDF page for reference
                    "type": chunk["type"],
                    "block_id": block_id,
                    "filename": filename,
                    "coordinates": coords,
                    "size": {"width": x2 - x1, "height": y2 - y1}
                })
                
            except Exception as e:
                print(f"  ✗ Error cropping chunk {chunk_idx}: {e}")
                continue
    
    # Save metadata JSON
    metadata_path = clippings_dir / "clippings_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(clipping_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total clippings extracted: {clipping_count}")
    print(f"Output directory: {clippings_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Count by type
    type_counts = {}
    for meta in clipping_metadata:
        chunk_type = meta["type"]
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
    
    print(f"\nBy type:")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  {chunk_type}: {count}")
    
    return clipping_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract table and figure clippings from parsed JSON")
    parser.add_argument("--pdf", type=str, 
                       default="data/sec_pdfs/tsla-20231231-60-64.pdf",
                       help="Path to PDF file")
    parser.add_argument("--parsed", type=str, 
                       default="data/sec_pdfs/parsed/tsla-20231-60-64.json",
                       help="Path to parsed JSON file")
    parser.add_argument("--output", type=str, default="data/sec_pdfs/clippings",
                       help="Output directory for clippings")
    parser.add_argument("--types", type=str, nargs="+", default=["Table", "Picture", "Figure"],
                       help="Element types to extract (default: Table Picture Figure)")
    parser.add_argument("--page-offset", type=int, default=59,
                       help="Page offset (JSON page 1 = PDF page offset+1, default: 59 for pages 60-64)")
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf)
    parsed_path = Path(args.parsed)
    output_dir = Path(args.output)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        exit(1)
    
    if not parsed_path.exists():
        print(f"Error: Parsed JSON file not found: {parsed_path}")
        exit(1)
    
    metadata = extract_clippings_from_parsed(
        pdf_path=pdf_path,
        parsed_json_path=parsed_path,
        output_dir=output_dir,
        chunk_types=args.types,
        page_offset=args.page_offset
    )

