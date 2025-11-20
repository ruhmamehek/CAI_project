"""Process layout-parser JSON files to add item number fields."""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, environment variables will need to be set manually

logger = logging.getLogger(__name__)

# Try to import Gemini for image descriptions
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not available. Picture descriptions will be skipped.")

# Try to import pdf2image for PDF image extraction
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. PDF image clipping will be skipped. Install with: pip install pdf2image")

# Try to import PIL for image manipulation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available. PDF image clipping will be skipped. Install with: pip install Pillow")

# Pattern to match SEC filing item numbers like "Item 1.", "Item 1A.", "Item 2.", etc.
# Only matches specific items: 1, 1A, 1B, 2, 3, 4, 5, 6, 7, 7A, 8, 9, 10, 11, 12, 13, 14, 15
ITEM_PATTERN = re.compile(r'Item\s+(1[0-5]|[1-9][A-Z]?|7A)\b', re.IGNORECASE)

# Allowed types for chunking
ALLOWED_TYPES = ["Text", "List item", "Table", "Caption"]

# Types to delete
DELETE_TYPES = ["Page header", "Page footer"]

# Picture type - processed separately with Gemini descriptions
PICTURE_TYPE = "Picture"


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Project root is determined by going up from this file's location
    until we find the project root (has backend/ and frontend/ directories).
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__).resolve()
    # This file is at backend/vectordb/data_processing.py
    # Project root is 2 levels up
    project_root = current_file.parent.parent.parent
    return project_root


def extract_ticker_and_year(filename: str) -> Dict[str, Optional[str]]:
    """
    Extract ticker and year from filename.
    
    Supports formats like:
    - jnj_2023.json -> ticker: "JNJ", year: 2023
    - tsla-2024.json -> ticker: "TSLA", year: 2024
    
    Args:
        filename: Filename (with or without path)
        
    Returns:
        Dictionary with 'ticker' and 'year' keys
    """
    # Get just the filename without path and extension
    stem = Path(filename).stem
    
    # Pattern to match: TICKER_separator_YEAR or TICKER-separator-YEAR
    # Separator can be underscore, dash, or space
    pattern = r'^([A-Z]+)[_\-](\d{4})$'
    match = re.match(pattern, stem, re.IGNORECASE)
    
    if match:
        ticker = match.group(1).upper()
        year = match.group(2)
        return {'ticker': ticker, 'year': year}
    
    # Try alternative pattern: TICKER followed by year anywhere
    pattern2 = r'([A-Z]+).*?(\d{4})'
    match2 = re.search(pattern2, stem, re.IGNORECASE)
    
    if match2:
        ticker = match2.group(1).upper()
        year = match2.group(2)
        return {'ticker': ticker, 'year': year}
    
    logger.warning(f"Could not extract ticker and year from filename: {filename}")
    return {'ticker': None, 'year': None}


def generate_image_description(image_path: Path, ticker: Optional[str] = None, year: Optional[str] = None, page: Optional[int] = None) -> Optional[str]:
    """
    Generate a description for an image using Gemini API.
    
    Uses base64 encoding with data URI format for reliable transmission.
    
    Args:
        image_path: Path to the image file
        ticker: Optional ticker symbol for context
        year: Optional year for context
        page: Optional page number for context
        
    Returns:
        Generated description or None if error
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available, skipping image description")
        return None
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, skipping image description")
        return None
    
    if not image_path.exists():
        logger.warning(f"Image path does not exist: {image_path}")
        return None
    
    try:
        import base64
        
        client = genai.Client(api_key=api_key)
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        # Read image as bytes
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        logger.debug(f"Read image {image_path.name}: {len(image_data)} bytes")
        
        # Determine MIME type from extension
        suffix = image_path.suffix.lower()
        mime_type = "image/png"
        if suffix in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif suffix == '.gif':
            mime_type = "image/gif"
        elif suffix == '.webp':
            mime_type = "image/webp"
        
        # Build context string
        context_parts = []
        if ticker:
            context_parts.append(f"Company: {ticker}")
        if year:
            context_parts.append(f"Year: {year}")
        if page:
            context_parts.append(f"Page: {page}")
        context_str = ", ".join(context_parts) if context_parts else "SEC filing document"
        
        prompt = f"""Analyze this image from a {context_str} and provide a detailed description.

Please describe:
1. What type of visual content this is (chart, graph, diagram, table, photo, etc.)
2. The key information, data points, trends, or content visible in the image
3. Any text, labels, or annotations visible in the image
4. The overall purpose or message of the image

Provide a comprehensive description that would help someone understand the image content without seeing it.
"""
        
        # Use base64 encoding for reliable transmission
        # Base64 encoding ensures proper encoding/decoding and is more reliable for transmission
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        logger.debug(f"Encoded image to base64: {len(image_base64)} chars (original: {len(image_data)} bytes)")
        
        # Decode base64 back to bytes for from_bytes (ensures proper encoding)
        # This approach ensures the bytes are properly encoded/decoded
        decoded_bytes = base64.b64decode(image_base64)
        image_part = types.Part.from_bytes(data=decoded_bytes, mime_type=mime_type)
        logger.debug(f"Created Part from {len(decoded_bytes)} bytes with mime_type: {mime_type}")
        
        logger.info(f"Calling Gemini API for image description: {image_path.name} ({mime_type}, {len(image_data)} bytes)")
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                image_part
            ]
        )
        
        if not response or not response.text:
            logger.warning(f"Empty response from Gemini for {image_path.name}")
            return None
        
        description = response.text.strip()
        logger.info(f"Generated description for {image_path.name}: {description[:100]}...")
        return description
    
    except Exception as e:
        logger.error(f"Error generating image description for {image_path}: {e}", exc_info=True)
        return None


def find_pdf_file(json_filename: str, base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find the PDF file corresponding to a JSON filename.
    
    Looks for PDF in: data/pdfs/{json_filename_stem}.pdf
    
    Args:
        json_filename: JSON filename (e.g., "tsla-2023.json")
        base_dir: Not used, kept for compatibility
        
    Returns:
        Path to PDF file or None if not found
    """
    project_root = get_project_root()
    
    # Extract filename without extension
    name_without_ext = Path(json_filename).stem
    
    # Look in data/pdfs/ directory
    pdf_dir = project_root / "data" / "pdfs"
    pdf_path = pdf_dir / f"{name_without_ext}.pdf"
    
    if pdf_path.exists():
        logger.debug(f"Found PDF file: {pdf_path}")
        return pdf_path
    
    logger.warning(f"Could not find PDF file: {pdf_path}")
    return None


def clip_image_from_pdf(
    pdf_path: Path,
    picture_obj: Dict,
    output_dir: Path,
    ticker: Optional[str] = None,
    year: Optional[str] = None,
    index: int = 0
) -> Optional[Path]:
    """
    Clip an image from a PDF page using coordinates from picture_obj.
    
    Args:
        pdf_path: Path to PDF file
        picture_obj: Picture object with coordinates (left, top, width, height, page_number, page_width, page_height)
        output_dir: Directory to save clipped image
        ticker: Ticker symbol for filename
        year: Year for filename
        index: Index for unique filename
        
    Returns:
        Path to saved image file or None if clipping failed
    """
    if not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
        logger.warning("pdf2image or PIL not available. Cannot clip images from PDF.")
        return None
    
    if not pdf_path.exists():
        logger.warning(f"PDF file not found: {pdf_path}")
        return None
    
    try:
        page_number = picture_obj.get('page_number')
        if page_number is None:
            logger.warning("No page_number in picture_obj")
            return None
        
        # Extract coordinates
        left = picture_obj.get('left', 0)
        top = picture_obj.get('top', 0)
        width = picture_obj.get('width', 0)
        height = picture_obj.get('height', 0)
        
        if width <= 0 or height <= 0:
            logger.warning(f"Invalid dimensions: {width}x{height}")
            return None
        
        # Convert PDF page to image
        images = convert_from_path(str(pdf_path), first_page=page_number, last_page=page_number)
        if not images:
            logger.warning(f"Could not convert page {page_number} from PDF")
            return None
        
        page_image = images[0]
        img_width, img_height = page_image.size
        
        # Scale coordinates if JSON page dimensions differ from PDF image dimensions
        json_page_width = picture_obj.get('page_width')
        json_page_height = picture_obj.get('page_height')
        
        if json_page_width and json_page_height and (json_page_width != img_width or json_page_height != img_height):
            scale_x = img_width / json_page_width
            scale_y = img_height / json_page_height
            x1 = int(left * scale_x)
            y1 = int(top * scale_y)
            x2 = int((left + width) * scale_x)
            y2 = int((top + height) * scale_y)
        else:
            # Use coordinates as-is
            x1 = int(left)
            y1 = int(top)
            x2 = int(left + width)
            y2 = int(top + height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Ensure valid crop region
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid crop coordinates: ({x1},{y1}) to ({x2},{y2})")
            return None
        
        # Crop the region
        cropped_image = page_image.crop((x1, y1, x2, y2))
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        ticker_str = ticker or 'unknown'
        year_str = year or 'unknown'
        filename = f"{ticker_str}_{year_str}_page_{page_number}_pic_{index:03d}.png"
        output_path = output_dir / filename
        
        # Save the clipped image
        cropped_image.save(output_path, "PNG")
        
        logger.debug(f"Clipped image saved: {output_path} ({x2-x1}x{y2-y1} pixels)")
        return output_path
        
    except Exception as e:
        logger.error(f"Error clipping image from PDF {pdf_path} page {page_number}: {e}", exc_info=True)
        return None


def extract_item_number_from_section_header(text: str) -> Optional[str]:
    """
    Extract item number from section header text if it contains a valid Item header.
    
    Only extracts if the text is a section header with one of the valid Item numbers.
    
    Args:
        text: Text to search for item number
        
    Returns:
        Item number string (e.g., "1", "1A", "2") or None
    """
    if not text:
        return None
    
    # Look for Item pattern in the text
    match = ITEM_PATTERN.search(text.strip())
    if match:
        return match.group(1).upper()  # Normalize to uppercase (e.g., "1a" -> "1A")
    
    return None


def add_chunk_metadata_tag(chunk: Dict) -> None:
    """
    Add metadata tag to the beginning of chunk text.
    
    Format: [Ticker="TSLA", Year="2024", Chunk_id="TSLA_2024_chunk_90"]
    
    Args:
        chunk: Chunk dictionary with 'text', 'ticker', 'year', and 'chunk_id' fields
    """
    if 'text' not in chunk:
        return
    
    ticker = chunk.get('ticker', 'unknown')
    year = chunk.get('year', 'unknown')
    chunk_id = chunk.get('chunk_id', 'unknown')
    
    tag = f'[Ticker="{ticker}", Year="{year}", Chunk_id="{chunk_id}"]'
    
    # Prepend tag to existing text
    chunk['text'] = tag + ' ' + chunk['text']


def process_picture_objects(
    picture_objects: List[Dict],
    ticker: Optional[str] = None,
    year: Optional[str] = None,
    base_dir: Optional[Path] = None,
    json_filename: Optional[str] = None,
    enable_gemini: bool = True
) -> List[Dict]:
    """
    Process Picture objects: clip images from PDF, save locally, create chunks with descriptions.
    
    Args:
        picture_objects: List of Picture type objects
        ticker: Ticker symbol
        year: Year
        base_dir: Base directory (not used, kept for compatibility)
        json_filename: JSON filename (used to find corresponding PDF)
        enable_gemini: Whether to generate descriptions with Gemini
        
    Returns:
        List of chunk objects for pictures
    """
    picture_chunks = []
    
    # Find PDF file
    if not json_filename:
        logger.warning("No json_filename provided. Cannot clip images from PDF.")
        return picture_chunks
    
    pdf_path = find_pdf_file(json_filename)
    if not pdf_path:
        logger.warning(f"Could not find PDF file for {json_filename}. Skipping picture processing.")
        return picture_chunks
    
    # Create output directory for clipped images: data/images/{ticker}/{year}/
    project_root = get_project_root()
    ticker_str = ticker or 'unknown'
    year_str = year or 'unknown'
    output_dir = project_root / "data" / "images" / ticker_str / year_str
    
    # Process each picture object
    for idx, picture_obj in enumerate(picture_objects):
        # Clip image from PDF
        image_path = clip_image_from_pdf(
            pdf_path=pdf_path,
            picture_obj=picture_obj,
            output_dir=output_dir,
            ticker=ticker,
            year=year,
            index=idx
        )
        
        if not image_path:
            logger.warning(f"Failed to clip image for picture object {idx} on page {picture_obj.get('page_number')}. Creating chunk without image_path.")
        
        # Generate description if Gemini is available and image_path exists
        description = None
        if image_path and enable_gemini and GEMINI_AVAILABLE:
            description = generate_image_description(
                image_path,
                ticker=ticker,
                year=year,
                page=picture_obj.get('page_number')
            )
        
        # Fallback description if Gemini not available or failed
        if not description:
            page = picture_obj.get('page_number', 'unknown')
            item_num = picture_obj.get('item_number', '')
            item_str = f" (Item {item_num})" if item_num else ""
            description = f"Picture image from page {page}{item_str}"
        
        # Create chunk for picture
        chunk_obj = {
            'text': description,
            'type': 'Picture',
            'item_number': picture_obj.get('item_number')  # May be set from context
        }
        
        # Add ticker and year
        if ticker:
            chunk_obj['ticker'] = ticker
        if year:
            chunk_obj['year'] = year
        
        # Preserve metadata
        if 'page_number' in picture_obj:
            chunk_obj['page_number'] = picture_obj['page_number']
        for key in ['left', 'top', 'width', 'height', 'page_width', 'page_height']:
            if key in picture_obj:
                chunk_obj[key] = picture_obj[key]
        
        # Store image path relative to project root
        if image_path:
            project_root = get_project_root()
            img_path = Path(image_path) if not isinstance(image_path, Path) else image_path
            try:
                relative_path = img_path.relative_to(project_root)
                chunk_obj['image_path'] = str(relative_path)
            except ValueError:
                # Path is not relative to project root - store as-is
                chunk_obj['image_path'] = str(img_path)
        
        # Generate chunk_id
        ticker_str = ticker or 'unknown'
        year_str = year or 'unknown'
        chunk_obj['chunk_id'] = f"{ticker_str}_{year_str}_picture_{idx}"
        
        picture_chunks.append(chunk_obj)
    
    logger.info(f"Processed {len(picture_chunks)} Picture objects into chunks")
    return picture_chunks


def add_item_numbers(
    data: List[Dict], 
    chunk_size: int = 512, 
    tokenizer_name: str = "gpt2",
    ticker: Optional[str] = None,
    year: Optional[str] = None,
    base_dir: Optional[Path] = None,
    enable_picture_processing: bool = True,
    json_filename: Optional[str] = None
) -> List[Dict]:
    """
    Process layout-parser data to:
    1. Filter to only include ["Text", "List item", "Table"] types
    2. Delete ["Page header", "Page footer"] types
    3. Process Picture objects separately with Gemini descriptions
    4. Detect section headers with Item numbers and assign to subsequent chunks
    5. Aggregate text blocks into chunks of max size 512 tokens with 15% overlap
    
    Args:
        data: List of objects with 'text' and 'type' fields
        chunk_size: Target chunk size in tokens (default: 512)
        tokenizer_name: Tokenizer model name for counting tokens
        ticker: Ticker symbol to add to each chunk
        year: Year to add to each chunk
        base_dir: Base directory for finding images (for Picture processing)
        enable_picture_processing: Whether to process Picture objects with Gemini
        
    Returns:
        List of chunk objects with added 'item_number', 'ticker', and 'year' fields
        Chunks have 15% overlap to ensure context continuity
    """
    # Initialize tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except ImportError:
        logger.warning(f"transformers library not available. Using simple word count approximation.")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}. Using simple word count approximation.")
    
    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        if tokenizer:
            return len(tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=10000))
        else:
            # Fallback: approximate tokens as words * 1.3 (rough estimate)
            return int(len(text.split()) * 1.3)
    
    # Step 1: Separate Picture objects for special processing
    picture_objects = []
    filtered_objects = []
    current_item_for_pictures = None
    
    for obj in data:
        obj_type = obj.get('type', '')
        text = obj.get('text', '')
        
        # Delete page headers and footers
        if obj_type in DELETE_TYPES:
            continue
        
        # Handle Picture objects separately
        if obj_type == PICTURE_TYPE:
            if enable_picture_processing:
                picture_obj = obj.copy()
                # Assign current item number if available
                picture_obj['item_number'] = current_item_for_pictures
                picture_objects.append(picture_obj)
            continue
        
        # Only keep allowed types
        if obj_type not in ALLOWED_TYPES:
            # Check if it's a section header - we need to process these separately
            if obj_type == "Section header":
                # Check if it contains a valid Item number
                item_num = extract_item_number_from_section_header(text)
                if item_num:
                    # Update current item for pictures
                    current_item_for_pictures = item_num
                    # Add as a special marker object
                    marker_obj = obj.copy()
                    marker_obj['_is_item_header'] = True
                    marker_obj['_item_number'] = item_num
                    filtered_objects.append(marker_obj)
            continue
        
        # Add allowed types
        if text.strip():  # Only add if there's actual text
            filtered_objects.append(obj.copy())
    
    # Step 2: Process objects and create chunks with item numbers
    chunks = []
    current_item = None  # Track item number for regular chunks
    current_chunk_items = []  # List of (text, obj) tuples to track objects with their texts
    current_chunk_token_count = 0
    overlap_text = None  # Store last 10% of words from previous chunk
    
    def finalize_chunk(include_overlap: bool = True):
        """
        Finalize current chunk and add to chunks list.
        
        If include_overlap is True, extracts the last 10% of words for the next chunk.
        """
        nonlocal current_chunk_items, current_chunk_token_count, overlap_text
        
        if not current_chunk_items:
            return
        
        # Create the full chunk text from all items
        chunk_text = ' '.join(text for text, _ in current_chunk_items)
        
        # Extract last 10% of words for overlap if requested
        if include_overlap:
            words = chunk_text.split()
            if len(words) >= 10:  # Only create overlap if chunk has at least 10 words
                overlap_word_count = max(1, len(words) // 10)  # 10% of words, at least 1
                overlap_words = words[-overlap_word_count:]
                overlap_text = ' '.join(overlap_words)
                # Calculate tokens for overlap
                overlap_token_count = count_tokens(overlap_text)
            else:
                overlap_text = None
                overlap_token_count = 0
        else:
            overlap_text = None
            overlap_token_count = 0
        
        chunk_obj = {
            'text': chunk_text,
            'type': 'Text',  # Default type for regular chunks
            'item_number': current_item
        }
        
        # Add ticker and year if provided
        if ticker:
            chunk_obj['ticker'] = ticker
        if year:
            chunk_obj['year'] = year
        
        # Preserve metadata from first object in chunk
        if current_chunk_items:
            first_obj = current_chunk_items[0][1]
            # Preserve original type if available (may be Table, Caption, List item, etc.)
            if 'type' in first_obj:
                original_type = first_obj.get('type')
                # Only override if it's one of our allowed types
                if original_type in ALLOWED_TYPES:
                    chunk_obj['type'] = original_type
            
            if 'page_number' in first_obj:
                chunk_obj['page_number'] = first_obj.get('page_number')
            # Preserve other relevant metadata if needed
            for key in ['left', 'top', 'width', 'height', 'page_width', 'page_height']:
                if key in first_obj:
                    chunk_obj[key] = first_obj.get(key)
        
        chunks.append(chunk_obj)
        
        # Prepare for next chunk: start with overlap text if any
        if overlap_text:
            # Create a pseudo-object for the overlap text to start the next chunk
            # Use the last object's metadata as a template
            last_obj = current_chunk_items[-1][1] if current_chunk_items else {}
            overlap_obj = last_obj.copy()
            overlap_obj['text'] = overlap_text
            overlap_obj['type'] = 'Text'
            # Mark this as overlap so we know it came from previous chunk
            overlap_obj['_is_overlap'] = True
            current_chunk_items = [(overlap_text, overlap_obj)]
            current_chunk_token_count = overlap_token_count
        else:
            # No overlap, start fresh
            current_chunk_items = []
            current_chunk_token_count = 0
    
    for obj in filtered_objects:
        # Check if this is an item header marker
        if obj.get('_is_item_header'):
            # First, finalize current chunk if any (no overlap when hitting item header)
            finalize_chunk(include_overlap=False)
            
            # Update current item number (for both regular chunks and future Picture objects)
            current_item = obj.get('_item_number')
            current_item_for_pictures = current_item  # Keep in sync
            logger.debug(f"Found item header: Item {current_item}")
            continue
        
        # Regular text block - add to current chunk
        text = obj.get('text', '').strip()
        if not text:
            continue
        
        text_tokens = count_tokens(text)
        
        # Check if adding this text would exceed chunk size
        # If so, finalize current chunk first (with overlap), then add the new text
        if current_chunk_token_count + text_tokens > chunk_size and len(current_chunk_items) > 0:
            # Check if current chunk has at least one real block (not just overlap from previous)
            has_real_blocks = any(not item[1].get('_is_overlap', False) for item in current_chunk_items)
            if has_real_blocks:
                finalize_chunk(include_overlap=True)
        
        # Add text and object to current chunk
        current_chunk_items.append((text, obj))
        current_chunk_token_count += text_tokens
    
    # Finalize last chunk if any (no overlap for final chunk)
    finalize_chunk(include_overlap=False)
    
    # Step 3: Process Picture objects separately
    if enable_picture_processing and picture_objects:
        picture_chunks = process_picture_objects(
            picture_objects,
            ticker=ticker,
            year=year,
            base_dir=base_dir,
            json_filename=json_filename,
            enable_gemini=True
        )
        # Add picture chunks to main chunks list
        chunks.extend(picture_chunks)
    
    logger.info(f"Created {len(chunks)} chunks from {len(filtered_objects)} filtered objects")
    if enable_picture_processing and picture_objects:
        logger.info(f"Added {len(picture_objects)} Picture chunks with Gemini descriptions")
    
    return chunks


def process_json_file(
    input_path: Path, 
    output_path: Optional[Path] = None,
    upload_to_chromadb_flag: bool = False,
    collection_name: str = "isaac_test_filings",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 100
) -> Dict:
    """
    Process a single JSON file to add item numbers.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (defaults to data/processed/{filename})
        
    Returns:
        Dictionary with processing statistics
    """
    if output_path is None:
        # Default to processed folder
        processed_dir = input_path.parent.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / input_path.name
    
    logger.info(f"Processing {input_path}")
    
    # Load JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    
    if not data:
        logger.warning(f"File {input_path} is empty")
        return {'total_objects': 0, 'objects_with_item_number': 0, 'unique_item_numbers': []}
    
    # Check if file is already processed (has chunks but no 'type' field)
    sample_obj = data[0]
    if 'type' not in sample_obj and 'item_number' in sample_obj:
        logger.warning(
            f"File {input_path} appears to already be processed (has 'item_number' field but no 'type' field). "
            f"Skipping. The original file structure with 'type' fields is required for processing."
        )
        return {'total_objects': len(data), 'objects_with_item_number': 0, 'unique_item_numbers': []}
    
    # Extract ticker and year from filename
    file_metadata = extract_ticker_and_year(input_path.name)
    ticker = file_metadata.get('ticker')
    year = file_metadata.get('year')
    
    if ticker and year:
        logger.info(f"Extracted ticker: {ticker}, year: {year} from filename")
    else:
        logger.warning(f"Could not extract ticker and year from filename: {input_path.name}")
    
    # Add item numbers (with Picture processing)
    # Use input_path parent as base_dir for image lookup
    base_dir = input_path.parent
    json_filename = input_path.name  # Get filename for PDF lookup
    processed_data = add_item_numbers(
        data, 
        ticker=ticker, 
        year=year,
        base_dir=base_dir,
        enable_picture_processing=True,
        json_filename=json_filename
    )
    
    # Count items with item numbers
    items_with_numbers = sum(1 for obj in processed_data if obj.get('item_number') is not None)
    unique_items = set(obj.get('item_number') for obj in processed_data if obj.get('item_number') is not None)
    
    # Add chunk_id to each chunk if not present
    for idx, chunk in enumerate(processed_data):
        if 'chunk_id' not in chunk:
            ticker_str = chunk.get('ticker', 'unknown')
            year_str = chunk.get('year', 'unknown')
            chunk['chunk_id'] = f"{ticker_str}_{year_str}_chunk_{idx}"
    
    # Add metadata tag to the beginning of each chunk's text
    for chunk in processed_data:
        add_chunk_metadata_tag(chunk)
    
    # Save processed JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    stats = {
        'total_objects': len(processed_data),
        'objects_with_item_number': items_with_numbers,
        'unique_item_numbers': sorted(unique_items) if unique_items else [],
        'output_path': str(output_path)
    }
    
    logger.info(f"Processed {len(processed_data)} objects, {items_with_numbers} have item numbers")
    logger.info(f"Found item numbers: {stats['unique_item_numbers']}")
    logger.info(f"Saved processed data to {output_path}")
    
    # Upload to ChromaDB if requested
    if upload_to_chromadb_flag:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from data_upload import upload_to_chromadb
            logger.info(f"Uploading {len(processed_data)} chunks to ChromaDB collection '{collection_name}'...")
            upload_to_chromadb(
                chunks=processed_data,
                collection_name=collection_name,
                embedding_model=embedding_model,
                batch_size=batch_size,
                replace=False  # Will replace when processing all files together
            )
            logger.info(f"Successfully uploaded chunks to ChromaDB")
        except ImportError as e:
            logger.error(f"Could not import upload_to_chromadb: {e}. Make sure data_upload.py is available.")
        except Exception as e:
            logger.error(f"Error uploading to ChromaDB: {e}")
            raise
    
    return stats


def process_all_json_files(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    file_pattern: str = "*.json",
    upload_to_chromadb_flag: bool = False,
    collection_name: str = "sec_filings",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 100
):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Output directory (defaults to data/processed/)
        file_pattern: Glob pattern for JSON files
    """
    if output_dir is None:
        # Default to processed folder relative to input_dir
        output_dir = input_dir.parent.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob(file_pattern))
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return
    
    logger.info(f"Processing {len(json_files)} JSON files from {input_dir}")
    
    # If uploading to ChromaDB, collect all chunks first, then upload together
    all_chunks = [] if upload_to_chromadb_flag else None
    
    all_stats = []
    for json_file in json_files:
        try:
            output_path = output_dir / json_file.name
            # Don't upload individual files, we'll upload all at once
            stats = process_json_file(
                json_file, 
                output_path,
                upload_to_chromadb_flag=False
            )
            stats['filename'] = json_file.name
            all_stats.append(stats)
            
            # Collect chunks for batch upload
            if upload_to_chromadb_flag and stats.get('output_path'):
                try:
                    with open(stats['output_path'], 'r', encoding='utf-8') as f:
                        file_chunks = json.load(f)
                        all_chunks.extend(file_chunks)
                except Exception as e:
                    logger.warning(f"Could not load chunks from {stats['output_path']}: {e}")
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    
    # Upload all chunks to ChromaDB at once (replacing existing data)
    if upload_to_chromadb_flag and all_chunks:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from data_upload import upload_to_chromadb
            logger.info(f"Uploading {len(all_chunks)} total chunks to ChromaDB collection '{collection_name}' (replacing existing data)...")
            upload_to_chromadb(
                chunks=all_chunks,
                collection_name=collection_name,
                embedding_model=embedding_model,
                batch_size=batch_size,
                replace=True  # Replace all existing data
            )
            logger.info(f"Successfully uploaded {len(all_chunks)} chunks to ChromaDB")
        except ImportError as e:
            logger.error(f"Could not import upload_to_chromadb: {e}. Make sure data_upload.py is available.")
        except Exception as e:
            logger.error(f"Error uploading to ChromaDB: {e}")
            raise
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    for stats in all_stats:
        print(f"\n{stats['filename']}:")
        print(f"  Total objects: {stats['total_objects']}")
        print(f"  Objects with item number: {stats['objects_with_item_number']}")
        print(f"  Item numbers found: {', '.join(stats['unique_item_numbers'])}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add item numbers to layout-parser JSON files")
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory containing JSON files to process'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output directory (defaults to input directory)'
    )
    parser.add_argument(
        '--pattern',
        default='*.json',
        help='File pattern for JSON files (default: *.json)'
    )
    parser.add_argument(
        '--file',
        type=Path,
        default=None,
        help='Process a single file instead of directory'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--upload-chromadb',
        action='store_true',
        help='Upload processed chunks to ChromaDB'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='sec_filings',
        help='ChromaDB collection name (default: sec_filings)'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='BAAI/bge-base-en-v1.5',
        help='Embedding model name (default: BAAI/bge-base-en-v1.5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for ChromaDB uploads (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.file:
        # Process single file
        process_json_file(
            args.file, 
            args.output,
            upload_to_chromadb_flag=args.upload_chromadb,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size
        )
    else:
        # Process directory
        process_all_json_files(
            args.input_dir, 
            args.output, 
            args.pattern,
            upload_to_chromadb_flag=args.upload_chromadb,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()

