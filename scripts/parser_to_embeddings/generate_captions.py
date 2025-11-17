"""Generate captions/descriptions for tables and figures using vision LLM.

This script uses a vision-capable LLM (e.g., Gemini) to generate text descriptions
of tables and figures, making them searchable in the RAG system.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_gemini(api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp") -> genai.Client:
    """
    Initialize Gemini client.
    
    Args:
        api_key: Gemini API key (if None, reads from env)
        model: Model name to use
        
    Returns:
        Initialized Gemini client
    """
    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    client = genai.Client(api_key=api_key)
    logger.info(f"Initialized Gemini client with model: {model}")
    return client, model


def generate_table_caption(client: genai.Client, model_name: str, image_path: Path, context: Optional[str] = None) -> str:
    """
    Generate a caption/description for a table image.
    
    Args:
        client: Gemini client instance
        model_name: Model name to use
        image_path: Path to table image
        context: Optional context text (e.g., surrounding text from PDF)
        
    Returns:
        Generated caption/description
    """
    try:
        # Read image as bytes
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        prompt = """Analyze this table from a SEC filing document and provide:
1. A concise title/caption (1-2 sentences)
2. A structured description of the key data points and values
3. Any notable patterns or important information

Format your response as:
TITLE: [brief title]
DESCRIPTION: [detailed description of table contents, including key values and structure]
"""
        
        if context:
            prompt += f"\n\nContext from document: {context}\n"
        
        # Use the new API format with image
        from google.genai import types
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type="image/png")
            ]
        )
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error generating table caption for {image_path}: {e}")
        return f"Table image from page (error generating description: {e})"


def generate_figure_caption(client: genai.Client, model_name: str, image_path: Path, context: Optional[str] = None) -> str:
    """
    Generate a caption/description for a figure/image.
    
    Args:
        client: Gemini client instance
        model_name: Model name to use
        image_path: Path to figure image
        context: Optional context text (e.g., surrounding text from PDF)
        
    Returns:
        Generated caption/description
    """
    try:
        # Read image as bytes
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        prompt = """Analyze this figure/chart/image from a SEC filing document and provide:
1. A concise title/caption (1-2 sentences)
2. A detailed description of what the figure shows
3. Key data points, trends, or information visible in the figure

Format your response as:
TITLE: [brief title]
DESCRIPTION: [detailed description of what the figure shows, including key data points, trends, and visual elements]
"""
        
        if context:
            prompt += f"\n\nContext from document: {context}\n"
        
        # Use the new API format with image
        from google.genai import types
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type="image/png")
            ]
        )
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error generating figure caption for {image_path}: {e}")
        return f"Figure image from page (error generating description: {e})"


def process_clippings(
    clippings_metadata_path: Path,
    clippings_dir: Path,
    output_path: Path,
    client: genai.Client,
    model_name: str,
    chunks_json_path: Optional[Path] = None
) -> List[Dict]:
    """
    Process clippings and generate captions.
    
    Args:
        clippings_metadata_path: Path to clippings_metadata.json
        clippings_dir: Directory containing clipping images
        output_path: Path to save enriched metadata with captions
        model: Gemini model instance
        chunks_json_path: Optional path to chunks JSON for context
        
    Returns:
        List of enriched clipping metadata
    """
    logger.info(f"Loading clippings metadata from: {clippings_metadata_path}")
    with open(clippings_metadata_path, 'r', encoding='utf-8') as f:
        clippings = json.load(f)
    
    # Load chunks for context if available
    chunks_by_page = {}
    if chunks_json_path and chunks_json_path.exists():
        logger.info(f"Loading chunks for context from: {chunks_json_path}")
        with open(chunks_json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        for chunk in chunks:
            page = chunk.get('page', chunk.get('page_number'))
            if page not in chunks_by_page:
                chunks_by_page[page] = []
            chunks_by_page[page].append(chunk)
    
    enriched_clippings = []
    
    for i, clipping in enumerate(clippings):
        logger.info(f"Processing clipping {i+1}/{len(clippings)}: {clipping['filename']}")
        
        image_path = clippings_dir / clipping['filename']
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Get context from surrounding chunks
        context = None
        page = clipping.get('page')
        if page and page in chunks_by_page:
            # Get text chunks from the same page
            page_chunks = chunks_by_page[page]
            text_chunks = [c.get('text', '') for c in page_chunks if c.get('type') == 'Text']
            if text_chunks:
                # Use chunks before and after (if available)
                context = ' '.join(text_chunks[:3])  # First 3 text chunks for context
        
        # Generate caption based on type
        if clipping['type'] == 'Table':
            caption = generate_table_caption(client, model_name, image_path, context)
        elif clipping['type'] == 'Figure':
            caption = generate_figure_caption(client, model_name, image_path, context)
        else:
            logger.warning(f"Unknown type: {clipping['type']}, skipping")
            continue
        
        # Create enriched clipping
        enriched_clipping = clipping.copy()
        enriched_clipping['caption'] = caption
        enriched_clipping['image_path'] = str(image_path.relative_to(clippings_dir.parent.parent))
        enriched_clippings.append(enriched_clipping)
        
        logger.info(f"  ✓ Generated caption ({len(caption)} chars)")
    
    # Save enriched metadata
    logger.info(f"Saving enriched metadata to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_clippings, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully processed {len(enriched_clippings)} clippings")
    return enriched_clippings


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate captions for table and figure clippings")
    parser.add_argument(
        "--clippings-metadata",
        type=str,
        default="data/sec_pdfs/clippings/tsla-20231231-60-64/clippings/clippings_metadata.json",
        help="Path to clippings_metadata.json"
    )
    parser.add_argument(
        "--clippings-dir",
        type=str,
        default="data/sec_pdfs/clippings/tsla-20231231-60-64/clippings",
        help="Directory containing clipping images"
    )
    parser.add_argument(
        "--chunks-json",
        type=str,
        default=None,
        help="Optional path to chunks JSON for context"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sec_pdfs/clippings/tsla-20231231-60-64/clippings/enriched_metadata.json",
        help="Output path for enriched metadata"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Gemini model name (e.g., gemini-2.0-flash-exp, gemini-2.5-flash)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (if not provided, reads from GEMINI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Initialize Gemini
    client, model_name = initialize_gemini(api_key=args.api_key, model=args.model)
    
    # Process clippings
    process_clippings(
        clippings_metadata_path=Path(args.clippings_metadata),
        clippings_dir=Path(args.clippings_dir),
        output_path=Path(args.output),
        client=client,
        model_name=model_name,
        chunks_json_path=Path(args.chunks_json) if args.chunks_json else None
    )


if __name__ == "__main__":
    main()

