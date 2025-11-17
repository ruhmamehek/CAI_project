"""Utility functions for RAG service."""

import re
from pathlib import Path
from typing import Optional


def get_image_mime_type(suffix: str) -> Optional[str]:
    """
    Get MIME type for an image file based on its extension.
    
    Args:
        suffix: File extension (e.g., '.png', '.jpg')
        
    Returns:
        MIME type string or None if not an image
    """
    suffix_lower = suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(suffix_lower)


def clean_image_text(text: str) -> Optional[str]:
    """
    Remove "Image: ..." lines from text since images are displayed separately.
    
    Args:
        text: Text that may contain "Image: ..." lines
        
    Returns:
        Cleaned text with Image lines removed, or None if empty after cleaning
    """
    if not text:
        return None
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not re.match(r'^\s*Image:\s*', line, flags=re.IGNORECASE):
            cleaned_lines.append(line)
    
    cleaned_text = '\n'.join(cleaned_lines).strip()
    return cleaned_text if cleaned_text else None


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent.parent


def is_path_safe(path: Path, root: Path) -> bool:
    """
    Check if a path is within the project root (security check).
    
    Args:
        path: Path to check
        root: Root directory to ensure path is within
        
    Returns:
        True if path is safe (within root), False otherwise
    """
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False

