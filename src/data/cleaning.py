"""Data cleaning utilities for SEC filings and other financial documents."""

import re
import html
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class SECFilingCleaner:
    """Clean and preprocess SEC filing text."""
    
    def __init__(
        self,
        min_chunk_length: int = 50,
        remove_xbrl: bool = True,
        remove_metadata: bool = True,
        remove_boilerplate: bool = True,
        min_alphanumeric_ratio: float = 0.3,
        max_metadata_matches: int = 2,
        max_url_count: int = 3,
        xbrl_line_length_threshold: int = 200
    ):
        """
        Args:
            min_chunk_length: Minimum character length for chunks to keep
            remove_xbrl: Whether to remove XBRL/XML structured data
            remove_metadata: Whether to remove filing metadata sections
            remove_boilerplate: Whether to remove common boilerplate text
            min_alphanumeric_ratio: Minimum ratio of letters in chunk (0.0-1.0)
            max_metadata_matches: Max metadata patterns before filtering chunk
            max_url_count: Max URLs before filtering chunk
            xbrl_line_length_threshold: Line length threshold for XBRL detection
        """
        self.min_chunk_length = min_chunk_length
        self.remove_xbrl = remove_xbrl
        self.remove_metadata = remove_metadata
        self.remove_boilerplate = remove_boilerplate
        self.min_alphanumeric_ratio = min_alphanumeric_ratio
        self.max_metadata_matches = max_metadata_matches
        self.max_url_count = max_url_count
        self.xbrl_line_length_threshold = xbrl_line_length_threshold
        
        # Patterns hardcoded because SEC filings follow standardized formats (SEC EDGAR + XBRL taxonomy)
        self.xbrl_patterns = [
            r'<link:.*?>',
            r'<xbrl:.*?>',
            r'<xbrli:.*?>',
            r'<link:calculationLink.*?</link:calculationLink>',
            r'<link:presentationLink.*?</link:presentationLink>',
            r'<link:definitionLink.*?</link:definitionLink>',
            r'<link:labelLink.*?</link:labelLink>',
            r'<link:referenceLink.*?</link:referenceLink>',
            r'<link:roleRef.*?/>',
            r'<link:arcroleRef.*?/>',
            r'<link:loc.*?/>',
            r'<link:calculationArc.*?/>',
            r'<link:presentationArc.*?/>',
            r'<link:definitionArc.*?/>',
            r'<link:labelArc.*?/>',
            r'<link:referenceArc.*?/>',
        ]
        
        self.metadata_patterns = [
            r'ACCESSION NUMBER:\s*\d+',
            r'CONFORMED SUBMISSION TYPE:\s*[\w-]+',
            r'PUBLIC DOCUMENT COUNT:\s*\d+',
            r'CONFORMED PERIOD OF REPORT:\s*\d+',
            r'FILED AS OF DATE:\s*\d+',
            r'DATE AS OF CHANGE:\s*\d+',
            r'COMPANY CONFORMED NAME:.*',
            r'CENTRAL INDEX KEY:\s*\d+',
            r'STANDARD INDUSTRIAL CLASSIFICATION:.*',
            r'IRS NUMBER:\s*[\d-]+',
            r'STATE OF INCORPORATION:\s*[A-Z]{2}',
            r'FISCAL YEAR END:\s*\d{4}',
            r'FORM TYPE:\s*[\w-]+',
            r'SEC ACT:\s*[\d\w\s]+',
            r'SEC FILE NUMBER:\s*[\d-]+',
            r'FILM NUMBER:\s*\d+',
            r'BUSINESS ADDRESS:.*',
            r'MAIL ADDRESS:.*',
            r'FORMER COMPANY:.*',
            r'FORMER CONFORMED NAME:.*',
            r'DATE OF NAME CHANGE:\s*\d+',
        ]
        
        self.boilerplate_patterns = [
            r'\(Mark One\)',
            r'Indicate by check mark.*?\.',
            r'See the definitions of.*?in Rule.*?of the Exchange Act\.',
            r'Commission File Number:\s*[\d-]+',
            r'DOCUMENTS INCORPORATED BY REFERENCE',
            r'TABLE OF CONTENTS',
            r'Page\s+\d+',
            r'Part [IVX]+',
            r'Item \d+[A-Z]?\.',
            r'This Annual Report on Form.*?contains forward-looking statements',
            r'Forward-looking statements provide current expectations',
            r'Unless otherwise stated, all information presented herein',
            r'The information contained on the websites.*?is not incorporated by reference',
        ]
        
    def decode_html_entities(self, text: str) -> str:
        """Decode HTML entities to readable characters."""
        return html.unescape(text)
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML/XML tags."""
        text = re.sub(r'<[^>]+>', '', text)
        return text
    
    def remove_xbrl_content(self, text: str) -> str:
        """Remove XBRL/XML structured data sections."""
        if not self.remove_xbrl:
            return text
        
        for pattern in self.xbrl_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        text = re.sub(r'<[^>]+\s+[^>]+>', '', text)
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if len(line) > self.xbrl_line_length_threshold and (
                'http://' in line or 
                'https://' in line or
                'xbrl' in line.lower() or
                'us-gaap' in line.lower() or
                re.search(r'[a-z]+:[A-Z][a-zA-Z]+', line)
            ):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_metadata_sections(self, text: str) -> str:
        """Remove SEC filing metadata sections."""
        if not self.remove_metadata:
            return text
        
        for pattern in self.metadata_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_stripped = line.strip()
            if any(re.search(pattern, line_stripped, re.IGNORECASE) 
                   for pattern in self.metadata_patterns):
                continue
            if re.match(r'^[\d\s\-:]+$', line_stripped) and len(line_stripped) > 10:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_boilerplate_text(self, text: str) -> str:
        """Remove common SEC form boilerplate."""
        if not self.remove_boilerplate:
            return text
        
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces/newlines to single space)."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def extract_main_content(self, text: str) -> str:
        """
        Extract main document content, skipping headers/footers.
        
        Uses heuristics to find content boundaries. If markers aren't found,
        returns the full text (fail-safe).
        """
        lines = text.split('\n')
        
        start_markers = [
            'PART I',
            'Item 1.',
            'Business',
            'Company Background',
            'The Company',
            'Management\'s Discussion',
        ]
        
        start_idx = 0
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            if any(marker.upper() in line_upper for marker in start_markers):
                start_idx = i
                break
        
        end_markers = [
            'SIGNATURES',
            'EXHIBIT',
            'CERTIFICATION',
            'Part IV',
            'Item 15.',
        ]
        
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            line_upper = lines[i].upper().strip()
            if any(marker.upper() in line_upper for marker in end_markers):
                end_idx = i
                break
        
        if start_idx < end_idx and start_idx > 0:
            content_lines = lines[start_idx:end_idx]
            return '\n'.join(content_lines)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        text = self.decode_html_entities(text)
        text = self.remove_html_tags(text)
        text = self.remove_xbrl_content(text)
        text = self.remove_metadata_sections(text)
        text = self.extract_main_content(text)
        text = self.remove_boilerplate_text(text)
        text = self.normalize_whitespace(text)
        return text
    
    def filter_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Filter out low-quality chunks.
        
        Safeguards are in place to preserve important financial data even if
        it has low alphanumeric ratio (e.g., financial tables with context).
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            Filtered list of chunks
        """
        filtered = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            
            if len(text.strip()) < self.min_chunk_length:
                continue
            
            alphanumeric_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
            
            has_financial_indicators = bool(
                re.search(r'\$[\d,]+', text) or
                re.search(r'\d+%', text) or
                re.search(r'\d{4}.*\d{4}.*\d{4}', text) or
                re.search(r'(million|billion|thousand)', text, re.IGNORECASE) or
                re.search(r'(revenue|sales|income|profit|loss|margin)', text, re.IGNORECASE)
            )
            
            if alphanumeric_ratio < self.min_alphanumeric_ratio:
                if not has_financial_indicators:
                    continue
                if re.search(r'us-gaap:|xbrl:|http://fasb\.org', text, re.IGNORECASE):
                    continue
            
            metadata_matches = sum(
                1 for pattern in self.metadata_patterns 
                if re.search(pattern, text, re.IGNORECASE)
            )
            if metadata_matches > self.max_metadata_matches:
                continue
            
            url_count = len(re.findall(r'https?://', text))
            if url_count > self.max_url_count:
                if alphanumeric_ratio < 0.5:
                    continue
            
            filtered.append(chunk)
        
        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} chunks "
                   f"({len(chunks) - len(filtered)} removed)")
        
        return filtered

