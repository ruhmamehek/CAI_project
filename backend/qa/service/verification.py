"""Verification module for RAG responses."""

import logging
import re
import math
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from .models import Chunk, Source

logger = logging.getLogger(__name__)


class FinancialVerifier:
    """Code-based verifier for financial numbers in text."""
    
    def __init__(self):
        """
        Initialize financial verifier with regex patterns.
        
        Includes noise patterns to filter out dates, years, and other
        non-financial numbers before extraction.
        """
        # 1. NOISE PATTERNS: Patterns to IGNORE (Mask out) before extraction
        self.noise_patterns = [
            # Years (1900-2030) standing alone (not preceded by $)
            # Expanded to catch more variations
            re.compile(r'(?<!\$)\b(?:19|20)\d{2}\b'),
            
            # Dates with month names (Dec 31, 2022, December 31, 2022, Dec. 31, 2022)
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}', re.IGNORECASE),
            
            # Dates with ordinal numbers (December 31st, 2022, 31st of December 2022)
            re.compile(r'\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}', re.IGNORECASE),
            re.compile(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th),?\s+\d{4}', re.IGNORECASE),
            
            # Date formats: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, MM-DD-YYYY
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),  # YYYY-MM-DD format
            
            # Quarter references (Q1 2023, Q1'23, 1Q23)
            re.compile(r'\bQ[1-4]\s*\d{2,4}\b', re.IGNORECASE),
            re.compile(r'\b\d{1}[Qq]\s*\d{2,4}\b', re.IGNORECASE),
            
            # Fiscal year references (FY 2023, FY2023, fiscal year 2023)
            re.compile(r'\b(?:FY|F\.Y\.)\s*\d{2,4}\b', re.IGNORECASE),
            re.compile(r'\bfiscal\s+year\s+\d{2,4}\b', re.IGNORECASE),
            
            # Time references (2023, 2022-2023, 2023 vs 2022)
            re.compile(r'\b\d{4}\s*[-–—]\s*\d{4}\b'),  # Year ranges with various dashes
            re.compile(r'\b\d{4}\s+(?:vs|versus|vs\.|compared to)\s+\d{4}\b', re.IGNORECASE),
            
            # SEC Headers (Item 1A, Note 4, Part II, Table 1, Figure 2)
            re.compile(r'\b(?:Item|Note|Part|Form|Table|Figure|Fig\.|Tab\.)\s+\d+[A-Z]?\b', re.IGNORECASE),
            re.compile(r'10-K|10-Q|8-K|Form\s+10', re.IGNORECASE),  # Filing types
            
            # Page numbers and references (Page 5, p. 10, pp. 20-30)
            re.compile(r'\b(?:Page|p\.|pp\.)\s+\d+(?:[-–—]\d+)?\b', re.IGNORECASE),
            
            # Phone numbers / CIK / Zip Codes
            re.compile(r'\b\d{5}(?:-\d{4})?\b'),  # Zip codes (but only if not preceded by $)
            re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),  # Phone numbers
            re.compile(r'CIK[:\s]+\d+', re.IGNORECASE),  # CIK numbers
            
            # Common date phrases (end of year, beginning of year, etc.)
            re.compile(r'\b(?:as of|at the end of|beginning of|as at)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
        ]
        
        # Date-related context words that indicate a number might be a date
        self.date_context_words = {
            'month', 'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'day', 'week', 'quarter', 'year', 'fiscal', 'period',
            'ended', 'ending', 'beginning', 'start', 'date', 'dated',
            'filed', 'filing', 'report', 'quarterly', 'annual'
        }
        
        # 2. VALUE PATTERN: The actual financial extraction
        # Balanced approach: captures financial numbers but filters noise in post-processing
        self.number_pattern = re.compile(
            r'('
            # Case A: Currency ($1,000 or $ 1000 or (1,000) for negative)
            # This will match: $1,234.56, ($1,234), $1.5M, ($100 million)
            r'\(?\-?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:million|billion|trillion|M|B|T)?\)?'
            r'|'
            # Case A2: Currency with $ and no commas ($100, $1234.56)
            r'\$\s?\d+\.?\d*\s?(?:million|billion|trillion|M|B|T)?'
            r'|'
            # Case B: Explicit Scale (10.5 million, 15B, 1.2M)
            r'\d+\.?\d*\s?(?:million|billion|trillion|M|B|T)'
            r'|'
            # Case C: Percentages (5.5%, 500 bps) - CRITICAL for Audit
            r'\d+(?:\.\d+)?\s?%|\d+(?:\.\d+)?\s?bps'
            r')',
            re.IGNORECASE
        )
    
    def _clean_noise(self, text: str) -> str:
        """
        Removes dates, years, and citations to prevent false positives.
        
        Uses multiple passes and context-aware filtering.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text with noise patterns replaced by whitespace
        """
        cleaned_text = text
        
        # First pass: Remove known date patterns
        for pattern in self.noise_patterns:
            # Replace noise with whitespace to preserve word boundaries
            cleaned_text = pattern.sub(' ', cleaned_text)
        
        # Second pass: Remove standalone years (1900-2030) that might have been missed
        # But preserve years that are part of financial values
        # Look for years that are NOT preceded by $ or followed by financial terms
        year_pattern = re.compile(
            r'(?<!\$)(?<!\d)\b(?:19|20)\d{2}\b(?!\s*(?:million|billion|trillion|M|B|T|%|percent))',
            re.IGNORECASE
        )
        cleaned_text = year_pattern.sub(' ', cleaned_text)
        
        return cleaned_text
    
    def _is_likely_date_context(self, text: str, match_start: int, match_end: int, window: int = 30) -> bool:
        """
        Check if a number appears in date-related context.
        
        More lenient - only flags very clear date contexts.
        
        Args:
            text: Full text
            match_start: Start position of matched number
            match_end: End position of matched number
            window: Number of characters before/after to check (reduced for precision)
            
        Returns:
            True if number appears in very clear date context
        """
        # Extract context around the match
        context_start = max(0, match_start - window)
        context_end = min(len(text), match_end + window)
        context = text[context_start:context_end].lower()
        
        # Check for very specific date-related phrases near the number
        # Only flag if we see clear date phrases
        strong_date_indicators = [
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}',  # Month + day
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Full date pattern
            r'\bfiscal\s+year',  # Fiscal year
            r'\bfiled\s+on',  # Filed on
            r'\bended\s+(?:december|january)',  # Ended December
            r'\b(as of|at the end of|beginning of)\s+\d{1,2}',  # As of date
        ]
        
        for pattern in strong_date_indicators:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        
        # Check for date-related words immediately adjacent (within 10 chars)
        immediate_context_start = max(0, match_start - 10)
        immediate_context_end = min(len(text), match_end + 10)
        immediate_context = text[immediate_context_start:immediate_context_end].lower()
        
        immediate_date_words = {'january', 'february', 'march', 'april', 'may', 'june',
                               'july', 'august', 'september', 'october', 'november', 'december',
                               'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}
        
        immediate_words = set(re.findall(r'\b\w+\b', immediate_context))
        if immediate_date_words & immediate_words:
            return True
        
        return False
    
    def _is_likely_date_number(self, value: float, match_str: str) -> bool:
        """
        Check if an extracted number is likely a date component.
        
        More lenient - only filters obvious dates without financial context.
        
        Args:
            value: Normalized numeric value
            match_str: Original matched string
            
        Returns:
            True if number looks like a date component
        """
        # Only filter out years 1900-2030 if they have NO financial context at all
        # This is more lenient - only filter if completely without financial markers
        if 1900 <= value <= 2030:
            # Only filter if it has NO financial context indicators
            has_currency = '$' in match_str
            has_scale = any(scale in match_str.lower() for scale in ['million', 'billion', 'trillion', 'm', 'b', 't'])
            has_percentage = '%' in match_str or 'bps' in match_str.lower()
            has_negative = '(' in match_str and ')' in match_str
            has_comma = ',' in match_str  # Likely part of larger number
            
            # Only filter if it has absolutely no financial indicators
            if not (has_currency or has_scale or has_percentage or has_negative or has_comma):
                return True  # Likely a year
        
        # Don't filter days of month - they're often part of financial dates
        # and the context checking will handle it
        
        return False
    
    def _normalize_financial_value(self, raw_str: str) -> Optional[float]:
        """
        Converts financial strings into comparable floats.
        
        Handles:
        - "$ (10,200.50)" -> -10200.50
        - "$1.5 million" -> 1500000.0
        - "15B" -> 15000000000.0
        - "1.2M" -> 1200000.0
        - "5.5%" -> 0.055
        - "100 bps" -> 0.01
        
        Args:
            raw_str: Raw string containing number
            
        Returns:
            Normalized float value or None if parsing fails
        """
        clean_str = raw_str.strip().lower()
        
        # Handle Percentages
        is_percentage = False
        is_bps = False
        if '%' in clean_str:
            is_percentage = True
            clean_str = clean_str.replace('%', '')
            multiplier = 0.01  # 5% -> 0.05
        elif 'bps' in clean_str:
            is_bps = True
            clean_str = clean_str.replace('bps', '')
            multiplier = 0.0001  # 100 bps -> 0.01
        else:
            multiplier = 1.0
        
        # Check for parentheses indicating negative (Common in SEC filings)
        is_negative = False
        if '(' in clean_str and ')' in clean_str:
            is_negative = True
            # Remove parentheses
            clean_str = clean_str.replace('(', '').replace(')', '')
        
        # Extract scale multiplier (million, billion, etc.)
        scale_patterns = {
            'trillion': 1e12,
            't': 1e12,
            'billion': 1e9,
            'b': 1e9,
            'million': 1e6,
            'm': 1e6,
        }
        
        for scale, mult in scale_patterns.items():
            if scale in clean_str:
                multiplier *= mult  # Multiply the existing multiplier
                # Remove scale text
                clean_str = re.sub(rf'\s?{scale}', '', clean_str, flags=re.IGNORECASE)
                break
        
        # Remove currency symbols and other non-numeric chars except dots and negative signs
        numeric_str = re.sub(r'[^\d\.\-]', '', clean_str)
        
        if not numeric_str or numeric_str == '.' or numeric_str == '-':
            return None
        
        try:
            val = float(numeric_str)
            
            # Apply negative logic (but not for percentages/bps - they're relative)
            if not is_percentage and not is_bps:
                if is_negative or (val > 0 and '-' in raw_str.lower()):
                    val = -abs(val)
            
            # Apply multiplier
            val = val * multiplier
            
            return val
        except ValueError:
            return None
    
    def extract_values(self, text: str) -> Set[float]:
        """
        Extracts all financial numbers from text as a set of floats.
        
        First cleans noise (dates, years, etc.) to prevent false positives,
        then extracts financial numbers and filters out date-like numbers.
        
        Args:
            text: Text to extract numbers from
            
        Returns:
            Set of normalized float values
        """
        # 1. Pre-process: Remove dates and headers
        sanitized_text = self._clean_noise(text)
        
        # 2. Extract financial numbers with position information
        values = set()
        
        # Find all matches with their positions
        for match_obj in self.number_pattern.finditer(sanitized_text):
            match_str = match_obj.group(0)
            match_start = match_obj.start()
            match_end = match_obj.end()
            
            # If match is tuple (from regex groups), take first element
            if isinstance(match_str, tuple):
                match_str = match_str[0]
            
            # Normalize the value first
            norm = self._normalize_financial_value(match_str)
            if norm is None:
                continue
            
            # Post-extraction filtering: Check if normalized value looks like a date
            # Only filter if it's very clearly a date (e.g., years 1900-2030 without financial context)
            if self._is_likely_date_number(norm, match_str):
                # Double-check context before filtering
                if self._is_likely_date_context(sanitized_text, match_start, match_end):
                    continue  # Skip this match - clearly a date
            
            values.add(norm)
        
        return values
    
    def verify(self, generated_text: str, source_text: str) -> Dict[str, Any]:
        """
        Verifies that every number in the generated text exists in the source.
        
        Args:
            generated_text: Text from generated response
            source_text: Text from source context
            
        Returns:
            Dictionary with verification results
        """
        gen_values = self.extract_values(generated_text)
        src_values = self.extract_values(source_text)
        
        if not gen_values:
            # No numbers in generated text - can't verify numerically
            return {
                "verified": True,
                "error": None,
                "missing_values": [],
                "verified_values": [],
                "score": 1.0
            }
        
        missing = []
        verified = []
        
        for g_val in gen_values:
            # Use isclose for float comparison (tolerates small precision errors)
            # any() checks if g_val matches ANY number in the source set
            match_found = any(math.isclose(g_val, s_val, rel_tol=1e-9, abs_tol=1.0) for s_val in src_values)
            
            if not match_found:
                missing.append(g_val)
            else:
                verified.append(g_val)
        
        if missing:
            score = max(0.0, 1.0 - (len(missing) / len(gen_values)))
            return {
                "verified": False,
                "error": f"Verification Failed. The following values in the response were not found in context: {missing}",
                "missing_values": missing,
                "verified_values": verified,
                "score": score
            }
        
        return {
            "verified": True,
            "error": None,
            "missing_values": [],
            "verified_values": verified,
            "score": 1.0
        }


@dataclass
class VerificationResult:
    """Result of verification checks."""
    overall_score: float  # 0.0 to 1.0
    answer_source_alignment: float  # How well answer aligns with sources
    citation_coverage: float  # How many sources are cited
    fact_verification_score: float  # How many facts are verified
    issues: List[str]  # List of identified issues
    verified_sources: List[str]  # List of source IDs that support the answer
    unverified_claims: List[str]  # Claims that couldn't be verified
    verified_numbers: List[float]  # List of numbers that were verified in the answer
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "answer_source_alignment": self.answer_source_alignment,
            "citation_coverage": self.citation_coverage,
            "fact_verification_score": self.fact_verification_score,
            "issues": self.issues,
            "verified_sources": self.verified_sources,
            "unverified_claims": self.unverified_claims,
            "verified_numbers": self.verified_numbers
        }


class RAGVerifier:
    """Verifier for RAG responses using code-based verification."""
    
    def __init__(self, llm_client=None):
        """
        Initialize RAG verifier.
        
        Args:
            llm_client: Optional LLM client (kept for backward compatibility, not used)
        """
        self.financial_verifier = FinancialVerifier()
    
    def verify(
        self,
        answer: str,
        chunks: List[Chunk],
        query: str
    ) -> VerificationResult:
        """
        Verify RAG response against retrieved sources.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks used for generation
            query: Original query
            
        Returns:
            VerificationResult with verification scores and issues
        """
        if not chunks:
            return VerificationResult(
                overall_score=0.0,
                answer_source_alignment=0.0,
                citation_coverage=0.0,
                fact_verification_score=0.0,
                issues=["No sources retrieved"],
                verified_sources=[],
                unverified_claims=[],
                verified_numbers=[]
            )
        
        # Perform various verification checks
        answer_source_alignment = self._check_answer_source_alignment(answer, chunks, query)
        citation_coverage = self._check_citation_coverage(answer, chunks)
        fact_verification = self._verify_facts(answer, chunks)
        
        # Collect issues
        issues = []
        if answer_source_alignment < 0.5:
            issues.append("Answer may not be well-supported by retrieved sources")
        if citation_coverage < 0.5:
            issues.append("Many sources are not cited in the answer")
        if fact_verification["score"] < 0.7:
            issues.append("Some claims in the answer could not be verified")
        
        # Calculate overall score (weighted average)
        overall_score = (
            answer_source_alignment * 0.4 +
            citation_coverage * 0.3 +
            fact_verification["score"] * 0.3
        )
        
        return VerificationResult(
            overall_score=overall_score,
            answer_source_alignment=answer_source_alignment,
            citation_coverage=citation_coverage,
            fact_verification_score=fact_verification["score"],
            issues=issues,
            verified_sources=fact_verification["verified_sources"],
            unverified_claims=fact_verification["unverified_claims"],
            verified_numbers=fact_verification.get("verified_numbers", [])
        )
    
    def _check_answer_source_alignment(
        self,
        answer: str,
        chunks: List[Chunk],
        query: str
    ) -> float:
        """
        Check if answer aligns with retrieved sources using source tag-based verification.
        
        Parses source tags and verifies that numbers in cited sections match their
        corresponding chunks, and that uncited numbers are found in any source.
        
        Args:
            answer: Generated answer with source tags
            chunks: Retrieved chunks
            query: Original query (not used in code-based verification)
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        if not chunks:
            return 0.0
        
        try:
            # Parse source tags from answer
            source_sections, uncited_text = self._parse_source_tags(answer)
            
            if not source_sections and not uncited_text:
                return 0.0
            
            total_numbers = 0
            verified_numbers = 0
            
            # Verify each source-tagged section
            for source_section in source_sections:
                chunk_id = source_section['chunk_id']
                cited_text = source_section['text']
                
                # Find the corresponding chunk
                chunk = self._find_chunk_by_id(chunk_id, chunks)
                
                if not chunk:
                    # Chunk not found - count all numbers as unverified
                    cited_numbers = self.financial_verifier.extract_values(cited_text)
                    total_numbers += len(cited_numbers)
                    continue
                
                # Extract chunk text (remove metadata tag)
                chunk_text = self._extract_chunk_text_from_context(chunk.text)
                
                # Extract numbers from cited text and chunk text
                cited_numbers = self.financial_verifier.extract_values(cited_text)
                chunk_numbers = self.financial_verifier.extract_values(chunk_text)
                
                total_numbers += len(cited_numbers)
                verified_numbers += len(cited_numbers & chunk_numbers)
            
            # Verify uncited text against all chunks combined
            if uncited_text:
                uncited_numbers = self.financial_verifier.extract_values(uncited_text)
                total_numbers += len(uncited_numbers)
                
                if uncited_numbers:
                    # Combine all chunk texts for verification
                    all_chunk_texts = []
                    for chunk in chunks:
                        chunk_text = self._extract_chunk_text_from_context(chunk.text)
                        all_chunk_texts.append(chunk_text)
                    combined_source = ' '.join(all_chunk_texts)
                    
                    # Verify uncited numbers
                    combined_numbers = self.financial_verifier.extract_values(combined_source)
                    verified_numbers += len(uncited_numbers & combined_numbers)
            
            # Calculate alignment score
            if total_numbers == 0:
                # No numbers to verify - score based on whether sources are cited
                alignment_score = 1.0 if source_sections else 0.5
            else:
                alignment_score = verified_numbers / total_numbers if total_numbers > 0 else 0.5
            
            return max(0.0, min(1.0, alignment_score))
            
        except Exception as e:
            logger.warning(f"Error in answer-source alignment check: {e}. Using fallback.", exc_info=True)
            return self._simple_alignment_check(answer, chunks)
    
    def _simple_alignment_check(self, answer: str, chunks: List[Chunk]) -> float:
        """
        Simple fallback alignment check using keyword overlap.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        if not chunks:
            return 0.0
        
        # Extract key terms from answer (simple approach)
        answer_words = set(answer.lower().split())
        
        # Check overlap with chunk texts
        overlaps = []
        for chunk in chunks[:5]:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
            overlaps.append(overlap)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0
    
    def _check_citation_coverage(self, answer: str, chunks: List[Chunk]) -> float:
        """
        Check how many sources are cited in the answer.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            Citation coverage score between 0.0 and 1.0
        """
        if not chunks:
            return 0.0
        
        # Extract source identifiers from chunks
        source_identifiers = []
        for chunk in chunks:
            metadata = chunk.metadata or {}
            ticker = metadata.get('ticker', '').upper()
            year = metadata.get('year', '')
            
            # Create multiple possible citation patterns
            # Note: filing_type is not used since all documents are 10-K
            identifiers = [
                ticker,
                f"{ticker} {year}",
                year
            ]
            source_identifiers.extend([id for id in identifiers if id])
        
        # Check if answer contains any source identifiers
        answer_upper = answer.upper()
        cited_count = sum(1 for identifier in source_identifiers if identifier and identifier.upper() in answer_upper)
        
        # Also check for common citation patterns
        citation_patterns = [
            r'\[Source[:\s]+\d+\]',
            r'\(Source[:\s]+\d+\)',
            r'according to',
            r'per the',
            r'in the.*filing',
            r'from.*filing'
        ]
        
        pattern_matches = sum(1 for pattern in citation_patterns if re.search(pattern, answer, re.IGNORECASE))
        
        # Calculate coverage
        total_sources = len(chunks)
        if total_sources == 0:
            return 0.0
        
        # Weight: explicit citations (ticker/year) + pattern matches
        explicit_citations = min(cited_count / max(total_sources, 1), 1.0)
        pattern_bonus = min(pattern_matches * 0.1, 0.3)  # Up to 0.3 bonus
        
        return min(explicit_citations + pattern_bonus, 1.0)
    
    def _parse_source_tags(self, answer: str) -> tuple[List[Dict[str, Any]], str]:
        """
        Parse answer to extract source-tagged sections and uncited text.
        
        Args:
            answer: Generated answer with source tags
            
        Returns:
            Tuple of (source_sections, uncited_text) where:
            - source_sections: List of dicts with 'ticker', 'year', 'chunk_id', 'text'
            - uncited_text: Text outside of source tags
        """
        import re
        
        source_sections = []
        uncited_parts = []
        
        # Pattern to match <source ticker="..." year="..." chunk_id="...">...</source>
        source_pattern = re.compile(
            r'<source\s+ticker="([^"]+)"\s+year="([^"]+)"\s+chunk_id="([^"]+)">(.*?)</source>',
            re.DOTALL | re.IGNORECASE
        )
        
        last_end = 0
        for match in source_pattern.finditer(answer):
            # Add text before this source tag to uncited
            if match.start() > last_end:
                uncited_parts.append(answer[last_end:match.start()])
            
            # Extract source tag information
            ticker = match.group(1)
            year = match.group(2)
            chunk_id = match.group(3)
            text = match.group(4).strip()
            
            source_sections.append({
                'ticker': ticker,
                'year': year,
                'chunk_id': chunk_id,
                'text': text
            })
            
            last_end = match.end()
        
        # Add remaining text after last source tag
        if last_end < len(answer):
            uncited_parts.append(answer[last_end:])
        
        uncited_text = ' '.join(uncited_parts).strip()
        
        return source_sections, uncited_text
    
    def _find_chunk_by_id(self, chunk_id: str, chunks: List[Chunk]) -> Optional[Chunk]:
        """
        Find a chunk by its chunk_id.
        
        Args:
            chunk_id: Chunk ID to find
            chunks: List of chunks to search
            
        Returns:
            Chunk if found, None otherwise
        """
        for chunk in chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def _extract_chunk_text_from_context(self, chunk_text: str) -> str:
        """
        Extract the actual chunk text, removing the metadata tag.
        
        The chunk text may start with either:
        - [Ticker="...", Year="...", Chunk_id="..."] (from ChromaDB)
        - [Source: ... chunk_id: ...] (from prompt builder context)
        
        We need to extract the text after this tag.
        
        Args:
            chunk_text: Full chunk text with metadata tag
            
        Returns:
            Chunk text without metadata tag
        """
        import re
        
        # Pattern 1: [Ticker="...", Year="...", Chunk_id="..."]
        ticker_pattern = re.compile(
            r'\[Ticker="[^"]+",\s*Year="[^"]+",\s*Chunk_id="[^"]+"\]\s*',
            re.IGNORECASE
        )
        
        # Pattern 2: [Source: ... chunk_id: ...]
        source_pattern = re.compile(
            r'\[Source:[^\]]+\]\s*',
            re.IGNORECASE
        )
        
        # Try to remove metadata tag (try both patterns)
        cleaned_text = ticker_pattern.sub('', chunk_text, count=1)
        cleaned_text = source_pattern.sub('', cleaned_text, count=1)
        
        return cleaned_text.strip()
    
    def _verify_facts(self, answer: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Verify facts/claims from the answer using source tag-based verification.
        
        Parses source tags in the answer, extracts numbers from cited and uncited text,
        and verifies that numbers in source tags match the corresponding chunks.
        
        Args:
            answer: Generated answer with source tags
            chunks: Retrieved chunks
            
        Returns:
            Dictionary with verification results
        """
        try:
            if not chunks:
                return {
                    "score": 0.0,
                    "verified_sources": [],
                    "unverified_claims": [],
                    "verified_numbers": []
                }
            
            # Parse source tags from answer
            source_sections, uncited_text = self._parse_source_tags(answer)
            
            verified_sources = []
            unverified_claims = []
            all_verified_numbers = []
            
            # Verify each source-tagged section
            for source_section in source_sections:
                chunk_id = source_section['chunk_id']
                cited_text = source_section['text']
                
                # Find the corresponding chunk
                chunk = self._find_chunk_by_id(chunk_id, chunks)
                
                if not chunk:
                    unverified_claims.append(
                        f"Chunk ID {chunk_id} not found in retrieved sources"
                    )
                    continue
                
                # Extract chunk text (remove metadata tag)
                chunk_text = self._extract_chunk_text_from_context(chunk.text)
                
                # Extract numbers from cited text and chunk text
                cited_numbers = self.financial_verifier.extract_values(cited_text)
                chunk_numbers = self.financial_verifier.extract_values(chunk_text)
                
                # Verify numbers in cited text exist in chunk
                missing_in_chunk = cited_numbers - chunk_numbers
                verified_in_chunk = cited_numbers & chunk_numbers
                
                if missing_in_chunk:
                    for num in missing_in_chunk:
                        unverified_claims.append(
                            f"Number {num} in citation for chunk {chunk_id} not found in source chunk"
                        )
                
                if verified_in_chunk:
                    all_verified_numbers.extend(verified_in_chunk)
                    if chunk_id not in verified_sources:
                        verified_sources.append(chunk_id)
            
            # Also verify uncited text against all chunks combined
            if uncited_text:
                uncited_numbers = self.financial_verifier.extract_values(uncited_text)
                
                if uncited_numbers:
                    # Combine all chunk texts for verification
                    all_chunk_texts = []
                    for chunk in chunks:
                        chunk_text = self._extract_chunk_text_from_context(chunk.text)
                        all_chunk_texts.append(chunk_text)
                    combined_source = ' '.join(all_chunk_texts)
                    
                    # Verify uncited numbers
                    combined_numbers = self.financial_verifier.extract_values(combined_source)
                    missing_uncited = uncited_numbers - combined_numbers
                    
                    if missing_uncited:
                        for num in missing_uncited:
                            unverified_claims.append(
                                f"Number {num} in uncited text not found in any source"
                            )
                    
                    # Add verified uncited numbers
                    verified_uncited = uncited_numbers & combined_numbers
                    all_verified_numbers.extend(verified_uncited)
            
            # Calculate score based on verification results
            total_cited_numbers = sum(
                len(self.financial_verifier.extract_values(section['text']))
                for section in source_sections
            )
            total_uncited_numbers = len(self.financial_verifier.extract_values(uncited_text)) if uncited_text else 0
            total_numbers = total_cited_numbers + total_uncited_numbers
            
            if total_numbers == 0:
                # No numbers to verify - score based on whether sources are cited
                score = 1.0 if source_sections else 0.5
            else:
                # Score based on proportion of verified numbers
                verified_count = len(all_verified_numbers)
                score = verified_count / total_numbers if total_numbers > 0 else 0.5
            
            # Remove duplicates from verified_numbers
            unique_verified_numbers = list(set(all_verified_numbers))
            
            return {
                "score": score,
                "verified_sources": verified_sources,
                "unverified_claims": unverified_claims,
                "verified_numbers": unique_verified_numbers
            }
            
        except Exception as e:
            logger.warning(f"Error in fact verification: {e}", exc_info=True)
            return {
                "score": 0.5,
                "verified_sources": [],
                "unverified_claims": [],
                "verified_numbers": []
            }

