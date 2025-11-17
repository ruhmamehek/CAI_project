"""Verification module for RAG responses."""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .models import Chunk, Source
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "answer_source_alignment": self.answer_source_alignment,
            "citation_coverage": self.citation_coverage,
            "fact_verification_score": self.fact_verification_score,
            "issues": self.issues,
            "verified_sources": self.verified_sources,
            "unverified_claims": self.unverified_claims
        }


class RAGVerifier:
    """Verifier for RAG responses."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize RAG verifier.
        
        Args:
            llm_client: LLM client for verification tasks
        """
        self.llm_client = llm_client
    
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
                unverified_claims=[]
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
            unverified_claims=fact_verification["unverified_claims"]
        )
    
    def _check_answer_source_alignment(
        self,
        answer: str,
        chunks: List[Chunk],
        query: str
    ) -> float:
        """
        Check if answer aligns with retrieved sources.
        
        Uses LLM to evaluate semantic alignment.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            query: Original query
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        try:
            # Build context from top chunks
            context_parts = []
            for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks
                metadata = chunk.metadata or {}
                ticker = metadata.get('ticker', 'Unknown')
                year = metadata.get('year', 'Unknown')
                filing_type = metadata.get('filing_type', 'Unknown')
                context_parts.append(
                    f"[Source {i+1}: {ticker} {filing_type} {year}]\n{chunk.text[:500]}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Use LLM to evaluate alignment
            prompt = f"""You are a verification assistant. Evaluate whether the given answer is well-supported by the provided sources.

Query: {query}

Sources:
{context}

Answer to verify:
{answer}

Evaluate:
1. Does the answer address the query?
2. Is the information in the answer supported by the sources?
3. Are there any contradictions between the answer and sources?

Respond with a JSON object containing:
- "score": a float between 0.0 and 1.0 (1.0 = fully supported, 0.0 = not supported)
- "reasoning": brief explanation

JSON response:"""
            
            response = self.llm_client.generate(prompt)
            
            # Extract JSON from response
            import json
            # Try to find JSON object in response (handle nested objects)
            # Look for content between first { and last }
            start_idx = response.find('{')
            if start_idx != -1:
                # Find matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    try:
                        json_str = response[start_idx:end_idx]
                        result = json.loads(json_str)
                        score = float(result.get("score", 0.5))
                        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
            
            # Fallback: simple keyword matching
            return self._simple_alignment_check(answer, chunks)
            
        except Exception as e:
            logger.warning(f"Error in answer-source alignment check: {e}. Using fallback.")
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
            filing_type = metadata.get('filing_type', '').upper()
            
            # Create multiple possible citation patterns
            identifiers = [
                ticker,
                f"{ticker} {year}",
                # f"{ticker} {filing_type}",
                # f"{ticker} {filing_type} {year}",
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
    
    def _verify_facts(self, answer: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Extract and verify facts/claims from the answer.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Build context from chunks
            context_parts = []
            source_map = {}  # Map chunk index to source info
            for i, chunk in enumerate(chunks[:10]):  # Use top 10 chunks
                metadata = chunk.metadata or {}
                ticker = metadata.get('ticker', 'Unknown')
                year = metadata.get('year', 'Unknown')
                # filing_type = metadata.get('filing_type', 'Unknown')
                chunk_id = chunk.chunk_id
                
                source_map[i] = {
                    "ticker": ticker,
                    "year": year,
                    # "filing_type": filing_type,
                    "chunk_id": chunk_id
                }
                
                context_parts.append(
                    f"[Source {i+1}: {ticker} {year}]\n{chunk.text[:800]}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Use LLM to extract and verify facts
            prompt = f"""You are a fact verification assistant. Extract key factual claims from the answer and verify them against the sources.

Sources:
{context}

Answer to verify:
{answer}

Task:
1. Extract 1-5 key factual claims from the answer
2. For each claim, check if it's supported by the sources
3. Identify which source(s) support each claim (use Source number)

Respond with a JSON object:
{{
    "claims": [
        {{
            "claim": "the factual claim text",
            "verified": true/false,
            "supporting_sources": [1, 2],  // source numbers
            "confidence": 0.0-1.0
        }}
    ]
}}

JSON response:"""
            
            response = self.llm_client.generate(prompt)
            
            # Extract JSON from response
            import json
            # Try to find JSON object in response (handle nested objects)
            start_idx = response.find('{')
            if start_idx != -1:
                # Find matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    try:
                        json_str = response[start_idx:end_idx]
                        result = json.loads(json_str)
                        claims = result.get("claims", [])
                        
                        if not claims:
                            return {
                                "score": 0.5,
                                "verified_sources": [],
                                "unverified_claims": []
                            }
                        
                        verified_count = sum(1 for claim in claims if claim.get("verified", False))
                        verified_sources = []
                        unverified_claims = []
                        
                        for claim in claims:
                            if claim.get("verified", False):
                                source_nums = claim.get("supporting_sources", [])
                                for src_num in source_nums:
                                    if src_num in source_map:
                                        chunk_id = source_map[src_num]["chunk_id"]
                                        if chunk_id not in verified_sources:
                                            verified_sources.append(chunk_id)
                            else:
                                unverified_claims.append(claim.get("claim", ""))
                        
                        score = verified_count / len(claims) if claims else 0.0
                        
                        return {
                            "score": score,
                            "verified_sources": verified_sources,
                            "unverified_claims": unverified_claims
                        }
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
            
            # Fallback
            return {
                "score": 0.5,
                "verified_sources": [],
                "unverified_claims": []
            }
            
        except Exception as e:
            logger.warning(f"Error in fact verification: {e}")
            return {
                "score": 0.5,
                "verified_sources": [],
                "unverified_claims": []
            }

