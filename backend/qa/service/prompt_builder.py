"""Prompt building for RAG queries."""

from typing import List, Optional, Dict, Any
from .models import Chunk
import logging

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builder for RAG prompts."""
    
    SYSTEM_PROMPT = """You are an expert financial analyst assistant with deep expertise in SEC filings, financial statements, and corporate analysis. You excel at:
- Multi-step calculations and financial analysis
- Synthesizing information from multiple sources
- Connecting quantitative data with qualitative narratives
- Performing inferential reasoning and projections
- Identifying patterns, contradictions, and relationships in financial data"""
    
    @staticmethod
    def build_context(chunks: List[Chunk], max_length: int = 50000) -> str:
        """
        Build context string from chunks.
        
        Args:
            chunks: List of retrieved chunks
            max_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        separator = "\n\n"
        separator_length = len(separator)
        current_length = 0
        logger.info(f"Given chunks are: {chunks}")
        for chunk in chunks:
            chunk_text = chunk.text
            metadata = chunk.metadata or {}
            ticker = metadata.get('ticker', 'Unknown')
            year = metadata.get('year', 'Unknown')
            filing_type = metadata.get('filing_type', 'Unknown')
            chunk_id = chunk.chunk_id
            
            # Format chunk with metadata
            header = f"[Source: {ticker} {filing_type} {year}, chunk_id: {chunk_id}]\n"
            chunk_with_meta = header + chunk_text
            chunk_length = len(chunk_with_meta)
            
            # Calculate total length if we add this chunk
            if context_parts:
                # Need separator between chunks
                total_length = current_length + separator_length + chunk_length
            else:
                # First chunk, no separator needed
                total_length = chunk_length
            
            # Check if adding this chunk would exceed max length
            if total_length > max_length:
                # If this is the first chunk and it's too long, truncate it
                if not context_parts:
                    # Truncate the chunk text to fit within max_length
                    available_length = max_length - len(header) - 3  # -3 for "..."
                    if available_length > 0:
                        truncated_text = chunk_text[:available_length] + "..."
                        chunk_with_meta = header + truncated_text
                        context_parts.append(chunk_with_meta)
                        current_length = len(chunk_with_meta)
                # Otherwise, stop adding chunks
                break
            
            context_parts.append(chunk_with_meta)
            # Update current length (including separators)
            if len(context_parts) == 1:
                current_length = chunk_length
            else:
                current_length = total_length
        
        context = separator.join(context_parts)
        return context
    
    @staticmethod
    def build_prompt(query: str, context: str) -> str:
        """
        Build prompt for LLM with query and context.
        
        Args:
            query: User query
            context: Context from retrieved chunks
            
        Returns:
            Formatted prompt
        """
        return f"""You are a financial analyst assistant. Answer the user's question based on the provided context from SEC filings.

Context from SEC filings:
{context}

Question: {query}

CRITICAL: Use Chain-of-Thought Reasoning

For complex questions involving calculations, multi-step analysis, synthesis, or inferential reasoning, you MUST show your reasoning process using the following format:

<thinking>
Step 1: [Reflect on the question and what information you need to find first]
Step 2: [Describe the next step or calculation]
Step 3: [Continue with subsequent steps]
...
Final Step: [Synthesize the findings]
</thinking>

<answer>
[Your final answer with proper citations]
</answer>

Examples of when to use chain-of-thought:

1. **Multi-Step Calculations**: 
   - "Calculate Net Working Capital for two years"
   - Show: Step 1: Find Current Assets Year 1, Step 2: Find Current Liabilities Year 1, Step 3: Calculate NWC Year 1, etc.

2. **Synthesis Questions**:
   - "Compare Gross Margin and Operating Margin trends"
   - Show: Step 1: Retrieve revenue and COGS, Step 2: Calculate Gross Margin, Step 3: Retrieve Operating Income, Step 4: Calculate Operating Margin, Step 5: Compare trends

3. **Narrative & Risk Analysis**:
   - "Identify risks and find evidence of materialization"
   - Show: Step 1: Extract top risks from Risk Factors, Step 2: Search MD&A for evidence, Step 3: Connect quantitative impact

4. **Inferential Reasoning**:
   - "Calculate debt covenant headroom"
   - Show: Step 1: Find covenant requirements, Step 2: Retrieve current financial metrics, Step 3: Calculate current ratio, Step 4: Calculate headroom

Instructions:
- ALWAYS use <thinking> tags for complex questions (calculations, multi-step analysis, synthesis)
- For simple factual questions, you may skip the thinking section
- Answer based solely on the provided context
- If the context doesn't contain enough information, say so in your thinking
- Cite specific sources using <source> tags (see below)
- Use professional financial terminology
- Show all calculations explicitly

CITATION FORMAT:
For all information presented in your answer that is drawn from a chunk, cite the chunk using:
<source ticker="AAPL" year="2023" chunk_id="1234567890"> [cited text] </source>

Each chunk has a source header: [Source: AAPL 2023, chunk_id: 1234567890]

Response format:
<thinking>
[Your reasoning steps - only for complex questions]
</thinking>

<answer>
[Your final answer with citations]
</answer>"""
    
    @staticmethod
    def build_empty_response() -> str:
        """Build response when no chunks are retrieved."""
        return "I couldn't find any relevant information in the SEC filings to answer your question."
    
    @staticmethod
    def build_filter_analysis_prompt(query: str) -> str:
        """
        Build prompt for analyzing query to determine appropriate filters.
        
        Args:
            query: User query
            
        Returns:
            Prompt for filter analysis
        """
        return f"""You are a financial data retrieval assistant. Analyze the following query and determine what filters should be applied to retrieve the most relevant SEC filing documents.

Query: {query}

Your task:
1. Identify if a specific company (ticker) is mentioned or implied
2. Identify if a specific year or time period is mentioned
3. Identify if a specific SEC Item number is mentioned (Item 1, Item 1A, Item 2, Item 7, Item 7A, etc.)
4. Identify if a specific filing type (10-K, 10-Q) is mentioned or would be most relevant
5. Explain your reasoning for each filter decision

Available filters:
- ticker: Company ticker symbol (e.g., "AAPL", "MSFT", "GOOGL", "JPM", "JNJ")
- year: Fiscal year as string (e.g., "2023", "2022", "2024")
- item_number: SEC Item number (e.g., "1", "1A", "1B", "2", "7", "7A", "8", "9", "10")
- filing_type: Type of filing ("10-K" for annual, "10-Q" for quarterly)

Common Item numbers:
- Item 1: Business
- Item 1A: Risk Factors
- Item 1B: Unresolved Staff Comments
- Item 2: Properties
- Item 3: Legal Proceedings
- Item 7: Management's Discussion and Analysis
- Item 7A: Quantitative and Qualitative Disclosures About Market Risk
- Item 8: Financial Statements and Supplementary Data

Response format (JSON):
{{
    "reasoning": "Step-by-step explanation of why each filter was chosen or not chosen",
    "filters": {{
        "ticker": "AAPL" or null,
        "year": "2023" or null,
        "item_number": "1" or "1A" or "7" or null,
        "filing_type": "10-K" or "10-Q" or null
    }},
    "confidence": 0.0-1.0
}}

Only include filters that are explicitly mentioned or strongly implied. If uncertain, set to null.
For item_number, extract the number part only (e.g., "1" for "Item 1", "1A" for "Item 1A").
Respond with ONLY the JSON object, no additional text."""

    @staticmethod
    def parse_response(response: str) -> tuple[str, Optional[str]]:
        """
        Parse response to extract thinking and answer sections.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (answer, thinking) where thinking may be None
        """
        import re
        
        # Try to extract thinking section
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL | re.IGNORECASE)
        thinking = thinking_match.group(1).strip() if thinking_match else None
        
        # Try to extract answer section
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # If no <answer> tags, use the whole response (minus thinking if present)
            if thinking_match:
                # Remove thinking section from response
                answer = response[:thinking_match.start()] + response[thinking_match.end():]
                answer = re.sub(r'</?thinking>', '', answer, flags=re.IGNORECASE).strip()
            else:
                answer = response.strip()
        
        return answer, thinking
    
    @staticmethod
    def parse_filter_analysis(response: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse filter analysis response from LLM.
        
        Args:
            response: Raw LLM response with JSON
            
        Returns:
            Tuple of (filters_dict, reasoning) where filters_dict may be None
        """
        import json
        import re
        
        # Try to extract JSON from response
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
                    
                    filters = result.get("filters", {})
                    # Remove null values
                    filters = {k: v for k, v in filters.items() if v is not None}
                    if not filters:
                        filters = None
                    
                    reasoning = result.get("reasoning", None)
                    
                    return filters, reasoning
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse filter analysis JSON: {e}")
                    return None, None
        
        return None, None

