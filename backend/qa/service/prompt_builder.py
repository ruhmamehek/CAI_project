"""Prompt building for RAG queries."""

from typing import List, Optional, Dict, Any
from .models import Chunk
from .utils import clean_image_text
import logging

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builder for RAG prompts."""
    
    SYSTEM_PROMPT = """You are an expert financial analyst assistant with deep expertise in SEC filings, financial statements, and corporate analysis. You excel at:
- Multi-step calculations and financial analysis
- Synthesizing information from multiple sources
- Connecting quantitative data with qualitative narratives
- Performing inferential reasoning and projections
- Identifying patterns, contradictions, and relationships in financial data
- Providing concise and accurate answers with citations"""
    
    @staticmethod
    def build_context(chunks: List[Chunk], max_length: int = 50000) -> str:
        """Build context string from chunks."""
        context_parts = []
        separator = "\n\n"
        separator_length = len(separator)
        current_length = 0
        for chunk in chunks:
            chunk_text = clean_image_text(chunk.text) or ""
            
            metadata = chunk.metadata or {}
            ticker = metadata.get('ticker', 'Unknown')
            year = metadata.get('year', 'Unknown')
            chunk_id = chunk.chunk_id
            
            header = f"[Source: {ticker} {year}, chunk_id: {chunk_id}]\n"
            chunk_with_meta = header + chunk_text
            chunk_length = len(chunk_with_meta)
            
            if context_parts:
                total_length = current_length + separator_length + chunk_length
            else:
                total_length = chunk_length
            
            if total_length > max_length:
                if not context_parts:
                    available_length = max_length - len(header) - 3
                    if available_length > 0:
                        truncated_text = chunk_text[:available_length] + "..."
                        chunk_with_meta = header + truncated_text
                        context_parts.append(chunk_with_meta)
                        current_length = len(chunk_with_meta)
                break
            
            context_parts.append(chunk_with_meta)
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
        return f"""{PromptBuilder.SYSTEM_PROMPT}

Context from SEC filings:
{context}

Question: {query}

### STRICT CITATION PROTOCOL
You are required to provide evidence for every factual statement. Follow these rules:

1. **Granularity:** Every distinct claim must be cited immediately. If a sentence draws from two different chunks, you must insert the relevant citation after each clause.
2. **Verifiability:** Inside the `<source>` tag, you must include a brief snippet of the text that supports your claim.
3. **Integrity:** Do not fabricate chunk IDs. Only use the IDs provided in the "Context" section above.

### CITATION FORMAT 
Use this exact XML format:
<source ticker="[TICKER]" year="[YEAR]" chunk_id="[ID]">[supporting text from chunk]</source>

### EXAMPLE
**Context Chunk:** <source ticker="MSFT" year="2023" chunk_id="8821">Revenue increased by 10% due to cloud growth.</source>
<source ticker="MSFT" year="2023" chunk_id="8822">Operating expenses rose 5% driven by R&D.</source>

**Correct Response:**
Microsoft reported a 10% revenue increase driven by their cloud division <source ticker="MSFT" year="2023" chunk_id="8821">Revenue increased by 10% due to cloud growth</source>, while operating expenses grew by 5% specifically due to research and development costs <source ticker="MSFT" year="2023" chunk_id="8822">Operating expenses rose 5% driven by R&D</source>.

Response format:
<answer>
[Your answer with strict inline citations]
</answer>
"""
    
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
4. Explain your reasoning for each filter decision

Note: All documents are 10-K annual reports, so filing type is not a filter option.

Available filters:
- ticker: Company ticker symbol (e.g., "AAPL", "MSFT", "GOOGL", "JPM", "JNJ")
- year: Fiscal year as string (e.g., "2023", "2022", "2024")
- item_number: SEC Item number (e.g., "1", "1A", "1B", "2", "7", "7A", "8", "9", "10")

Item Labels & Definitions:
* **Item 1 (Business):** Core operations, products/services, subsidiaries, markets, and competitive landscape.
* **Item 1A (Risk Factors):** Significant risks to the company, its industry, or its stock (excluding how they address them).
* **Item 1B (Unresolved Staff Comments):** Outstanding questions or disputes with the SEC staff.
* **Item 2 (Properties):** Physical assets, including plants, mines, factories, and real estate.
* **Item 3 (Legal Proceedings):** Pending lawsuits, litigation, or government legal actions.
* **Item 5 (Market & Equity):** Stock performance, dividends, share buybacks, and number of shareholders.
* **Item 6 (Selected Financial Data):** A high-level 5-year summary of financial trends.
* **Item 7 (MD&A):** Management’s narrative explanation of financial results, liquidity, capital resources, and critical accounting estimates.
* **Item 7A (Market Risk):** Exposure to external market forces like interest rates, currency exchange, and commodity prices.
* **Item 8 (Financial Statements):** The official audited Income Statement, Balance Sheet, Cash Flows, and Auditor's Opinion.
* **Item 9 (Accountant Changes):** Disagreements with auditors or changes in accounting firms.
* **Item 9A (Controls):** Effectiveness of internal financial controls and disclosure procedures.
* **Item 10 (Governance):** Directors, executive officers, code of ethics, and board qualifications.
* **Item 11 (Compensation):** Executive pay, salaries, bonuses, and compensation policies.
* **Item 12 (Ownership):** Stock ownership by management, directors, and major (>5%) shareholders.
* **Item 13 (Relationships):** Related party transactions and director independence.
* **Item 14 (Accountant Fees):** Fees paid to the external auditing firm for services.
* **Item 15 (Exhibits):** Legal contracts, bylaws, and list of subsidiaries.

Response format (JSON):
{{
    "reasoning": "Brief explanation on the process that led to the filters and why the query fits the filters that have been applied. End off with a list of the filters that have been applied. for exampled: [ticker: AAPL, year: 2023, item_number: 1]",
    "filters": {{
        "ticker": "AAPL" or null,
        "year": "2023" or null,
        "item_number": "1" or "1A" or "7" or null
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

