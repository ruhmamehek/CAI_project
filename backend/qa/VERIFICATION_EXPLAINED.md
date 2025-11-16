# RAG Verification System - How It Works

## Overview

The verification system validates RAG-generated answers by checking three aspects:
1. **Answer-Source Alignment** - Does the answer match the retrieved sources?
2. **Citation Coverage** - Are sources properly cited in the answer?
3. **Fact Verification** - Can each factual claim be verified against sources?

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Query Pipeline                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  1. Retrieve Chunks (ChromaDB)      │
        │     - Vector similarity search      │
        │     - Returns top K chunks         │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  2. Rerank Chunks (Cross-Encoder)   │
        │     - Improve relevance ranking      │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  3. Generate Answer (LLM)           │
        │     - Uses chunks as context        │
        │     - Produces final answer        │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │             4. VERIFICATION (if enabled)                │
        │                                                           │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │  A. Answer-Source Alignment Check                 │   │
        │  │     ┌──────────────────────────────────────┐    │   │
        │  │     │ 1. Build context from top 5 chunks   │    │   │
        │  │     │ 2. Send to LLM with prompt:          │    │   │
        │  │     │    - Query                            │    │   │
        │  │     │    - Sources                          │    │   │
        │  │     │    - Answer                           │    │   │
        │  │     │ 3. LLM evaluates:                      │    │   │
        │  │     │    - Does answer address query?       │    │   │
        │  │     │    - Is info supported by sources?   │    │   │
        │  │     │    - Any contradictions?             │    │   │
        │  │     │ 4. LLM returns JSON: {score: 0.0-1.0}│    │   │
        │  │     │ 5. Fallback: Keyword overlap if LLM  │    │   │
        │  │     │    fails                              │    │   │
        │  │     └──────────────────────────────────────┘    │   │
        │  │                                                 │   │
        │  │  B. Citation Coverage Check                     │   │
        │  │     ┌──────────────────────────────────────┐    │   │
        │  │     │ 1. Extract source identifiers:       │    │   │
        │  │     │    - Ticker (AAPL)                    │    │   │
        │  │     │    - Year (2023)                      │    │   │
        │  │     │    - Filing type (10-Q)               │    │   │
        │  │     │    - Combinations                      │    │   │
        │  │     │ 2. Check if answer contains:         │    │   │
        │  │     │    - Explicit citations (ticker/year) │    │   │
        │  │     │    - Citation patterns:               │    │   │
        │  │     │      * [Source: 1]                    │    │   │
        │  │     │      * "according to"                  │    │   │
        │  │     │      * "in the filing"                │    │   │
        │  │     │ 3. Calculate:                          │    │   │
        │  │     │    coverage = citations / total_sources│    │   │
        │  │     │    + pattern bonus (up to 0.3)        │    │   │
        │  │     └──────────────────────────────────────┘    │   │
        │  │                                                 │   │
        │  │  C. Fact Verification                           │   │
        │  │     ┌──────────────────────────────────────┐    │   │
        │  │     │ 1. Build context from top 10 chunks   │    │   │
        │  │     │ 2. Send to LLM with prompt:          │    │   │
        │  │     │    - Extract 1-5 key factual claims   │    │   │
        │  │     │    - Verify each against sources      │    │   │
        │  │     │    - Identify supporting sources     │    │   │
        │  │     │ 3. LLM returns JSON:                 │    │   │
        │  │     │    {                                  │    │   │
        │  │     │      "claims": [                      │    │   │
        │  │     │        {                               │    │   │
        │  │     │          "claim": "...",              │    │   │
        │  │     │          "verified": true/false,      │    │   │
        │  │     │          "supporting_sources": [1,2]  │    │   │
        │  │     │        }                              │    │   │
        │  │     │      ]                                │    │   │
        │  │     │    }                                  │    │   │
        │  │     │ 4. Calculate:                         │    │   │
        │  │     │    score = verified_claims / total    │    │   │
        │  │     │    verified_sources = unique IDs     │    │   │
        │  │     │    unverified_claims = list          │    │   │
        │  │     └──────────────────────────────────────┘    │   │
        │  └──────────────────────────────────────────────────┘   │
        │                                                           │
        │  ┌──────────────────────────────────────────────────┐   │
        │  │  D. Aggregate Results                             │   │
        │  │     ┌──────────────────────────────────────┐    │   │
        │  │     │ 1. Calculate overall score:          │    │   │
        │  │     │    overall = alignment * 0.4 +       │    │   │
        │  │     │              citation * 0.3 +         │    │   │
        │  │     │              facts * 0.3                │    │   │
        │  │     │ 2. Collect issues:                     │    │   │
        │  │     │    - alignment < 0.5 → warning         │    │   │
        │  │     │    - citation < 0.5 → warning         │    │   │
        │  │     │    - facts < 0.7 → warning             │    │   │
        │  │     │ 3. Return VerificationResult          │    │   │
        │  │     └──────────────────────────────────────┘    │   │
        │  └──────────────────────────────────────────────────┘   │
        └─────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  5. Return QueryResponse            │
        │     - answer                        │
        │     - sources                       │
        │     - verification (if enabled)      │
        └─────────────────────────────────────┘
```

## Detailed Flow

### Step 1: Answer-Source Alignment (40% weight)

**Purpose**: Check if the generated answer semantically aligns with the retrieved sources.

**Process**:
1. Takes top 5 chunks and builds a context string
2. Sends to LLM with a verification prompt asking:
   - Does the answer address the query?
   - Is the information supported by sources?
   - Are there contradictions?
3. LLM returns a JSON with a score (0.0-1.0)
4. **Fallback**: If LLM fails, uses keyword overlap:
   - Extracts words from answer and chunks
   - Calculates overlap percentage
   - Averages across top 5 chunks

**Example**:
```
Query: "What was Apple's revenue in 2023?"
Answer: "Apple's revenue was $394 billion in 2023"
Sources: [chunks about Apple's financials]

LLM evaluates: "The answer addresses the query and the revenue figure 
appears in the sources. Score: 0.85"
```

### Step 2: Citation Coverage (30% weight)

**Purpose**: Check if sources are properly cited in the answer.

**Process**:
1. Extracts identifiers from chunks:
   - Ticker: "AAPL"
   - Year: "2023"
   - Filing type: "10-Q"
   - Combinations: "AAPL 2023", "AAPL 10-Q 2023"
2. Searches answer for:
   - **Explicit citations**: ticker, year, filing type
   - **Citation patterns**: 
     - `[Source: 1]`
     - `(Source: 1)`
     - "according to"
     - "per the filing"
     - "in the ... filing"
3. Calculates coverage:
   - `explicit_citations = citations_found / total_sources`
   - `pattern_bonus = min(pattern_matches * 0.1, 0.3)`
   - `coverage = min(explicit_citations + pattern_bonus, 1.0)`

**Example**:
```
Answer: "Apple's revenue was $394 billion [Source: AAPL 10-Q 2023]"
Sources: 5 chunks

Found: "AAPL", "2023", "10-Q", "[Source: ...]" pattern
Coverage: 1.0 (all sources cited)
```

### Step 3: Fact Verification (30% weight)

**Purpose**: Extract and verify each factual claim in the answer.

**Process**:
1. Takes top 10 chunks and builds context
2. Sends to LLM with prompt to:
   - Extract 1-5 key factual claims
   - Verify each claim against sources
   - Identify supporting source numbers
3. LLM returns JSON:
   ```json
   {
     "claims": [
       {
         "claim": "Apple's revenue was $394 billion",
         "verified": true,
         "supporting_sources": [1, 3],
         "confidence": 0.95
       }
     ]
   }
   ```
4. Calculates:
   - `score = verified_claims / total_claims`
   - Collects `verified_sources` (unique chunk IDs)
   - Collects `unverified_claims` (claims that couldn't be verified)

**Example**:
```
Claims extracted:
1. "Apple's revenue was $394 billion" → verified ✓ (sources 1, 3)
2. "Revenue increased 8% YoY" → verified ✓ (source 2)
3. "iPhone sales were $200B" → unverified ✗ (not in sources)

Score: 2/3 = 0.67
```

### Step 4: Aggregate Results

**Overall Score Calculation**:
```python
overall_score = (
    answer_source_alignment * 0.4 +  # 40% weight
    citation_coverage * 0.3 +          # 30% weight
    fact_verification_score * 0.3     # 30% weight
)
```

**Issue Detection**:
- `alignment < 0.5` → "Answer may not be well-supported by retrieved sources"
- `citation < 0.5` → "Many sources are not cited in the answer"
- `facts < 0.7` → "Some claims in the answer could not be verified"

## Error Handling

All three verification steps have fallback mechanisms:

1. **Answer-Source Alignment**: Falls back to keyword overlap if LLM fails
2. **Citation Coverage**: Always works (no LLM dependency)
3. **Fact Verification**: Returns default score (0.5) if LLM fails

If verification completely fails, the system:
- Logs the error
- Continues without verification (doesn't break the query)
- Returns response without verification object

## Performance Considerations

- **LLM Calls**: Verification makes 2 additional LLM calls per query
  - One for alignment check
  - One for fact verification
- **Timeout**: Verification can take 30-60 seconds depending on LLM response time
- **Cost**: Adds LLM API costs (2x verification calls per query)

## Configuration

Verification can be controlled via:

1. **Config file** (`config.yaml`):
   ```yaml
   verification:
     enable: true
   ```

2. **Environment variable**:
   ```bash
   ENABLE_VERIFICATION=true
   ```

3. **Per-request**:
   ```json
   {
     "query": "...",
     "enable_verification": true
   }
   ```

## Output Format

```json
{
  "answer": "...",
  "sources": [...],
  "verification": {
    "overall_score": 0.78,
    "answer_source_alignment": 0.45,
    "citation_coverage": 1.00,
    "fact_verification_score": 1.00,
    "issues": [
      "Answer may not be well-supported by retrieved sources"
    ],
    "verified_sources": ["chunk_id_1", "chunk_id_2"],
    "unverified_claims": []
  }
}
```

