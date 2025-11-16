# SEC Filings QA Service

RAG-powered question-answering service for SEC filings using ChromaDB and LLMs.

## Quick Start

### Option 1: Docker (Recommended)

1. **Create `.env` file** in the `qa/` directory:
   ```bash
   cd backend/qa
   cat > .env << EOF
   # ChromaDB
   CHROMA_API_KEY=your_chroma_api_key
   CHROMA_TENANT=your_tenant_id
   CHROMA_DATABASE=your_database_name
   
   # LLM (Gemini)
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key
   EOF
   ```

2. **Start the service:**
   ```bash
   docker compose up -d
   ```

3. **Check status:**
   ```bash
   docker compose ps
   docker compose logs -f
   ```

4. **Access the service:**
   - API: `http://localhost:8000`
   - Health check: `http://localhost:8000/health`

### Option 2: Local Environment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export CHROMA_API_KEY=your_chroma_api_key
   export CHROMA_TENANT=your_tenant_id
   export CHROMA_DATABASE=your_database_name
   export GEMINI_API_KEY=your_gemini_api_key
   export LLM_PROVIDER=gemini
   ```

   Or create a `.env` file in the `qa/` directory (or project root).

3. **Run the service:**
   ```bash
   python qa_service.py
   ```

   The service runs on `http://localhost:5000` by default (or `PORT` env var).

## Configuration

### Required Environment Variables

- **ChromaDB:**
  - `CHROMA_API_KEY` - ChromaDB Cloud API key
  - `CHROMA_TENANT` - ChromaDB tenant ID
  - `CHROMA_DATABASE` - ChromaDB database name

- **LLM:**
  - `GEMINI_API_KEY` - Google Gemini API key
  - `LLM_PROVIDER` - Set to `gemini` (default)

### Optional Environment Variables

- `PORT` - Server port (default: 5000 locally, 8000 in Docker)
- `FLASK_DEBUG` - Enable debug mode (default: false)
- `CONFIG_PATH` - Path to config.yaml (default: auto-detected)
- `RAG_TOP_K` - Number of chunks to retrieve (default: 20)
- `RERANK_MAX_LENGTH` - Max length for reranking (default: 512)
- `ENABLE_RERANKING` - Enable reranking (default: true)
- `ENABLE_VERIFICATION` - Enable verification (default: true)

### Config File

The service uses `config.yaml` for retrieval and reranking settings. It looks for:
1. `CONFIG_PATH` environment variable
2. `config.yaml` in the `qa/` directory
3. `config.yaml` in the project root (2 levels up)

## API Endpoints

- `GET /health` - Health check
- `POST /query` - Submit a query and get an answer
- `POST /retrieve` - Retrieve chunks without generating response
- `GET /collection/info` - Get ChromaDB collection information

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Apple revenue in 2023?",
    "filters": {"ticker": "AAPL", "year": "2023"},
    "top_k": 10
  }'
```

### Example Query with Verification

The verification system is enabled by default. The response will include a `verification` object:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Apple revenue in 2023?",
    "filters": {"ticker": "AAPL", "year": "2023"},
    "top_k": 10,
    "enable_verification": true
  }'
```

Response includes verification results:
```json
{
  "answer": "...",
  "sources": [...],
  "verification": {
    "overall_score": 0.85,
    "answer_source_alignment": 0.90,
    "citation_coverage": 0.75,
    "fact_verification_score": 0.80,
    "issues": [],
    "verified_sources": ["chunk_id_1", "chunk_id_2"],
    "unverified_claims": []
  }
}
```

To disable verification for a specific request:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Apple revenue in 2023?",
    "enable_verification": false
  }'
```

## Docker Commands

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Rebuild
docker compose up --build -d

# Check health
curl http://localhost:8000/health
```

## Testing Verification

### Automated Test Script

Run the verification test script to verify the system is working:

```bash
# Install requests if needed
pip install requests

# Run the test script
python test_verification.py
```

The script will:
- Check that the service is running
- Verify that verification results are included in responses
- Test that verification can be disabled
- Validate that scores are in the correct range [0.0, 1.0]

### Manual Testing

1. **Start the service** (if not already running):
   ```bash
   python qa_service.py
   # or
   docker compose up -d
   ```

2. **Make a query and check for verification**:
   ```bash
   curl -X POST http://localhost:5000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What was Apple revenue in 2023?"}' | jq '.verification'
   ```

3. **What to look for**:
   - `verification` object should be present in the response
   - `overall_score` should be between 0.0 and 1.0
   - `answer_source_alignment`, `citation_coverage`, and `fact_verification_score` should all be present
   - `issues` array may contain warnings if verification finds problems
   - `verified_sources` should list chunk IDs that support the answer
   - `unverified_claims` should list any claims that couldn't be verified

4. **Check service logs** for verification activity:
   ```bash
   # Look for log messages like:
   # "Verification completed. Overall score: 0.85"
   ```

## Troubleshooting

### Docker credential helper error
If you see `docker-credential-desktop not found`, remove credential helper from Docker config:
```bash
cat > ~/.docker/config.json << EOF
{
  "auths": {},
  "currentContext": "desktop-linux"
}
EOF
```

### Verification not appearing in responses

1. **Check configuration**: Ensure `verification.enable: true` in `config.yaml` or `ENABLE_VERIFICATION=true` in environment
2. **Check logs**: Look for verification errors in the service logs
3. **Test with explicit flag**: Set `"enable_verification": true` in the request body
4. **Check LLM availability**: Verification uses the LLM, so ensure your LLM API key is valid
