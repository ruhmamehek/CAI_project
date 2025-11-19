# Data Processing and ChromaDB Upload

This module processes SEC filing JSON files and uploads them to ChromaDB for vector search.

## What It Does

1. **Processes JSON files** from `data/layout-parser/`
   - Filters content by type (Text, List item, Table only)
   - Removes page headers and footers
   - Detects SEC Item sections (Item 1, Item 1A, Item 2, etc.)
   - Chunks text into 512-token segments
   - Extracts ticker and year from filenames

2. **Saves processed chunks** to `data/processed/`
   - Each chunk includes: text, ticker, year, item_number, and metadata

3. **Uploads to ChromaDB** (optional)
   - Replaces existing collection data
   - Generates embeddings using specified model
   - Stores chunks with all metadata

## Quick Start

### 1. Setup Environment Variables

Create a `.env` file in the project root:

```env
CHROMA_API_KEY=your_api_key
CHROMA_TENANT=your_tenant
CHROMA_DATABASE=your_database
```

### 2. Install Dependencies

```bash
pip install chromadb sentence-transformers python-dotenv transformers
```

### 3. Process and Upload

**Option A: Using the bash script (recommended)**
```bash
./scripts/process_and_upload.sh
```

**Option B: Using Python directly**
```bash
# Process all files and upload to ChromaDB
python backend/vectordb/data_processing.py data/layout-parser/ --upload-chromadb

# Process only (no upload)
python backend/vectordb/data_processing.py data/layout-parser/

# Process single file
python backend/vectordb/data_processing.py data/layout-parser/ --file data/layout-parser/jpm_2024.json --upload-chromadb
```

## Usage

### Process Files Only

```bash
python backend/vectordb/data_processing.py data/layout-parser/
```

Output: Processed chunks saved to `data/processed/`

### Process and Upload to ChromaDB

```bash
python backend/vectordb/data_processing.py data/layout-parser/ --upload-chromadb
```

This will:
- Process all JSON files
- Delete existing ChromaDB collection (if exists)
- Create new collection with embeddings
- Upload all chunks

### Command Line Options

```
--upload-chromadb          Upload processed chunks to ChromaDB
--collection-name NAME     ChromaDB collection name (default: sec_filings)
--embedding-model MODEL    Embedding model (default: BAAI/bge-base-en-v1.5)
--batch-size SIZE          Batch size for uploads (default: 100)
--file PATH                Process single file instead of directory
--verbose, -v              Enable verbose logging
```

### Example with Custom Options

```bash
python backend/vectordb/data_processing.py data/layout-parser/ \
  --upload-chromadb \
  --collection-name my_collection \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --batch-size 50 \
  --verbose
```

## Input Files

Place your layout-parser JSON files in `data/layout-parser/` with naming format:
- `{TICKER}_{YEAR}.json` (e.g., `jpm_2024.json`)
- `{TICKER}-{YEAR}.json` (e.g., `tsla-2024.json`)

Each file should contain an array of objects with:
- `type`: One of "Text", "List item", "Table", "Section header", etc.
- `text`: The text content

## Output Format

Processed chunks in `data/processed/` include:

```json
{
  "text": "Chunk text content...",
  "chunk_id": "JPM_2024_chunk_0",
  "ticker": "JPM",
  "year": "2024",
  "item_number": "1",
  "page_number": 19,
  "left": 23.0,
  "top": 129.0,
  ...
}
```

## ChromaDB Collection Structure

The collection is created with:
- **Embedding model**: `BAAI/bge-base-en-v1.5` (default, configurable)
- **Metadata fields**: ticker, year, item_number, page_number, doc_id, etc.
- **Documents**: Chunk text content
- **IDs**: Unique chunk IDs (format: `{ticker}_{year}_chunk_{index}`)

## Processing Details

### Chunking Strategy
- Chunks are limited to **512 tokens** (approximately)
- Uses GPT-2 tokenizer (falls back to word count approximation if unavailable)
- Aggregates consecutive text blocks until size limit

### Item Number Detection
Detects and tags chunks with SEC Item numbers:
- Item 1, 1A, 1B, 1C
- Item 2, 3, 4, 5, 6
- Item 7, 7A
- Item 8, 9, 9A, 9B, 9C
- Item 10-15

Item numbers are extracted from "Section header" type blocks and assigned to all subsequent chunks until the next Item header.

### Type Filtering
- **Included**: Text, List item, Table
- **Excluded**: Page header, Page footer
- **Special handling**: Section headers (processed for Item detection only)

## Troubleshooting

**Error: "ModuleNotFoundError: No module named 'chromadb'"**
```bash
pip install chromadb
```

**Error: "Collection already exists"**
- The script automatically deletes and recreates the collection when `--upload-chromadb` is used with `replace=True` (default)

**Warning: "Could not extract ticker and year"**
- Ensure filenames follow the format: `{TICKER}_{YEAR}.json` or `{TICKER}-{YEAR}.json`

**Files skipped with warning**
- Files that are already processed (have `item_number` but no `type` field) are skipped
- Use original layout-parser format files as input

## Files

- `data_processing.py`: Main processing script
- `data_upload.py`: ChromaDB upload functionality
- `scripts/process_and_upload.sh`: Convenience bash script

