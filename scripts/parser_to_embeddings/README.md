# Parser to Embeddings Pipeline

This directory contains scripts for processing PDF documents, extracting tables/figures, generating captions, and preparing data for RAG vector database upload.

## Workflow Overview

```
PDF → Extract Chunks → Extract Clippings → Generate Captions → Prepare Unified Chunks → Upload to ChromaDB
```

## Step-by-Step Guide

### 1. Extract Chunks from PDF

Extract text chunks with layout information (Text, Table, Figure, etc.):

```bash
python scripts/parser_to_embeddings/extract_chunks.py \
  --pdf data/sec_pdfs/tsla-20231231.pdf \
  --output data/sec_pdfs/parsed/tsla-20231231_chunks.json \
  --start-page 60 \
  --end-page 64
```

**Output:** JSON file with text chunks, coordinates, and layout types.

### 2. Extract Table/Figure Clippings

Extract image clippings for tables and figures:

```bash
python scripts/parser_to_embeddings/extract_clippings_from_parsed.py \
  --pdf data/sec_pdfs/tsla-20231231-60-64.pdf \
  --parsed data/sec_pdfs/parsed/tsla-20231-60-64.json \
  --output data/sec_pdfs/clippings \
  --page-offset 0
```

**Output:** 
- Image files: `data/sec_pdfs/clippings/{pdf_name}/clippings/*.png`
- Metadata: `clippings_metadata.json`

### 3. Generate Captions for Tables/Figures

**YES, you should generate captions!** This makes tables/figures searchable in your RAG system.

Use Gemini vision model to generate descriptions:

```bash
python scripts/parser_to_embeddings/generate_captions.py \
  --clippings-metadata data/sec_pdfs/clippings/tsla-20231231-60-64/clippings/clippings_metadata.json \
  --clippings-dir data/sec_pdfs/clippings/tsla-20231231-60-64/clippings \
  --chunks-json data/sec_pdfs/parsed/tsla-20231231_chunks.json \
  --output data/sec_pdfs/clippings/tsla-20231231-60-64/clippings/enriched_metadata.json \
  --model gemini-1.5-flash
```

**Requirements:**
- Set `GEMINI_API_KEY` environment variable
- Or pass `--api-key` argument

**Output:** `enriched_metadata.json` with captions for each table/figure.

### 4. Prepare Unified Chunks

Combine text chunks and table/figure chunks into a unified format:

```bash
python scripts/parser_to_embeddings/prepare_multimodal_chunks.py \
  --text-chunks data/sec_pdfs/parsed/tsla-20231231_chunks.json \
  --enriched-clippings data/sec_pdfs/clippings/tsla-20231231-60-64/clippings/enriched_metadata.json \
  --output data/processed/unified_chunks.json \
  --ticker TSLA \
  --filing-type "10-K" \
  --year 2023 \
  --accession-number "000162828024002390" \
  --base-path data/sec_pdfs
```

**Output:** Unified JSON with both text and table/figure chunks, ready for ChromaDB upload.

### 5. Upload to ChromaDB

Upload unified chunks to your vector database:

```bash
python backend/vectordb/data_upload.py \
  --chunks-file data/processed/unified_chunks.json \
  --collection-name sec_filings \
  --embedding-model "BAAI/bge-base-en-v1.5" \
  --batch-size 100
```

**Requirements:**
- Set environment variables: `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`

## Why Generate Captions?

**Benefits:**
1. **Searchability**: Tables/figures become searchable via their captions
2. **Context**: LLM can understand what tables/figures contain without seeing the image
3. **Better Retrieval**: Semantic search works on text descriptions
4. **RAG Integration**: Captions are embedded and retrieved just like text chunks

**Example:**
- Without caption: Table image is not searchable
- With caption: "Primary Manufacturing Facilities table showing locations in Austin, Fremont, Nevada, Germany, China, New York, and California with ownership status"

## Data Structure

### Text Chunk
```json
{
  "chunk_id": "text_001_000",
  "text": "Our cybersecurity risk management...",
  "type": "Text",
  "page": 60,
  "ticker": "TSLA",
  "filing_type": "10-K",
  "year": "2023"
}
```

### Table/Figure Chunk
```json
{
  "chunk_id": "table_001_000",
  "text": "TITLE: Primary Manufacturing Facilities\nDESCRIPTION: Table showing...",
  "type": "Table",
  "page": 60,
  "image_path": "data/sec_pdfs/clippings/.../page_001_table_000.png",
  "filename": "page_001_table_000.png",
  "ticker": "TSLA",
  "filing_type": "10-K",
  "year": "2023"
}
```

## RAG Integration

When retrieving chunks, the RAG service now returns:

```json
{
  "chunk_id": "table_001_000",
  "text": "TITLE: Primary Manufacturing Facilities...",
  "chunk_type": "Table",
  "page": 60,
  "image_path": "data/sec_pdfs/clippings/.../page_001_table_000.png",
  "score": 0.85
}
```

Your frontend can:
1. Display text chunks normally
2. Display table/figure chunks with the image from `image_path`
3. Show page numbers for context

## Tips

1. **Batch Processing**: Process multiple PDFs in a loop
2. **Error Handling**: Some images may fail caption generation - they'll be skipped
3. **Context**: Providing chunks JSON to caption generation adds context from surrounding text
4. **Storage**: Store images in a location accessible to your RAG service
5. **Metadata**: Ensure all document metadata (ticker, year, etc.) is consistent

## Troubleshooting

**Issue: Gemini API errors**
- Check API key is set correctly
- Verify API quota/limits
- Try a different model (e.g., `gemini-1.5-pro`)

**Issue: Image paths not found**
- Ensure `--base-path` is set correctly in step 4
- Use absolute paths or paths relative to your application root

**Issue: ChromaDB upload fails**
- Check metadata values are strings/numbers (not complex objects)
- Verify collection name and credentials

