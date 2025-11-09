# Explainable & Faithful RAG for Financial QA

**Citation-Disciplined Answers with Span-Level Verification**

This project implements a retrieval-augmented conversational assistant for finance that generates short, citation-disciplined answers grounded in SEC filings and FOMC texts.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Settings

Edit `config.yaml` with your preferences:
- **Email**: Required for SEC EDGAR downloads (update `data.sec.email`)
- **Companies**: Ticker symbols to download (update `data.sec.companies`)
- **Years**: Years of filings to download (update `data.sec.years`)
- **Device**: Set to `"auto"` to auto-detect best device (CUDA/MPS/CPU)

## Data Pipeline

### Step 1: Download SEC Filings

Download SEC filings (10-K, 10-Q) for specified companies:

```bash
python scripts/process_data.py --config config.yaml --download
```

**What it does:**
- Downloads SEC filings from EDGAR database
- Saves to `data/sec-edgar-filings/`
- Files are organized by company ticker and filing type

**Configuration:**
- Update `config.yaml` with your email (required by SEC)
- Set companies: `["AAPL", "MSFT", "GOOGL"]`
- Set years: `[2022, 2023, 2024]`

### Step 2: Process and Chunk Documents

Process raw filings into narrative chunks and extract tables:

```bash
python scripts/process_data.py --config config.yaml
```

**What it does:**
- Parses SEC filing HTML/XML with a section-aware preprocessor derived from Unstructured’s SEC pipeline
- Separates narrative text from tabular content
- Chunks narrative sections into overlapping segments (configurable size/overlap)
- Saves clean text chunks and structured tables to `data/processed/`

**Output:**
- `data/processed/sec_chunks.json` — Narrative chunks with section metadata
- `data/processed/sec_tables.json` — Extracted tables (plain text plus HTML when available)

### Step 3: Build FAISS Index

Build a searchable index from processed chunks:

```bash
python scripts/build_index.py --config config.yaml
```

**What it does:**
- Loads processed chunks from `data/processed/sec_chunks.json`
- Encodes chunks using BGE embedding model (`BAAI/bge-base-en-v1.5`)
- Builds FAISS index for fast similarity search
- Saves index to `data/indices/`

**Output:**
- `data/indices/index.faiss` - FAISS index file
- `data/indices/chunks.json` - Chunk metadata

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── sec-edgar-filings/     # Raw SEC filings (downloaded)
│   ├── processed/             # Processed chunks (generated)
│   └── indices/                # FAISS indices (generated)
├── src/
│   ├── data/
│   │   ├── acquisition.py      # SEC/FOMC data download
│   │   └── processing.py       # Document parsing & chunking
│   └── retrieval/
│       └── dense_retriever.py  # Dense retrieval with FAISS
└── scripts/
    ├── process_data.py         # Download & process data
    └── build_index.py          # Build FAISS index
```

- [ ] Add reranking module
- [ ] Implement generation pipeline with citations
- [ ] Add NLI verification layer
- [ ] Build evaluation framework

## Attribution

Portions of the SEC parsing utilities in `src/prepline_sec_filings/` are adapted from [Unstructured-IO/pipeline-sec-filings](https://github.com/Unstructured-IO/pipeline-sec-filings) (commit `babc6430e36c76c21a9c963ceda4867c9b5d28a9`), released under the Apache License 2.0.  
Our preprocessing approach is informed by the section-aware pipeline described in *A Scalable Data-Driven Framework for Systematic Analysis of SEC 10-K Filings Using Large Language Models* ([arXiv:2409.17581](https://arxiv.org/abs/2409.17581)).

