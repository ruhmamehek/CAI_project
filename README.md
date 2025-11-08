# Explainable & Faithful RAG for Financial QA

**Citation-Disciplined Answers with Span-Level Verification**

This project implements a retrieval-augmented conversational assistant for finance that generates short, citation-disciplined answers grounded in SEC filings and FOMC texts.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/              # Raw SEC filings, FOMC texts
│   ├── processed/        # Chunked and indexed documents
│   └── qa/               # QA datasets
├── src/
│   └── data/
│       ├── acquisition.py    # SEC/FOMC data collection
│       └── processing.py      # Parsing, chunking
└── scripts/
    └── process_data.py        # Data processing script
```

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure settings**:
Edit `config.yaml` with your preferences.

3. **Process data**:
```bash
python scripts/process_data.py
```

## Next Steps

- Implement retrieval system
- Build generation pipeline
- Add verification layer
- Set up evaluation framework
