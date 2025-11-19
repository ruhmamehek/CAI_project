#!/bin/bash

# Script to process all JSON files in data/layout-parser and upload to ChromaDB

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
INPUT_DIR="$PROJECT_ROOT/data/layout-parser"
COLLECTION_NAME="sec_filings"
EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
BATCH_SIZE=100
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --collection-name)
            COLLECTION_NAME="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Process all JSON files in data/layout-parser and upload to ChromaDB"
            echo ""
            echo "Options:"
            echo "  --input-dir DIR          Input directory (default: data/layout-parser)"
            echo "  --collection-name NAME   ChromaDB collection name (default: sec_filings)"
            echo "  --embedding-model MODEL  Embedding model (default: BAAI/bge-base-en-v1.5)"
            echo "  --batch-size SIZE        Batch size for uploads (default: 100)"
            echo "  --verbose, -v            Enable verbose logging"
            echo "  --help, -h               Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. ChromaDB upload may fail.${NC}"
    echo "Make sure you have CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE set."
fi

# Count JSON files
JSON_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.json" | wc -l | tr -d ' ')

if [ "$JSON_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No JSON files found in $INPUT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Processing and uploading documents to ChromaDB${NC}"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Collection name: $COLLECTION_NAME"
echo "Embedding model: $EMBEDDING_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Files found: $JSON_COUNT"
echo ""

# Build command
CMD="python3 backend/vectordb/data_processing.py \"$INPUT_DIR\" --upload-chromadb --collection-name \"$COLLECTION_NAME\" --embedding-model \"$EMBEDDING_MODEL\" --batch-size $BATCH_SIZE"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# Run the command
echo -e "${GREEN}Starting processing...${NC}"
echo ""

if eval "$CMD"; then
    echo ""
    echo -e "${GREEN}✓ Successfully processed and uploaded all documents to ChromaDB${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Error occurred during processing${NC}"
    exit 1
fi

