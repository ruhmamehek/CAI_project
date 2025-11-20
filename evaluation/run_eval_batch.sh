#!/bin/bash
# Batch evaluation script for retrieval quality with k values 5, 10, 15, 20, 25
# 
# Usage:
#   ./evaluation/run_eval_batch.sh /path/to/finder.csv
#   ./evaluation/run_eval_batch.sh /path/to/finder.csv --base-url http://localhost:5001
#   ./evaluation/run_eval_batch.sh /path/to/finder.csv --limit 100

set -e  # Exit on error

# Default CSV path (can be overridden by first argument)
DEFAULT_CSV_PATH="evaluation/exports/finder_train_no_reasoning_subset.csv"

# Get the CSV path from first argument if provided, otherwise use default
if [ -n "${1:-}" ]; then
    CSV_PATH="$1"
    shift
else
    CSV_PATH="$DEFAULT_CSV_PATH"
fi

# Default values
BASE_URL="${BASE_URL:-http://localhost:8000}"
START="${START:-0}"
LIMIT="${LIMIT:-50}"
K_LIST="5,10,15,20,25"
OUTPUT_CSV="${OUTPUT_CSV:-evaluation/batch_metrics.csv}"
OUTPUT_JSON="${OUTPUT_JSON:-evaluation/batch_summary.json}"
AUGMENT_CSV="${AUGMENT_CSV:-evaluation/augmented_results.csv}"
CALL_SLEEP="${CALL_SLEEP:-0.5}"

# Parse additional arguments
FILTERS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --start)
            START="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --filters)
            FILTERS="$2"
            shift 2
            ;;
        --output-csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --output-json)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        --augment-csv)
            AUGMENT_CSV="$2"
            shift 2
            ;;
        --call-sleep)
            CALL_SLEEP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get the project root directory (parent of evaluation/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Check if CSV file exists
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found: $CSV_PATH"
    exit 1
fi

# Build the command as an array to handle spaces in paths properly
PYTHONPATH="$(pwd)"
CMD_ARGS=(
    "evaluation/eval_retrieval_batch.py"
    "--csv" "$CSV_PATH"
    "--base-url" "$BASE_URL"
    "--k-list" "$K_LIST"
    "--start" "$START"
    "--limit" "$LIMIT"
    "--output-csv" "$OUTPUT_CSV"
    "--output-json" "$OUTPUT_JSON"
    "--augment-csv" "$AUGMENT_CSV"
    "--call-sleep" "$CALL_SLEEP"
)

if [ -n "$FILTERS" ]; then
    CMD_ARGS+=("--filters" "$FILTERS")
fi

# Print the command
echo "Running evaluation with k values: $K_LIST"
echo "CSV file: $CSV_PATH"
echo "Command: PYTHONPATH=\"$PYTHONPATH\" python ${CMD_ARGS[*]}"
echo ""

# Execute the command
PYTHONPATH="$PYTHONPATH" python "${CMD_ARGS[@]}"

echo ""
echo "Evaluation complete!"
echo "  - Per-row metrics: $OUTPUT_CSV"
echo "  - Summary JSON: $OUTPUT_JSON"
echo "  - Augmented CSV: $AUGMENT_CSV"

