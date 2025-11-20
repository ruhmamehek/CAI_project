"""Utility script to fix malformed chunk_ids list formatting in a FinDER CSV.

Some rows have a multi-line representation like:

    chunk_ids = [
        "MSFT_2024_chunk_192"
        "MSFT_2024_chunk_193"
        "MSFT_2024_chunk_194"
    ]

Which is not valid JSON/Python list syntax (missing commas). This script extracts the quoted
IDs and rewrites the cell as a proper JSON-style list string:

    ["MSFT_2024_chunk_192","MSFT_2024_chunk_193","MSFT_2024_chunk_194"]

Usage:
    PYTHONPATH=$(pwd) python evaluation/clean_chunk_ids.py \
        --input /path/to/raw.csv \
        --output /path/to/cleaned.csv

It preserves all other columns and only modifies cells that appear malformed AND yield a
non-empty list of extracted IDs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List

import pandas as pd


MULTILINE_PATTERN = re.compile(r"chunk_ids\s*=\s*\[", re.IGNORECASE)
QUOTED_ID_PATTERN = re.compile(r'"([^"\n\r]+)"')


def extract_ids(cell: str) -> List[str]:
    """Return list of quoted IDs if cell matches malformed multi-line pattern, else []."""
    if not isinstance(cell, str):
        return []
    if not MULTILINE_PATTERN.search(cell):
        return []
    ids = QUOTED_ID_PATTERN.findall(cell)
    # Filter out stray bracket tokens
    ids = [i.strip() for i in ids if i.strip() and i.strip() not in ("]", "[")]
    return ids


def normalize_cell(cell: str) -> str:
    """Normalize a chunk_ids cell; if multi-line malformed, rewrite as JSON list string."""
    ids = extract_ids(cell)
    if ids:
        return "[" + ",".join(json.dumps(i) for i in ids) + "]"  # ensure proper quoting
    return cell


def main(input_csv: str, output_csv: str, column_name: str | None) -> None:
    df = pd.read_csv(input_csv)
    # Auto-detect column if not provided
    if column_name is None:
        candidates = ["chunk_ids", "chunk ids", "chunk_id_list"]
        for c in candidates:
            if c in df.columns:
                column_name = c
                break
        if column_name is None:
            # heuristic fallback
            for c in df.columns:
                cl = str(c).lower()
                if "chunk" in cl and "id" in cl:
                    column_name = c
                    break
    if column_name is None or column_name not in df.columns:
        raise RuntimeError("Could not locate chunk_ids column; specify --column explicitly.")

    modified = 0
    for idx, val in df[column_name].items():
        try:
            new_val = normalize_cell(val)
        except Exception:
            new_val = val
        if new_val != val:
            df.at[idx, column_name] = new_val
            modified += 1

    df.to_csv(output_csv, index=False)
    print(f"Wrote cleaned CSV to {output_csv}. Modified {modified} cells in column '{column_name}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fix malformed chunk_ids list formatting in FinDER CSV.")
    p.add_argument("--input", required=True, help="Path to the raw CSV.")
    p.add_argument("--output", required=True, help="Path to write the cleaned CSV.")
    p.add_argument("--column", required=False, default=None, help="Optional explicit chunk_ids column name.")
    args = p.parse_args()
    try:
        main(args.input, args.output, args.column)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
