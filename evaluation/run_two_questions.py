#!/usr/bin/env python
"""
Run the QA service for exactly two questions from a FinDER-style CSV.

Usage (from repo root):
  PYTHONPATH=$(pwd) python evaluation/run_two_questions.py \
      --csv evaluation/exports/finder_train_no_reasoning_subset.csv \
      --rows 0 1 \
      --base-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import pandas as pd
import requests


def load_questions(csv_path: str, rows: List[int]) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    questions: List[Dict[str, Any]] = []
    for idx in rows:
        if idx < 0 or idx >= len(df):
            raise IndexError(f"Row index {idx} out of range 0..{len(df) - 1}")
        row = df.iloc[idx]
        questions.append(
            {
                "row": idx,
                "qid": str(row.get("_id", idx)),
                "question": str(row.get("text", "")).strip(),
                "answer": str(row.get("answer", "")).strip(),
            }
        )
    return questions


def post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        return resp.json()
    except Exception:
        return {"status": resp.status_code, "raw": resp.text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QA service for exactly two questions.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the FinDER CSV file (default subset works well).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs=2,
        default=[0, 1],
        help="Two row indices to evaluate (default: 0 1).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="QA service base URL (default: http://localhost:8000).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K value for retrieval (default: 10).")
    parser.add_argument(
        "--filters",
        type=str,
        default=None,
        help='Optional JSON string for filters, e.g. \'{"ticker":"MSFT"}\'.',
    )
    parser.add_argument(
        "--call-sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between the two calls (default: 0.5).",
    )
    args = parser.parse_args()

    try:
        questions = load_questions(args.csv, list(args.rows))
    except Exception as exc:
        print(f"Failed to load questions: {exc}")
        sys.exit(1)

    filters_obj = None
    if args.filters:
        try:
            filters_obj = json.loads(args.filters)
        except Exception as exc:
            print(f"Warning: failed to parse filters JSON: {exc}")

    query_url = f"{args.base_url.rstrip('/')}/query"

    for idx, item in enumerate(questions):
        payload = {
            "query": item["question"],
            "top_k": args.top_k,
        }
        if filters_obj is not None:
            payload["filters"] = filters_obj

        print("=" * 60)
        print(f"Row {item['row']} (QID: {item['qid']})")
        print(f"Question: {item['question']}")
        print(f"Calling {query_url}")
        response = post_json(query_url, payload)
        print("Response:")
        print(json.dumps(response, indent=2))

        if idx == 0 and args.call_sleep > 0:
            import time

            time.sleep(args.call_sleep)


if __name__ == "__main__":
    main()

