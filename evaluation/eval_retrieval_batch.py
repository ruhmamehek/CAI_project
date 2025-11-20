"""
Batch evaluator for retrieval quality using chunk IDs.

For each eligible row in a FinDER CSV, it:
- loads the question and gold chunk IDs
- calls the QA service /retrieve (and /query for answer, optional)
- extracts retrieved chunk IDs from the response
- computes ID-based metrics: hit@k, precision@k, recall@k, mrr@k, nDCG@k
- skips rows with empty gold chunk IDs

Outputs per-row metrics and aggregates (macro means; micro precision/recall).

Usage (from repo root):
  PYTHONPATH=$(pwd) python evaluation/eval_retrieval_batch.py \
    --csv "/path/to/finder.csv" \
    --base-url http://localhost:5001 \
    --top-k 10 \
    --start 0 --limit 50 \
    --output-csv evaluation/batch_metrics.csv \
    --output-json evaluation/batch_summary.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from evaluation.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    ndcg_at_k,
)


@dataclass
class RowData:
    qid: str
    question: str
    gold_chunk_ids: List[str]


def _parse_id_list(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x) for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    # Handle malformed multi-line 'chunk_ids = [' blocks (missing commas)
    if 'chunk_ids' in s and '[' in s and ']' in s and '\n' in s:
        import re
        ids = re.findall(r'"([^"\n\r]+)"', s)
        ids = [i.strip() for i in ids if i.strip() and i.strip() not in (']', '[')]
        if ids:
            return ids
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, (list, tuple, set)):
            return [str(x) for x in lit if str(x).strip()]
    except Exception:
        pass
    for delim in [",", "|", ";"]:
        if delim in s:
            parts = [p.strip() for p in s.split(delim)]
            return [x for x in parts if x]
    return [t for t in s.split() if t]


def _find_gold_ids_column(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "chunk_ids",
        "chunk ids",
        "gold_chunk_ids",
        "reference_chunk_ids",
        "chunk_id_list",
    ]
    cols = [c for c in df.columns]
    for c in preferred:
        if c in cols:
            return c
    # heuristic fallback
    for c in cols:
        cl = str(c).lower()
        if "chunk" in cl and "id" in cl:
            return c
    return None


def _load_row(df: pd.DataFrame, idx: int, gold_col: str) -> Optional[RowData]:
    r = df.iloc[idx]
    gold_ids = _parse_id_list(r.get(gold_col))
    if not gold_ids:
        return None
    qid = str(r.get("_id", idx))
    question = str(r.get("text", "")).strip()
    return RowData(qid=qid, question=question, gold_chunk_ids=gold_ids)


def post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Tuple[int, Dict[str, Any]]:
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    return resp.status_code, data


def _extract_chunk_id_from_item(item: Dict[str, Any]) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    for key in ("chunk_id", "chunkId", "id", "_id"):
        if key in item and str(item[key]).strip():
            return str(item[key]).strip()
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else None
    if meta:
        for key in ("chunk_id", "chunkId", "id", "_id"):
            if key in meta and str(meta[key]).strip():
                return str(meta[key]).strip()
    return None


def _extract_retrieved_chunk_ids(retrieve_data: Dict[str, Any]) -> List[str]:
    if not isinstance(retrieve_data, dict):
        return []
    candidates = []
    for list_key in ("chunks", "results", "retrieved", "matches", "documents"):
        if list_key in retrieve_data and isinstance(retrieve_data[list_key], list):
            candidates = retrieve_data[list_key]
            break
    if not candidates and isinstance(retrieve_data.get("data"), list):
        candidates = retrieve_data["data"]
    ids: List[str] = []
    seen = set()
    for item in candidates:
        cid = _extract_chunk_id_from_item(item)
        if cid and cid not in seen:
            ids.append(cid)
            seen.add(cid)
    return ids


def evaluate_rows(
    csv_path: str,
    base_url: str,
    top_k: int,
    k_list: Optional[List[int]],
    start: int,
    limit: Optional[int],
    filters: Optional[str],
    sleep: float,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    gold_col = _find_gold_ids_column(df)
    if not gold_col:
        raise RuntimeError("Could not find a gold chunk IDs column in the CSV.")

    # Optional filters
    filters_obj: Optional[Dict[str, Any]] = None
    if filters:
        try:
            filters_obj = json.loads(filters)
        except Exception as e:
            print(f"Warning: failed to parse filters JSON: {e}")

    # Iterate rows
    rows_total = len(df)
    i = max(0, start)
    processed = 0
    limit = limit if (limit is not None and limit >= 0) else None

    per_row: List[Dict[str, Any]] = []
    # Decide the list of ks to compute; ensure positive ints and sorted unique
    if k_list:
        ks = sorted({int(k) for k in k_list if isinstance(k, int) and k > 0})
    else:
        ks = [int(top_k)] if top_k > 0 else [10]

    max_k = max(ks) if ks else top_k

    # Micro aggregation counters per k
    tp_sum: Dict[int, int] = {k: 0 for k in ks}
    denom_precision: Dict[int, int] = {k: 0 for k in ks}  # add k per processed query
    total_relevant = 0  # denominator for micro recall is independent of k

    while i < rows_total and (limit is None or processed < limit):
        row = _load_row(df, i, gold_col)
        if row is None:
            i += 1
            continue  # skip empty gold ids

        # Retrieve once with the maximum required top_k to support all ks
        payload = {"query": row.question, "top_k": max_k, "filters": filters_obj}
        retrieve_url = f"{base_url.rstrip('/')}/retrieve"
        try:
            rc, retrieve_data = post_json(retrieve_url, payload)
        except requests.exceptions.RequestException as e:
            print(f"Row {i} request failed: {e}")
            i += 1
            continue
        if rc != 200:
            print(f"Row {i} retrieve HTTP {rc}, skipping")
            i += 1
            continue

        retrieved_ids = _extract_retrieved_chunk_ids(retrieve_data or {})
        if not retrieved_ids:
            print(f"Row {i} no retrieved IDs; skipping")
            i += 1
            continue

        gold_set = set(row.gold_chunk_ids)

        # counts for micro aggregation per k
        for k in ks:
            k_eff = min(k, len(retrieved_ids))
            tp_k = sum(1 for rid in retrieved_ids[:k_eff] if rid in gold_set)
            tp_sum[k] += tp_k
            denom_precision[k] += k  # use requested k as denominator per our convention
        total_relevant += len(gold_set)

        # metrics for each k
        row_metrics: Dict[str, Any] = {
            "row_index": i,
            "qid": row.qid,
            "gold_count": len(gold_set),
            "retrieved_count": len(retrieved_ids),
            # Store the full retrieved list (up to max_k) for transparency
            "retrieved_chunk_ids": retrieved_ids[:max_k],
            "gold_chunk_ids": row.gold_chunk_ids,
        }
        for k in ks:
            k_eff = min(k, len(retrieved_ids))
            row_metrics[f"hit@{k}"] = hit_at_k(retrieved_ids, gold_set, k)
            row_metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, gold_set, k)
            row_metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, gold_set, k)
            row_metrics[f"mrr@{k}"] = mrr_at_k(retrieved_ids, gold_set, k)
            row_metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, gold_set, k)
        per_row.append(row_metrics)

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed} rows (up to CSV idx {i}).")
        i += 1
        if sleep > 0:
            time.sleep(sleep)

    if not per_row:
        return {
            "count": 0,
            "per_row": [],
            "macro": {},
            "micro": {},
        }

    # Macro averages (mean across rows) for each k
    macro: Dict[str, Dict[str, float]] = {}
    for k in ks:
        def _mean_k(key: str) -> float:
            vals = [r.get(f"{key}@{k}", 0.0) for r in per_row]
            return float(sum(vals)) / float(len(vals)) if vals else 0.0
        macro[str(k)] = {
            "hit": _mean_k("hit"),
            "precision": _mean_k("precision"),
            "recall": _mean_k("recall"),
            "mrr": _mean_k("mrr"),
            "ndcg": _mean_k("ndcg"),
        }

    # Micro precision/recall per k
    micro: Dict[str, Dict[str, float]] = {}
    for k in ks:
        micro_precision_k = (tp_sum[k] / float(denom_precision[k])) if denom_precision[k] > 0 else 0.0
        micro_recall_k = (tp_sum[k] / float(total_relevant)) if total_relevant > 0 else 0.0
        micro[str(k)] = {
            "precision": micro_precision_k,
            "recall": micro_recall_k,
        }

    return {
        "count": len(per_row),
        "per_row": per_row,
        "ks": ks,
        "macro": macro,
        "micro": micro,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch retrieval evaluator using chunk IDs.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the FinDER CSV file.")
    parser.add_argument("--base-url", type=str, default="http://localhost:5001", help="QA service base URL.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval (used when --k-list is not provided).")
    parser.add_argument("--k-list", type=str, default=None, help="Comma-separated list of k values for @k metrics, e.g. '5,10,20'. Retrieval is done once with max(k).")
    parser.add_argument("--start", type=int, default=0, help="Start row index (inclusive).")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of rows to evaluate (skipping empties).")
    parser.add_argument("--filters", type=str, default=None, help="Optional JSON string for filters.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to write per-row metrics CSV.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write summary JSON.")
    parser.add_argument("--augment-csv", type=str, default=None, help="Optional path to write the original CSV augmented with per-row metrics and retrieved IDs (new columns).")
    args = parser.parse_args()

    try:
        # Parse k-list
        if args.k_list:
            try:
                k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
                k_list = [k for k in k_list if k > 0]
            except Exception:
                k_list = None
                print("Warning: failed to parse --k-list; falling back to --top-k.")
        else:
            k_list = None

        result = evaluate_rows(
            csv_path=args.csv,
            base_url=args.base_url,
            top_k=args.top_k,
            k_list=k_list,
            start=args.start,
            limit=args.limit,
            filters=args.filters,
            sleep=args.sleep,
        )
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print summary
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in result.items() if k != "per_row"}, indent=2))

    # Optionally write files
    if args.output_csv and result.get("per_row"):
        pd.DataFrame(result["per_row"]).to_csv(args.output_csv, index=False)
        print(f"Wrote per-row metrics to {args.output_csv}")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "per_row"}, f, indent=2)
        print(f"Wrote summary to {args.output_json}")

    # Augment original CSV with new columns, leaving non-evaluated rows untouched
    if args.augment_csv:
        try:
            df = pd.read_csv(args.csv)
            # Ensure columns exist
            ks = result.get("ks", []) or []
            new_cols = ["retrieved_count", "retrieved_chunk_ids", "gold_chunk_ids"]
            for k in ks:
                new_cols += [
                    f"retrieval_hit@{k}",
                    f"retrieval_precision@{k}",
                    f"retrieval_recall@{k}",
                    f"retrieval_mrr@{k}",
                    f"retrieval_ndcg@{k}",
                ]
            for c in new_cols:
                if c not in df.columns:
                    df[c] = None
            # Fill values for processed rows
            for r in result.get("per_row", []):
                idx = r["row_index"]
                if 0 <= idx < len(df):
                    df.at[idx, "retrieved_count"] = r.get("retrieved_count")
                    for k in ks:
                        df.at[idx, f"retrieval_hit@{k}"] = r.get(f"hit@{k}")
                        df.at[idx, f"retrieval_precision@{k}"] = r.get(f"precision@{k}")
                        df.at[idx, f"retrieval_recall@{k}"] = r.get(f"recall@{k}")
                        df.at[idx, f"retrieval_mrr@{k}"] = r.get(f"mrr@{k}")
                        df.at[idx, f"retrieval_ndcg@{k}"] = r.get(f"ndcg@{k}")
                    # Store IDs as JSON strings to avoid delimiter ambiguity
                    df.at[idx, "retrieved_chunk_ids"] = json.dumps(r.get("retrieved_chunk_ids", []))
                    df.at[idx, "gold_chunk_ids"] = json.dumps(r.get("gold_chunk_ids", []))
            df.to_csv(args.augment_csv, index=False)
            print(f"Wrote augmented CSV to {args.augment_csv}")
        except Exception as e:
            print(f"Failed to write augmented CSV: {e}")


if __name__ == "__main__":
    main()
