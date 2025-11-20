"""Simple one-sample evaluator for the local QA service against the FinDER CSV subset.

Usage:
  # Default base URL http://localhost:5001, first row in the CSV
  python evaluation/test_rag_on_finder.py \
	--csv evaluation/exports/finder_train_no_reasoning_subset.csv

  # Specify a different row index and service URL
  python evaluation/test_rag_on_finder.py \
	--csv evaluation/exports/finder_train_no_reasoning_subset.csv \
	--row 0 \
	--base-url http://localhost:5001

Notes:
  - This script calls the QA service endpoints:
	  POST /query                  ==> gets model answer + sources
	  POST /retrieve               ==> baseline retrieve (no rerank)
	  POST /retrieve/rerank        ==> retrieve with reranking
	  POST /retrieve/filter-rerank ==> retrieve with filter reasoning + reranking
  - It then compares (per retrieval method):
	  (a) model answer vs. gold answer (token overlap)
	  (b) retrieved chunks vs. gold reference text (token overlap)
  - No external NLP deps required; overlap is a simple Jaccard metric over tokens.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ID- and Text-based retrieval metrics
from evaluation.metrics import (
	hit_at_k,
	precision_at_k,
	recall_at_k,
	mrr_at_k,
	ndcg_at_k,
	text_hit_at_k,
	text_precision_at_k,
	text_recall_at_k,
	text_mrr_at_k,
	text_ndcg_at_k,
)


def _normalize(text: str) -> List[str]:
	"""Very naive tokenization + normalization.

	Lowercase, strip, split on whitespace, remove trivial punctuation.
	"""
	if text is None:
		return []
	t = text.lower().strip()
	# lightweight punctuation removal
	for ch in ",.;:()[]{}<>\"'`!?/\\|#*$^~":
		t = t.replace(ch, " ")
	return [tok for tok in t.split() if tok]


def jaccard(a: str, b: str) -> float:
	"""Jaccard overlap between token sets of two strings."""
	A, B = set(_normalize(a)), set(_normalize(b))
	if not A and not B:
		return 1.0
	if not A or not B:
		return 0.0
	inter = len(A & B)
	union = len(A | B)
	return inter / union if union else 0.0


@dataclass
class Sample:
	qid: str
	question: str
	gold_answer: str
	gold_references: List[str]
	gold_chunk_ids: List[str]


def _parse_id_list(val: Any) -> List[str]:
	"""Parse a list of IDs from various CSV cell formats.

	Supports: Python-list strings (e.g., "['id1','id2']"), JSON-like, or delimited strings.
	Returns a list of non-empty string IDs.
	"""
	if val is None:
		return []
	# Already a list-like
	if isinstance(val, (list, tuple, set)):
		return [str(x) for x in val if str(x).strip()]
	# Special handling for malformed multi-line 'chunk_ids = [' blocks missing commas
	if isinstance(val, str) and 'chunk_ids' in val and '[' in val and ']' in val and '\n' in val:
		import re, json
		ids = re.findall(r'"([^"\n\r]+)"', val)
		ids = [i.strip() for i in ids if i.strip() and i.strip() not in (']', '[')]
		if ids:
			return ids
	s = str(val).strip()
	if not s:
		return []
	# Try literal_eval for list-ish strings
	try:
		lit = ast.literal_eval(s)
		if isinstance(lit, (list, tuple, set)):
			return [str(x) for x in lit if str(x).strip()]
	except Exception:
		pass
	# Fallback: split on common delimiters
	parts = []
	for delim in [",", "|", ";"]:
		if delim in s:
			parts = [p.strip() for p in s.split(delim)]
			break
	if not parts:
		parts = s.split()  # whitespace
	return [p for p in parts if p]


def load_sample(csv_path: str, row: int) -> Sample:
	df = pd.read_csv(csv_path)
	if row < 0 or row >= len(df):
		raise IndexError(f"Row index {row} out of range 0..{len(df)-1}")
	r = df.iloc[row]

	# references column is a string representation of a Python list
	refs_raw = r.get("references", "[]")
	try:
		refs = ast.literal_eval(refs_raw) if isinstance(refs_raw, str) else list(refs_raw)
	except Exception:
		refs = [str(refs_raw)]

	# Find a column that carries chunk IDs for the gold reference
	# Common variants: 'chunk_ids', 'chunk ids', 'chunk_id_list'
	candidates = [
		"chunk_ids",
		"chunk ids",
		"gold_chunk_ids",
		"reference_chunk_ids",
		"chunk_id_list",
	]
	gold_ids: List[str] = []
	for col in candidates:
		if col in r.index:
			gold_ids = _parse_id_list(r.get(col))
			break
	if not gold_ids:
		# heuristic: any column name containing both 'chunk' and 'id'
		for col in r.index:
			cl = str(col).lower()
			if "chunk" in cl and "id" in cl:
				gold_ids = _parse_id_list(r.get(col))
				if gold_ids:
					break

	return Sample(
		qid=str(r.get("_id", row)),
		question=str(r.get("text", "")).strip(),
		gold_answer=str(r.get("answer", "")).strip(),
		gold_references=[str(x) for x in refs if isinstance(x, str)],
		gold_chunk_ids=gold_ids,
	)


def post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Tuple[int, Dict[str, Any]]:
	resp = requests.post(url, json=payload, timeout=timeout)
	try:
		data = resp.json()
	except Exception:
		data = {"raw": resp.text}
	return resp.status_code, data


def _extract_chunk_id_from_item(item: Dict[str, Any]) -> Optional[str]:
	"""Robustly extract a chunk ID field from a retrieval item with various possible shapes."""
	if not isinstance(item, dict):
		return None
	for key in ("chunk_id", "chunkId", "id", "_id"):
		if key in item and str(item[key]).strip():
			return str(item[key]).strip()
	# Try nested metadata
	meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else None
	if meta:
		for key in ("chunk_id", "chunkId", "id", "_id"):
			if key in meta and str(meta[key]).strip():
				return str(meta[key]).strip()
	return None


def _extract_retrieved_chunk_ids(retrieve_data: Dict[str, Any]) -> List[str]:
	"""Extract ordered retrieved chunk IDs from various possible response formats."""
	if not isinstance(retrieve_data, dict):
		return []
	candidates = []
	for list_key in ("chunks", "results", "retrieved", "matches", "documents"):
		if list_key in retrieve_data and isinstance(retrieve_data[list_key], list):
			candidates = retrieve_data[list_key]
			break
	# Some APIs return the list directly
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


def main(csv_path: str, row: int, base_url: str, top_k: int, filters: Optional[str], metric_threshold: float, call_sleep: float) -> None:
	sample = load_sample(csv_path, row)
	print(f"Loaded sample {_short(sample.qid)}: {sample.question}")

	# Skip rows that don't have gold chunk IDs, as requested
	if not sample.gold_chunk_ids:
		print("Row has empty gold chunk IDs. Skipping evaluation for this sample.")
		return

	# Optional filters: pass a JSON string like '{"ticker":"MSFT","year":"2024"}'
	filters_obj: Optional[Dict[str, Any]] = None
	if filters:
		try:
			filters_obj = json.loads(filters)
		except Exception as e:
			print(f"Warning: failed to parse filters JSON: {e}")

	# Prepare payload
	payload = {
		"query": sample.question,
		"top_k": top_k,
		"filters": filters_obj,
	}

	# 1) Retrieve chunks-only from all three retrieval endpoints
	base = base_url.rstrip("/")
	endpoints = [
		("baseline", f"{base}/retrieve", {**payload}),
		("rerank", f"{base}/retrieve/rerank", {**payload}),
		("filter_rerank", f"{base}/retrieve/filter-rerank", {**payload}),
	]

	retrieve_responses: Dict[str, Dict[str, Any]] = {}
	retrieved_ids_by_method: Dict[str, List[str]] = {}

	for idx_ep, (method_name, url, pl) in enumerate(endpoints):
		print(f"\nPOST {url}  [{method_name}]\nBody: {json.dumps({k:v for k,v in pl.items() if v is not None})}")
		try:
			rc, retrieve_data = post_json(url, pl)
			print(f"Status: {rc}")
		except requests.exceptions.RequestException as e:
			print(f"{method_name} retrieve request failed: {e}")
			retrieve_data = {}

		retrieve_responses[method_name] = retrieve_data or {}
		retrieved_ids_by_method[method_name] = _extract_retrieved_chunk_ids(retrieve_data or {})

		if call_sleep > 0 and (idx_ep + 1) < len(endpoints):
			time.sleep(call_sleep)

	if call_sleep > 0:
		time.sleep(call_sleep)

	# 2) Full query (LLM answer)
	query_url = f"{base_url.rstrip('/')}/query"
	print(f"\nPOST {query_url}\nBody: {json.dumps({k:v for k,v in payload.items() if v is not None})}")
	try:
		rc2, query_data = post_json(query_url, payload)
		print(f"Status: {rc2}")
	except requests.exceptions.RequestException as e:
		print(f"Query request failed: {e}")
		query_data = {}

	# --- Comparisons ---
	print("\n=== GOLD ANSWER ===")
	print(sample.gold_answer or "<empty>")

	model_answer = (query_data or {}).get("answer", "")
	print("\n=== MODEL ANSWER ===")
	print(model_answer or "<empty>")

	if model_answer:
		ans_sim = jaccard(sample.gold_answer, model_answer)
		print(f"\nAnswer Jaccard similarity: {ans_sim:.3f}")

	print("\n=== GOLD CHUNK IDS (subset) ===")
	print(sample.gold_chunk_ids[:10])

	# Compare retrieved chunks using IDs (primary) and text (fallback) per method
	for method_name, retrieved_ids in retrieved_ids_by_method.items():
		print(f"\n=== {method_name.upper()} RETRIEVED CHUNK IDS (subset) ===")
		print(retrieved_ids[:10])

		# ID-based retrieval metrics
		if retrieved_ids:
			k_eff = min(top_k, len(retrieved_ids))
			print(f"\n=== {method_name.upper()} RETRIEVAL METRICS (IDs, k={k_eff}) ===")
			hit = hit_at_k(retrieved_ids, set(sample.gold_chunk_ids), k_eff)
			prec = precision_at_k(retrieved_ids, set(sample.gold_chunk_ids), k_eff)
			rec = recall_at_k(retrieved_ids, set(sample.gold_chunk_ids), k_eff)
			mrr = mrr_at_k(retrieved_ids, set(sample.gold_chunk_ids), k_eff)
			ndcg = ndcg_at_k(retrieved_ids, set(sample.gold_chunk_ids), k_eff)
			print(f"hit@{k_eff}: {hit:.3f} | precision@{k_eff}: {prec:.3f} | recall@{k_eff}: {rec:.3f} | mrr@{k_eff}: {mrr:.3f} | nDCG@{k_eff}: {ndcg:.3f}")
		else:
			print(f"\n(No ID-based metrics for {method_name}: no retrieved chunk IDs found in response.)")

	# Optional: text-based metrics as a fallback diagnostic (baseline only)
	baseline_resp = retrieve_responses.get("baseline") or {}
	chunks_any: List[Dict[str, Any]] = []
	for key in ("chunks", "results", "retrieved", "matches", "documents"):
		if isinstance(baseline_resp.get(key), list):
			chunks_any = baseline_resp.get(key) or []
			break
	retrieved_texts = [str(c.get("text", "")) for c in chunks_any if isinstance(c, dict)]

	if retrieved_texts and sample.gold_references:
		k_text = min(top_k, len(retrieved_texts))
		print("\n=== BASELINE RETRIEVAL METRICS (text fallback, threshold={:.2f}, k={}) ===".format(metric_threshold, k_text))
		hit_t = text_hit_at_k(retrieved_texts, sample.gold_references, k_text, threshold=metric_threshold)
		prec_t = text_precision_at_k(retrieved_texts, sample.gold_references, k_text, threshold=metric_threshold)
		rec_t = text_recall_at_k(retrieved_texts, sample.gold_references, k_text, threshold=metric_threshold)
		mrr_t = text_mrr_at_k(retrieved_texts, sample.gold_references, k_text, threshold=metric_threshold)
		ndcg_t = text_ndcg_at_k(retrieved_texts, sample.gold_references, k_text, threshold=metric_threshold)
		print(f"hit@{k_text}: {hit_t:.3f} | precision@{k_text}: {prec_t:.3f} | recall@{k_text}: {rec_t:.3f} | mrr@{k_text}: {mrr_t:.3f} | nDCG@{k_text}: {ndcg_t:.3f}")

	# Sources summary (if any)
	sources = (query_data or {}).get("sources", [])
	if sources:
		print("\n=== SOURCES (top few) ===")
		for s in sources[:5]:
			print(json.dumps(s))


def _truncate(s: str, max_chars: int = 1500) -> str:
	return s if len(s) <= max_chars else s[:max_chars] + "\n... [truncated] ..."


def _short(s: str, n: int = 12) -> str:
	return s if len(s) <= n else s[:n] + "…"


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate one FinDER sample against the local QA service.")
	parser.add_argument("--csv", type=str, required=True, help="Path to the FinDER subset CSV file.")
	parser.add_argument("--row", type=int, default=0, help="Row index to evaluate from the CSV (default: 0).")
	parser.add_argument(
		"--base-url",
		type=str,
		default="http://localhost:5001",
		help="Base URL for the QA service (default: http://localhost:5001)",
	)
	parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval (default: 10).")
	parser.add_argument(
		"--filters",
		type=str,
		default=None,
		help='Optional JSON string for filters, e.g. "{\"ticker\":\"MSFT\",\"year\":\"2024\"}"',
	)
	parser.add_argument(
		"--metric-threshold",
		type=float,
		default=0.2,
		help="Similarity threshold for text-based matching fallback (default: 0.2)",
	)
	parser.add_argument(
		"--call-sleep",
		type=float,
		default=0.5,
		help="Seconds to sleep between service calls (default: 0.5)",
	)
	args = parser.parse_args()

	try:
		main(args.csv, args.row, args.base_url, args.top_k, args.filters, args.metric_threshold, args.call_sleep)
	except KeyboardInterrupt:
		sys.exit(130)
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)

