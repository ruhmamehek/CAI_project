"""
Retrieval metrics utilities for RAG evaluation.

Provides the following per-query metrics computed at cutoff k:
- hit@k: 1.0 if any relevant item appears in the top-k results, else 0.0
- precision@k: number of relevant items in top-k divided by k
- recall@k: number of relevant items in top-k divided by total number of relevant items
- mrr@k: reciprocal rank of the first relevant item in the top-k (0.0 if none)
- nDCG@k: normalized Discounted Cumulative Gain at k (supports binary or graded relevance)

Contract (per function):
- Inputs
  - retrieved_ids: ranked list (best first) of retrieved identifiers (doc IDs, chunk IDs, etc.)
  - relevant: can be either a set/list of relevant ids (binary relevance) or a dict mapping id -> gain (graded)
  - k: positive integer cutoff
- Outputs
  - float in [0, 1] for hit@k, precision@k, recall@k, mrr@k, ndcg@k
- Edge cases
  - If k <= 0 or retrieved_ids is empty, returns 0.0 for all metrics
  - For recall@k with no relevant items in ground truth, returns 0.0 (undefined recall treated as 0)
  - For nDCG@k, if the ideal DCG is 0 (no relevant items), returns 0.0

Notes
- precision@k uses k as the denominator even if fewer than k items are returned (missing items count as non-relevant),
  which is a common IR convention. If you prefer to normalize by the number of retrieved items, adjust denominator accordingly.
- nDCG@k uses the standard formulation with gains: DCG = sum_{i=1..k} (2^gain_i - 1) / log2(i+1).
  For binary relevance (0/1), this reduces to sum_{relevant} 1/log2(i+1).
"""
from __future__ import annotations

from math import log2
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union


Relevant = Union[Iterable[Any], Mapping[Any, float]]


def _as_gain_dict(relevant: Relevant) -> Dict[Any, float]:
	"""Normalize relevant signal to a dict[id] -> gain (float >= 0).

	- If `relevant` is a mapping, treat values as graded gains.
	- Otherwise, treat it as an iterable of ids with binary gain 1.0.
	"""
	if isinstance(relevant, Mapping):
		# ensure float values
		return {k: float(v) for k, v in relevant.items()}
	# iterable of ids (binary)
	return {rid: 1.0 for rid in relevant}


def hit_at_k(retrieved_ids: Sequence[Any], relevant: Relevant, k: int) -> float:
	"""Hit@k: 1.0 if any relevant appears in the top-k; else 0.0.

	Args:
		retrieved_ids: Ranked list of retrieved ids (best first).
		relevant: Set/list of relevant ids OR dict[id] -> gain.
		k: Cutoff.
	"""
	if k <= 0 or not retrieved_ids:
		return 0.0
	gains = _as_gain_dict(relevant)
	limit = min(k, len(retrieved_ids))
	for i in range(limit):
		if gains.get(retrieved_ids[i], 0.0) > 0.0:
			return 1.0
	return 0.0


def precision_at_k(retrieved_ids: Sequence[Any], relevant: Relevant, k: int) -> float:
	"""Precision@k: (# relevant in top-k) / k.

	If fewer than k items are retrieved, missing ranks are treated as non-relevant.
	"""
	if k <= 0:
		return 0.0
	if not retrieved_ids:
		return 0.0
	gains = _as_gain_dict(relevant)
	limit = min(k, len(retrieved_ids))
	rel_count = sum(1 for i in range(limit) if gains.get(retrieved_ids[i], 0.0) > 0.0)
	return rel_count / float(k)


def recall_at_k(retrieved_ids: Sequence[Any], relevant: Relevant, k: int) -> float:
	"""Recall@k: (# relevant in top-k) / (# relevant total).

	If there are zero relevant items total, returns 0.0.
	"""
	gains = _as_gain_dict(relevant)
	# count relevant total as those with positive gain
	relevant_total = sum(1 for g in gains.values() if g > 0.0)
	if relevant_total == 0:
		return 0.0
	if k <= 0 or not retrieved_ids:
		return 0.0
	limit = min(k, len(retrieved_ids))
	rel_count = sum(1 for i in range(limit) if gains.get(retrieved_ids[i], 0.0) > 0.0)
	return rel_count / float(relevant_total)


def mrr_at_k(retrieved_ids: Sequence[Any], relevant: Relevant, k: int) -> float:
	"""MRR@k: reciprocal rank of the first relevant item in the top-k (0.0 if none)."""
	if k <= 0 or not retrieved_ids:
		return 0.0
	gains = _as_gain_dict(relevant)
	limit = min(k, len(retrieved_ids))
	for i in range(limit):
		if gains.get(retrieved_ids[i], 0.0) > 0.0:
			return 1.0 / float(i + 1)
	return 0.0


def _dcg_at_k_gains(gains_ranked: Sequence[float], k: int) -> float:
	"""Compute DCG at k given a ranked list of gains for each rank (1-based),
	using DCG = sum_{i=1..k} (2^g_i - 1) / log2(i + 1).
	"""
	if k <= 0:
		return 0.0
	limit = min(k, len(gains_ranked))
	dcg = 0.0
	for i in range(limit):
		g = gains_ranked[i]
		# denominator: log2(i+2) because ranks are 1-based (i starts at 0)
		den = log2(i + 2)
		# standard formulation for graded relevance
		dcg += (2.0 ** g - 1.0) / den if den > 0.0 else 0.0
	return dcg


def ndcg_at_k(retrieved_ids: Sequence[Any], relevant: Relevant, k: int) -> float:
	"""nDCG@k with binary or graded relevance.

	- If `relevant` is a set/list, treats relevance as binary (1 for relevant, 0 otherwise).
	- If `relevant` is a dict id->gain, uses the provided graded gains.
	- Returns 0.0 if there is no relevant/gainful item in ground truth.
	"""
	if k <= 0 or not retrieved_ids:
		return 0.0
	gain_map = _as_gain_dict(relevant)
	# Gains for the retrieved ranking
	retrieved_gains: List[float] = [float(gain_map.get(doc_id, 0.0)) for doc_id in retrieved_ids[:k]]
	dcg = _dcg_at_k_gains(retrieved_gains, k)
	# Ideal gains: sort all available gains descending, take top-k
	ideal_gains_sorted: List[float] = sorted((float(g) for g in gain_map.values()), reverse=True)
	idcg = _dcg_at_k_gains(ideal_gains_sorted, k)
	if idcg <= 0.0:
		return 0.0
	return dcg / idcg


__all__ = [
	"hit_at_k",
	"precision_at_k",
	"recall_at_k",
	"mrr_at_k",
	"ndcg_at_k",
]

# # ---------------------------
# # Text-based metric variants
# # ---------------------------

# def _normalize_tokens(text: str) -> List[str]:
# 	if text is None:
# 		return []
# 	t = text.lower().strip()
# 	for ch in ",.;:()[]{}<>\"'`!?/\\|#*$^~":
# 		t = t.replace(ch, " ")
# 	return [tok for tok in t.split() if tok]


# def _jaccard_similarity(a: str, b: str) -> float:
# 	A, B = set(_normalize_tokens(a)), set(_normalize_tokens(b))
# 	if not A and not B:
# 		return 1.0
# 	if not A or not B:
# 		return 0.0
# 	inter = len(A & B)
# 	union = len(A | B)
# 	return inter / union if union else 0.0


# def _binary_matches_by_rank(
# 	retrieved_texts: Sequence[str],
# 	gold_texts: Sequence[str],
# 	k: int,
# 	*,
# 	threshold: float = 0.2,
# 	sim_fn: Callable[[str, str], float] = _jaccard_similarity,
# ) -> Tuple[List[int], int, int]:
# 	"""Greedy one-to-one matching of retrieved texts to gold texts up to k ranks.

# 	Returns (matches_per_rank, matched_count, first_match_rank), where:
# 	- matches_per_rank is a list of 0/1 for ranks 1..k (list length == min(k, len(retrieved_texts)))
# 	- matched_count is the number of unique gold texts matched (<= len(gold_texts))
# 	- first_match_rank is 1-based rank of first match, or 0 if none
# 	"""
# 	if k <= 0 or not retrieved_texts or not gold_texts:
# 		return [], 0, 0

# 	limit = min(k, len(retrieved_texts))
# 	unmatched_gold = set(range(len(gold_texts)))
# 	matches: List[int] = [0] * limit
# 	first_rank = 0

# 	for i in range(limit):
# 		rt = retrieved_texts[i]
# 		# find best unmatched gold by similarity
# 		best_g = -1
# 		best_s = -1.0
# 		for g_idx in list(unmatched_gold):
# 			s = sim_fn(rt, gold_texts[g_idx])
# 			if s > best_s:
# 				best_s = s
# 				best_g = g_idx
# 		if best_s >= threshold and best_g != -1:
# 			matches[i] = 1
# 			unmatched_gold.remove(best_g)
# 			if first_rank == 0:
# 				first_rank = i + 1

# 	matched_count = sum(matches)
# 	return matches, matched_count, first_rank


# def text_hit_at_k(
# 	retrieved_texts: Sequence[str], gold_texts: Sequence[str], k: int, *, threshold: float = 0.2,
# 	sim_fn: Callable[[str, str], float] = _jaccard_similarity,
# ) -> float:
# 	"""Hit@k using text similarities. Returns 1.0 if any retrieved text matches any gold text above threshold."""
# 	matches, matched_count, _ = _binary_matches_by_rank(retrieved_texts, gold_texts, k, threshold=threshold, sim_fn=sim_fn)
# 	return 1.0 if matched_count > 0 else 0.0


# def text_precision_at_k(
# 	retrieved_texts: Sequence[str], gold_texts: Sequence[str], k: int, *, threshold: float = 0.2,
# 	sim_fn: Callable[[str, str], float] = _jaccard_similarity,
# ) -> float:
# 	"""Precision@k using text similarities, normalized by k (missing ranks count as non-relevant)."""
# 	if k <= 0:
# 		return 0.0
# 	if not retrieved_texts:
# 		return 0.0
# 	matches, matched_count, _ = _binary_matches_by_rank(retrieved_texts, gold_texts, k, threshold=threshold, sim_fn=sim_fn)
# 	return float(sum(matches)) / float(k)


# def text_recall_at_k(
# 	retrieved_texts: Sequence[str], gold_texts: Sequence[str], k: int, *, threshold: float = 0.2,
# 	sim_fn: Callable[[str, str], float] = _jaccard_similarity,
# ) -> float:
# 	"""Recall@k using text similarities, with unique gold matches to avoid double counting."""
# 	gold_total = len([g for g in gold_texts if _normalize_tokens(g)])
# 	if gold_total == 0:
# 		return 0.0
# 	if k <= 0 or not retrieved_texts:
# 		return 0.0
# 	_, matched_count, _ = _binary_matches_by_rank(retrieved_texts, gold_texts, k, threshold=threshold, sim_fn=sim_fn)
# 	return float(matched_count) / float(gold_total)


# def text_mrr_at_k(
# 	retrieved_texts: Sequence[str], gold_texts: Sequence[str], k: int, *, threshold: float = 0.2,
# 	sim_fn: Callable[[str, str], float] = _jaccard_similarity,
# ) -> float:
# 	"""MRR@k using text similarities, based on rank of first match above threshold."""
# 	if k <= 0 or not retrieved_texts:
# 		return 0.0
# 	_, _, first_rank = _binary_matches_by_rank(retrieved_texts, gold_texts, k, threshold=threshold, sim_fn=sim_fn)
# 	return 1.0 / float(first_rank) if first_rank > 0 else 0.0


# def text_ndcg_at_k(
# 	retrieved_texts: Sequence[str], gold_texts: Sequence[str], k: int, *, threshold: float = 0.2,
# 	sim_fn: Callable[[str, str], float] = _jaccard_similarity,
# ) -> float:
# 	"""Binary nDCG@k using text similarities.

# 	- We compute a 0/1 gain at each rank based on whether the retrieved text matches an unmatched gold text (>= threshold).
# 	- The ideal DCG assumes perfect ranking of up to min(k, #gold) relevant items, each with gain 1.
# 	"""
# 	if k <= 0 or not retrieved_texts:
# 		return 0.0
# 	matches, _, _ = _binary_matches_by_rank(retrieved_texts, gold_texts, k, threshold=threshold, sim_fn=sim_fn)
# 	if not gold_texts:
# 		return 0.0
# 	# DCG with binary gains
# 	dcg = _dcg_at_k_gains([float(m) for m in matches], k)
# 	# Ideal DCG with up to min(k, len(gold_texts)) ones
# 	ideal_ones = [1.0] * min(k, len([g for g in gold_texts if _normalize_tokens(g)]))
# 	idcg = _dcg_at_k_gains(ideal_ones, k)
# 	if idcg <= 0.0:
# 		return 0.0
# 	return dcg / idcg


# __all__ += [
# 	"text_hit_at_k",
# 	"text_precision_at_k",
# 	"text_recall_at_k",
# 	"text_mrr_at_k",
# 	"text_ndcg_at_k",
# ]
