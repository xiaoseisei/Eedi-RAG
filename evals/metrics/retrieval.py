from __future__ import annotations

import math
from collections.abc import Sequence


def _validate(ranked_ids: Sequence[str], qrels: dict[str, int], k: int | None = None) -> set[str]:
    if k is not None and k <= 0:
        raise ValueError("k must be greater than zero")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked_ids contain duplicate document IDs")
    positive = {document_id for document_id, relevance in qrels.items() if relevance > 0}
    if not positive:
        raise ValueError("qrels must contain positive relevance judgments")
    return positive


def recall_at_k(ranked_ids: Sequence[str], qrels: dict[str, int], k: int) -> float:
    positive = _validate(ranked_ids, qrels, k)
    hits = positive.intersection(ranked_ids[:k])
    return len(hits) / len(positive)


def candidate_recall_at_k(ranked_ids: Sequence[str], qrels: dict[str, int], k: int) -> float:
    positive = _validate(ranked_ids, qrels, k)
    return float(bool(positive.intersection(ranked_ids[:k])))


def precision_at_k(ranked_ids: Sequence[str], qrels: dict[str, int], k: int) -> float:
    positive = _validate(ranked_ids, qrels, k)
    return len(positive.intersection(ranked_ids[:k])) / k


def mean_reciprocal_rank(ranked_ids: Sequence[str], qrels: dict[str, int]) -> float:
    positive = _validate(ranked_ids, qrels)
    for rank, document_id in enumerate(ranked_ids, start=1):
        if document_id in positive:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], qrels: dict[str, int], k: int) -> float:
    _validate(ranked_ids, qrels, k)

    def discounted_gain(relevances: Sequence[int]) -> float:
        return sum((2**relevance - 1) / math.log2(rank + 1) for rank, relevance in enumerate(relevances, 1))

    actual = [qrels.get(document_id, 0) for document_id in ranked_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = discounted_gain(ideal)
    if ideal_dcg == 0:
        raise ValueError("nDCG is undefined without positive graded relevance")
    return discounted_gain(actual) / ideal_dcg


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("percentile values must be finite")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)

