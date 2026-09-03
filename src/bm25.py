from __future__ import annotations

"""Small, dependency-free BM25 lexical index used beside Chroma Dense search."""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


# Keep decimals and English contractions intact, while splitting Chinese into
# character tokens so short Chinese queries have a useful lexical channel.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]|[^\W_]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English/math text deterministically."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return [token for token in _TOKEN_RE.findall(normalized) if token.strip()]


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Support the equality/$eq/$in subset used by current Chroma filters."""
    if not where:
        return True
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
        elif actual != expected:
            return False
    return True


@dataclass(frozen=True)
class BM25Hit:
    document_id: str
    score: float
    document: str
    metadata: dict[str, Any]


class BM25Index:
    """In-memory Okapi BM25 index over one Chroma collection snapshot."""

    def __init__(
        self,
        rows: Iterable[tuple[str, str, dict[str, Any] | None]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        for document_id, document, metadata in rows:
            text = str(document or "")
            tokens = tokenize(text)
            frequencies = Counter(tokens)
            self._ids.append(str(document_id))
            self._documents.append(text)
            self._metadatas.append(dict(metadata or {}))
            self._term_frequencies.append(frequencies)
            self._doc_lengths.append(len(tokens))
            document_frequency.update(frequencies.keys())
        self.document_count = len(self._ids)
        self.average_document_length = (
            sum(self._doc_lengths) / self.document_count if self.document_count else 0.0
        )
        self._idf = {
            term: math.log(1.0 + (self.document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_chroma_collection(cls, collection: Any, **kwargs: Any) -> "BM25Index":
        payload = collection.get(include=["documents", "metadatas"])
        ids = payload.get("ids", [])
        documents = payload.get("documents", []) or []
        metadatas = payload.get("metadatas", []) or []
        rows = [
            (str(document_id), documents[index] if index < len(documents) else "", metadatas[index] if index < len(metadatas) else {})
            for index, document_id in enumerate(ids)
        ]
        return cls(rows, **kwargs)

    def _score_row(self, query_terms: Counter[str], row_index: int) -> float:
        if not query_terms:
            return 0.0
        frequencies = self._term_frequencies[row_index]
        length = self._doc_lengths[row_index]
        avgdl = self.average_document_length or 1.0
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            idf = self._idf.get(term, 0.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * length / avgdl)
            score += idf * (tf * (self.k1 + 1.0) / denominator)
        return score

    def search(self, query: str, *, top_k: int = 20, where: dict[str, Any] | None = None) -> list[BM25Hit]:
        if top_k <= 0:
            raise ValueError("BM25 top_k must be positive")
        query_terms = Counter(tokenize(query))
        scored: list[BM25Hit] = []
        for index, document_id in enumerate(self._ids):
            if not _matches_where(self._metadatas[index], where):
                continue
            score = self._score_row(query_terms, index)
            if score <= 0:
                continue
            scored.append(BM25Hit(document_id, score, self._documents[index], self._metadatas[index]))
        scored.sort(key=lambda hit: (-hit.score, hit.document_id))
        return scored[:top_k]

    def get(self, document_id: str) -> BM25Hit | None:
        for index, candidate_id in enumerate(self._ids):
            if candidate_id == str(document_id):
                return BM25Hit(candidate_id, 0.0, self._documents[index], self._metadatas[index])
        return None
