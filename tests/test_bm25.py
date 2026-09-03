from __future__ import annotations

import pytest

from src.bm25 import BM25Index, tokenize


def test_tokenize_preserves_decimal_and_chinese_characters() -> None:
    tokens = tokenize("四舍五入 5.45 don't")
    assert "5.45" in tokens
    assert "四" in tokens and "入" in tokens
    assert "don't" in tokens


def test_bm25_ranks_exact_lexical_match_first() -> None:
    index = BM25Index(
        [
            ("a", "round 5.45 to one decimal place", {}),
            ("b", "rounding decimals generally", {}),
            ("c", "lowest common multiple", {}),
        ]
    )
    hits = index.search("5.45 one decimal", top_k=3)
    assert [hit.document_id for hit in hits] == ["a"]
    assert hits[0].score > 0


def test_bm25_supports_equality_and_in_filters() -> None:
    index = BM25Index(
        [
            ("a", "rounding", {"session_id": 1, "slot": "student"}),
            ("b", "rounding", {"session_id": 2, "slot": "student"}),
        ]
    )
    assert [hit.document_id for hit in index.search("rounding", where={"session_id": 2})] == ["b"]
    assert [hit.document_id for hit in index.search("rounding", where={"session_id": {"$in": [1]}})] == ["a"]


def test_bm25_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        BM25Index([], k1=0)
    with pytest.raises(ValueError):
        BM25Index([], b=2)
