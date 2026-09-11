"""Strict loader and source-independent validation for business graph Gold cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "case_id", "category", "query", "expected_router", "expected_plan",
    "expected_sql_scope", "expected_graph_paths", "expected_card_evidence",
    "expected_window_evidence", "expected_prompt", "expected_citation_contract",
    "expected_status", "ideal_answer",
}
VALID_STATUSES = {"SUCCESS", "NO_EVIDENCE", "UNSUPPORTED", "FAILED"}


def load_gold_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        missing = REQUIRED_TOP_LEVEL - set(case)
        if missing:
            raise ValueError(f"line {line_number} missing Gold fields: {sorted(missing)}")
        case_id = str(case["case_id"])
        if not case_id or case_id in seen:
            raise ValueError(f"duplicate or blank Gold case_id: {case_id!r}")
        seen.add(case_id)
        status = case["expected_status"]
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid Gold status: {status}")
        if status != "SUCCESS" and (case["expected_card_evidence"] or case["expected_window_evidence"]):
            raise ValueError(f"non-success Gold case contains evidence: {case_id}")
        if case["expected_prompt"].get("rerank_unit") != "WINDOW":
            raise ValueError(f"Gold must use Window reranking: {case_id}")
        if not isinstance(case["ideal_answer"].get("required_claims"), list):
            raise ValueError(f"Gold ideal_answer.required_claims must be a list: {case_id}")
        cases.append(case)
    if not cases:
        raise ValueError("Gold dataset is empty")
    return cases


__all__ = ["load_gold_cases", "REQUIRED_TOP_LEVEL", "VALID_STATUSES"]
