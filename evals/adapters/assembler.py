from __future__ import annotations

import time
import json
from typing import Any


def _json_value(value: Any) -> Any:
    """Normalize backend scalar/array types before they enter auditable JSON."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    try:
        json.dumps(value)
    except TypeError as exc:
        raise TypeError(f"trace contains non-JSON value: {type(value).__name__}") from exc
    return value


def assemble_with_trace(raw_query: str, retrieval_results: dict[str, Any], assembler: Any) -> dict[str, Any]:
    """Run the production assembler and expose the evidence/prompt trace."""
    started = time.perf_counter()
    context = assembler.assemble(raw_query=raw_query, retrieval_results=retrieval_results)
    selected_ids = []
    for attr in ("selected_misconception", "selected_strategy"):
        selected = getattr(context, attr, None)
        if selected and selected.get("chunk_id"):
            selected_ids.append(str(selected["chunk_id"]))
    selected_ids.extend(
        str(item["chunk_id"])
        for item in (getattr(context, "selected_windows", None) or [])
        if item.get("chunk_id")
    )
    selected_ids.extend(
        str(item["chunk_id"])
        for item in (getattr(context, "selected_evidence_units", None) or [])
        if item.get("chunk_id")
    )
    prompt = str(getattr(context, "prompt_context_markdown", ""))
    return _json_value({
        "input_candidate_ids": [
            str(candidate["chunk_id"])
            for key in ("misconceptions", "strategies", "fallback_windows", "evidence_units")
            for candidate in retrieval_results.get(key, [])
            if candidate.get("chunk_id")
        ],
        "final_context_ids": selected_ids,
        "final_evidence_turns": list(getattr(context, "evidence_turns", []) or []),
        "actual_prompt_context": prompt,
        "estimated_tokens": int(getattr(context, "estimated_token_count", 0) or 0),
        "stage_latency_ms": (time.perf_counter() - started) * 1000.0,
    })
