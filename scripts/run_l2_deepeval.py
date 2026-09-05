from __future__ import annotations

"""Build and run the L2 Gold/Real end-to-end DeepEval loop."""

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.contracts import DeepEvalCasePayload, DeepEvalSettings
from evals.deepeval_adapter import (
    DeepEvalUnavailableError,
    build_deepeval_test_case,
    make_deepeval_model,
    probe_deepeval,
)
from evals.live_storage import hash_storage_artifacts
from evals.metrics.citation import audit_grounding
from evals.contracts import EvidenceRef
from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction
from src.generator_contract import summarize_claim_coverage
from src.reranker import estimate_text_tokens


def _resolve_l2_route(
    *,
    reranker_backend: str,
    rerank_unit: str | None,
    reranker_pool_size: int | None,
    parent_card_count: int,
    evidence_selection_count: int,
) -> dict[str, Any]:
    """Resolve L2 runtime defaults without contacting external services."""

    if reranker_backend not in {"none", "siliconflow"}:
        raise ValueError("reranker_backend must be none or siliconflow")
    if rerank_unit is not None and rerank_unit not in {
        "card",
        "logical_evidence",
        "anchored_logical_window",
    }:
        raise ValueError(
            "rerank_unit must be card, logical_evidence, or anchored_logical_window"
        )
    if parent_card_count <= 0 or evidence_selection_count <= 0:
        raise ValueError("parent_card_count and evidence_selection_count must be positive")
    if reranker_pool_size is not None and reranker_pool_size <= 0:
        raise ValueError("reranker_pool_size must be positive")
    resolved_unit = rerank_unit or (
        "anchored_logical_window" if reranker_backend == "siliconflow" else "card"
    )
    resolved_pool = reranker_pool_size or (
        20 if resolved_unit == "anchored_logical_window" else 15
    )
    return {
        "reranker_backend": reranker_backend,
        "rerank_unit": resolved_unit,
        "reranker_pool_size": resolved_pool,
        "parent_card_count": parent_card_count,
        "evidence_selection_count": evidence_selection_count,
    }


def _resolve_expected_artifact_hash(
    l1_manifest: dict[str, Any], override: str | None
) -> tuple[str, bool]:
    """Resolve the artifact hash without mutating the frozen L1 manifest."""

    frozen = str(l1_manifest.get("artifact_hashes", {}).get("qwen_combined_sha256", "")).strip()
    if not frozen:
        raise ValueError("L1 manifest is missing artifact_hashes.qwen_combined_sha256")
    if override is None:
        return frozen, True
    candidate = str(override).strip()
    if len(candidate) != 64 or any(char not in "0123456789abcdefABCDEF" for char in candidate):
        raise ValueError("--expected-artifact-hash must be a 64-character SHA-256 hex digest")
    return candidate, candidate.casefold() == frozen.casefold()


def _select_l2_cases(
    golden: list[dict[str, Any]], *, case_start: int, max_cases: int
) -> list[tuple[int, dict[str, Any]]]:
    """Select a 1-based contiguous case shard without renumbering cases."""

    if case_start <= 0:
        raise ValueError("case_start must be positive")
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")
    start_index = case_start - 1
    if start_index >= len(golden):
        raise ValueError(
            f"case_start {case_start} exceeds golden case count {len(golden)}"
        )
    return list(enumerate(golden[start_index : start_index + max_cases], case_start))


def _load_checkpoint(
    checkpoint_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    """Load only complete two-track cases from an existing checkpoint."""

    traces_path = checkpoint_dir / "traces.jsonl"
    metrics_path = checkpoint_dir / "metrics.jsonl"
    completed_path = checkpoint_dir / "completed_cases.jsonl"
    traces: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    if traces_path.exists():
        for line in traces_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                traces.append(json.loads(line))
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                metrics.append(json.loads(line))
    completed: set[int] = set()
    if completed_path.exists():
        for line in completed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(int(json.loads(line)["case_index"]))
    else:
        by_case: dict[int, set[str]] = {}
        for row in traces:
            case_id = str(row.get("case_id", ""))
            try:
                case_index = int(case_id.rsplit("-", 1)[-1])
            except ValueError:
                continue
            by_case.setdefault(case_index, set()).add(str(row.get("track", "")))
        completed = {
            case_index
            for case_index, tracks in by_case.items()
            if {"gold", "real"} <= tracks
        }
    if completed:
        traces = [
            row
            for row in traces
            if int(str(row.get("case_id", "0")).rsplit("-", 1)[-1]) in completed
        ]
        metrics = [
            row
            for row in metrics
            if int(str(row.get("case_id", "0")).rsplit("-", 1)[-1]) in completed
        ]
    else:
        traces = []
        metrics = []
    return traces, metrics, completed


def _append_checkpoint(
    checkpoint_dir: Path,
    traces: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> None:
    """Append completed case records with flush+fsync for crash recovery."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("traces.jsonl", traces), ("metrics.jsonl", metrics)):
        if not rows:
            continue
        with (checkpoint_dir / filename).open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    tracks_by_case: dict[int, set[str]] = {}
    for row in traces:
        if not row.get("case_id"):
            continue
        case_index = int(str(row["case_id"]).rsplit("-", 1)[-1])
        tracks_by_case.setdefault(case_index, set()).add(str(row.get("track", "")))
    case_indices = sorted(
        case_index
        for case_index, tracks in tracks_by_case.items()
        if {"gold", "real"} <= tracks
    )
    if case_indices:
        with (checkpoint_dir / "completed_cases.jsonl").open("a", encoding="utf-8") as handle:
            for case_index in case_indices:
                handle.write(json.dumps({"case_index": case_index}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sessions(path: Path) -> dict[int, Any]:
    from src.models import CleanedSession

    sessions: dict[int, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            session = CleanedSession.model_validate(json.loads(line))
            sessions[session.intervention_id] = session
    return sessions


def _evidence_refs(case: dict[str, Any], sessions: dict[int, Any]) -> tuple[list[EvidenceRef], list[EvidenceRef]]:
    required: list[EvidenceRef] = []
    authoritative: list[EvidenceRef] = []
    session_ids = sorted({int(item["session_id"]) for item in case.get("verbatim_grounding_quotes", [])})
    for session_id in session_ids:
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"missing session {session_id} for L2 case")
        for turn in session.turns:
            authoritative.append(EvidenceRef(
                session_id=session_id,
                turn_id=turn.turn_id,
                speaker="tutor" if turn.is_tutor else "student",
                quote_text=turn.text,
            ))
    for item in case.get("verbatim_grounding_quotes", []):
        required.append(EvidenceRef(
            session_id=int(item["session_id"]),
            turn_id=int(item["turn_id"]),
            speaker=item["speaker"],
            quote_text=item["quote_text"],
        ))
    return required, authoritative


def _claim_segments(text: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"(?<=[。！？!?；;])\s*|\r?\n+", str(text or ""))
        if segment.strip()
    ]


def _claim_tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", str(text or "").casefold()))


def _row_value(row: Any, field: str) -> Any:
    if isinstance(row, dict):
        return row.get(field)
    return getattr(row, field, None)


def _context_coverage(
    case: dict[str, Any],
    required_turns: list[Any],
    context_turns: list[Any],
    context_nodes: list[str],
    *,
    parent_cards: list[Any] | None = None,
    candidate_units: list[Any] | None = None,
) -> dict[str, Any]:
    """Report R/C/W/A Turn coverage and a transparent lexical claim proxy."""

    def item_turn_keys(item: Any) -> set[tuple[int, int]]:
        metadata = _row_value(item, "metadata") or {}
        session_id = metadata.get("session_id") if isinstance(metadata, dict) else None
        raw_ids = metadata.get("source_turn_ids", []) if isinstance(metadata, dict) else []
        if isinstance(raw_ids, str):
            try:
                raw_ids = json.loads(raw_ids)
            except json.JSONDecodeError:
                raw_ids = []
        keys: set[tuple[int, int]] = set()
        if session_id is not None:
            for turn_id in raw_ids or []:
                try:
                    keys.add((int(session_id), int(turn_id)))
                except (TypeError, ValueError):
                    continue
        for turn in _row_value(item, "evidence_turns") or []:
            turn_session = _row_value(turn, "session_id")
            turn_id = _row_value(turn, "turn_id")
            if turn_session is not None and turn_id is not None:
                keys.add((int(turn_session), int(turn_id)))
        return keys

    required_keys = {
        (int(_row_value(item, "session_id")), int(_row_value(item, "turn_id")))
        for item in required_turns
    }
    context_keys = {
        (int(_row_value(item, "session_id")), int(_row_value(item, "turn_id")))
        for item in context_turns
        if _row_value(item, "session_id") is not None
        and _row_value(item, "turn_id") is not None
    }
    covered_keys = required_keys & context_keys
    parent_keys = set().union(*(item_turn_keys(item) for item in parent_cards or []))
    candidate_keys = set().union(*(item_turn_keys(item) for item in candidate_units or []))
    claims = _claim_segments(str(case.get("ground_truth", "")))
    context_text = "\n".join(str(node) for node in context_nodes if str(node).strip())
    if not context_text:
        context_text = "\n".join(
            str(_row_value(item, "text") or _row_value(item, "quote_text") or "")
            for item in context_turns
        )
    context_tokens = _claim_tokens(context_text)
    claim_scores: list[float] = []
    for claim in claims:
        tokens = _claim_tokens(claim)
        if not tokens:
            continue
        numeric_tokens = set(re.findall(r"\d+(?:\.\d+)?", claim))
        numeric_hits = numeric_tokens & set(re.findall(r"\d+(?:\.\d+)?", context_text))
        token_score = len(tokens & context_tokens) / len(tokens)
        if numeric_tokens and numeric_hits != numeric_tokens:
            token_score = 0.0
        claim_scores.append(token_score)
    claim_recall = sum(claim_scores) / len(claim_scores) if claim_scores else None
    supported_claim_count = sum(score >= 0.5 for score in claim_scores)
    return {
        "required_turn_count": len(required_keys),
        "covered_turn_count": len(covered_keys),
        "turn_recall": len(covered_keys) / len(required_keys) if required_keys else None,
        "parent_pointer_count": len(required_keys & parent_keys),
        "parent_pointer_recall": (
            len(required_keys & parent_keys) / len(required_keys) if parent_cards else None
        ),
        "candidate_turn_count": len(required_keys & candidate_keys),
        "candidate_turn_recall": (
            len(required_keys & candidate_keys) / len(required_keys) if candidate_units else None
        ),
        "claim_count": len(claim_scores),
        "supported_claim_count": supported_claim_count,
        "claim_recall": claim_recall,
        "claim_recall_method": "lexical_context_coverage_proxy",
    }


def _build_gold_context(case: dict[str, Any], sessions: dict[int, Any]):
    from src.reranker import GoldAssembledContext

    required, _ = _evidence_refs(case, sessions)
    evidence_turns = [
        {
            "session_id": item.session_id,
            "turn_id": item.turn_id,
            "speaker": item.speaker,
            "text": item.quote_text,
        }
        for item in required
    ]
    lines = [
        "# Gold Context (evaluation-only upper bound)",
        f"Query: {case['question']}",
        f"Subject path: {case.get('subject_path', '')}",
        "The following evidence is an evaluation-only gold selection and must not be used by production retrieval.",
        "## Required evidence",
    ]
    for item in evidence_turns:
        lines.append(f"- [Session {item['session_id']} Turn {item['turn_id']}] [{item['speaker']}]: {item['text']}")
    return GoldAssembledContext(
        raw_query=case["question"],
        prompt_context_markdown="\n".join(lines),
        evidence_turns=evidence_turns,
        estimated_token_count=max(1, len("\n".join(lines)) // 2),
    )


def _build_gold_context_v2(
    case: dict[str, Any], sessions: dict[int, Any], storage: Any
):
    """Build a sufficient, evaluation-only Gold Context from real artifacts.

    Gold v2 contains stored card facts and complete deterministic W6/S3 raw
    windows for the case's source sessions.  It intentionally excludes the
    case's expected answer and qrels so DeepEval can test grounded reasoning
    rather than prompt leakage.
    """

    from src.evidence_index import build_evidence_index
    from src.reranker import GoldAssembledContext

    source_session_ids = sorted(
        {int(item["session_id"]) for item in case.get("verbatim_grounding_quotes", [])}
    )
    if not source_session_ids:
        raise ValueError("Gold Context v2 requires at least one source session")
    misconception_rows = storage.query_misconceptions_sql()
    strategy_rows = storage.query_strategies_sql()
    context_nodes: list[str] = []
    derived_facts = case.get("derived_facts", [])
    if derived_facts:
        if not isinstance(derived_facts, list) or any(
            not isinstance(item, str) or not item.strip() for item in derived_facts
        ):
            raise ValueError("Gold Context v2 derived_facts must be a non-empty list of strings")
        expected_answer = str(case.get("ground_truth", "")).strip()
        if any(item.strip() == expected_answer for item in derived_facts):
            raise ValueError("Gold Context v2 derived_facts must not copy ground_truth")
        context_nodes.append(
            "## [DERIVED_FACT] Independently derived facts\n"
            "These facts must be independently computable from the question or structured data.\n"
            + "\n".join(f"- {item.strip()}" for item in derived_facts)
        )
    def has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, bytes)):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        size = getattr(value, "size", None)
        if size is not None:
            return int(size) > 0
        return True

    for session_id in source_session_ids:
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"missing session {session_id} for Gold Context v2")
        subject_path = session.subjects.paths[0] if session.subjects.paths else ""
        card_lines = [
            f"## [CARD_FACT] Session {session_id}",
            "These are stored card fields, not verbatim dialogue.",
            f"[Session {session_id}] subject_path={subject_path}",
            f"question={session.question.question_text}",
        ]
        for label, rows in (
            ("misconception", misconception_rows),
            ("strategy", strategy_rows),
        ):
            matching = [row for row in rows if int(row.get("session_id", -1)) == session_id]
            for row in matching:
                safe_fields = {
                    key: row.get(key)
                    for key in (
                        "misconception_name",
                        "error_choice",
                        "deep_mechanism",
                        "confusion_triggers",
                        "pedagogical_goal",
                        "strategy_category",
                        "key_aha_question",
                        "scaffolding_steps",
                        "talk_moves",
                        "resolution_outcome",
                    )
                    if has_value(row.get(key))
                }
                card_lines.append(
                    f"card_type={label} "
                    + json.dumps(safe_fields, ensure_ascii=False, sort_keys=True, default=str)
                )
        context_nodes.append("\n".join(card_lines))

    required_by_session: dict[int, set[int]] = {}
    for item in case.get("verbatim_grounding_quotes", []):
        required_by_session.setdefault(int(item["session_id"]), set()).add(int(item["turn_id"]))

    evidence_turns: list[dict[str, Any]] = []
    seen_turns: set[tuple[int, int]] = set()
    for session_id in source_session_ids:
        session = sessions[session_id]
        index = build_evidence_index(session, window_size=6, step=3)
        required_turn_ids = required_by_session.get(session_id, set())
        selected_ranges: list[tuple[int, int]] = []
        for window in index.windows:
            window_ids = set(window.source_turn_ids)
            if window_ids & required_turn_ids:
                selected_ranges.append((window.window_start_turn, window.window_end_turn))
        selected_ranges.sort()
        merged_ranges: list[tuple[int, int]] = []
        for start, end in selected_ranges:
            if merged_ranges and start <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
            else:
                merged_ranges.append((start, end))
        turn_by_id = {int(turn.turn_id): turn for turn in session.turns}
        for start, end in merged_ranges:
            node_lines = [
                f"## [AUTHORITATIVE_TURN] Session {session_id} · Turn {start}~{end}",
                "The following are verbatim stored dialogue turns.",
            ]
            for turn_id in sorted(turn_by_id):
                if not start <= turn_id <= end:
                    continue
                turn = turn_by_id[turn_id]
                speaker = "tutor" if turn.is_tutor else "student"
                node_lines.append(f"[AUTHORITATIVE_TURN] [Turn {turn.turn_id}] [{speaker}] {turn.text}")
                key = (session_id, int(turn.turn_id))
                if key not in seen_turns:
                    evidence_turns.append(
                        {
                            "session_id": session_id,
                            "turn_id": int(turn.turn_id),
                            "speaker": speaker,
                            "text": turn.text,
                        }
                    )
                    seen_turns.add(key)
            context_nodes.append("\n".join(node_lines))

    inference_boundary = (
        "## [ALLOWED_INFERENCE] Inference boundary\n"
        "You may infer a likely misconception or teaching rationale from the card facts and dialogue, "
        "but label it as an inference and explain its evidence basis. Do not present an inference as a verbatim Turn fact."
    )
    prompt = "\n\n".join(
        [
            "# Gold Context v2 (evaluation-only sufficient context)",
            "This context is built from real stored card facts and required-evidence raw Turns.",
            "Expected answers and qrels are intentionally excluded from the Generator input.",
            *context_nodes,
            inference_boundary,
        ]
    )
    estimated = max(1, len(prompt) // 2)
    return GoldAssembledContext(
        raw_query=case["question"],
        prompt_context_markdown=prompt,
        evidence_turns=evidence_turns,
        deepeval_context_nodes=context_nodes,
        estimated_token_count=estimated,
        budget_violation=False,
        truncation_loss=0,
    )


def audit_gold_alignment(
    case: dict[str, Any], card_rows: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Report explicit Golden-vs-card choice conflicts without changing scores."""

    ground_truth = str(case.get("ground_truth", ""))
    gold_choices = {
        token.upper()
        for token in re.findall(r"(?:选了|选择|choice|option)\s*([A-D])", ground_truth, flags=re.IGNORECASE)
    }
    if not gold_choices:
        return []
    warnings: list[dict[str, str]] = []
    session_ids = {
        int(item["session_id"])
        for item in case.get("verbatim_grounding_quotes", [])
        if item.get("session_id") is not None
    }
    for row in card_rows:
        if int(row.get("session_id", -1)) not in session_ids:
            continue
        card_value = str(row.get("error_choice", "")).strip().upper()
        if card_value and card_value not in gold_choices:
            warnings.append(
                {
                    "case_id": str(case.get("case_id", "")),
                    "session_id": str(row.get("session_id")),
                    "field": "error_choice",
                    "card_value": card_value,
                    "gold_value": sorted(gold_choices)[0],
                    "reason": "explicit Golden student choice conflicts with stored card error_choice",
                }
            )
    return warnings


def build_l2_dataset(
    *,
    output_dir: Path,
    golden_path: Path,
    l1_manifest_path: Path,
    artifact_hash: str,
    generator_model: str,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite L2 dataset: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    l1_manifest = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
    split_path = l1_manifest_path.parent / l1_manifest["split"]["file"]
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    queries: list[dict[str, Any]] = []
    expected_answers: list[dict[str, Any]] = []
    expected_evidence: list[dict[str, Any]] = []
    for index, case in enumerate(golden, 1):
        case_id = f"eedi-l2-{index:04d}"
        query_id = f"eedi-l1-{index:04d}"
        split = split_manifest["query_assignments"][query_id]
        source_sessions = sorted({int(item["session_id"]) for item in case.get("verbatim_grounding_quotes", [])})
        queries.append({
            "case_id": case_id,
            "query": case["question"],
            "intent": case.get("category", "UNKNOWN"),
            "split": split,
            "source_session_ids": source_sessions,
            "slices": {"category": case.get("category", "unknown"), "language": "zh-query/en-evidence"},
        })
        expected_answers.append({
            "case_id": case_id,
            "expected_answer": case["ground_truth"],
            "expected_facts": [case["ground_truth"]],
            "subject_path": case.get("subject_path", ""),
        })
        expected_evidence.append({
            "case_id": case_id,
            "required_evidence_quads": case.get("verbatim_grounding_quotes", []),
            "source_session_ids": source_sessions,
        })
    for name, rows in (
        ("queries.jsonl", queries),
        ("expected_answers.jsonl", expected_answers),
        ("expected_evidence.jsonl", expected_evidence),
    ):
        (output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    split_copy = output_dir / "split_manifest.json"
    split_copy.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "layer": "L2",
        "generator_model": generator_model,
        "generator_config": generator_config,
        "gold_context_policy": "evaluation_only_not_in_production_retrieval",
        "tracks": ["gold", "real"],
        "metrics": ["faithfulness", "answer_relevancy", "contextual_recall", "contextual_precision", "pedagogical_geval", "citation_audit"],
    }
    manifest = {
        "schema_version": "l2-deepeval-dataset/v1",
        "dataset_version": "eedi-l2-golden30-v1",
        "qrels_version": l1_manifest["qrels_version"],
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "case_count": len(queries),
        "split_counts": split_manifest["split_counts"],
        "session_leakage_count": split_manifest["session_leakage_count"],
        "l1_manifest_sha256": _sha256(l1_manifest_path),
        "artifact_hash": artifact_hash,
        "generator": {"model": generator_model, "config": generator_config},
        "files": {
            name: _sha256(output_dir / name)
            for name in ("queries.jsonl", "expected_answers.jsonl", "expected_evidence.jsonl", "split_manifest.json")
        },
        "system_config": config,
        "system_config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# L2 DeepEval dataset\n\n"
        "This dataset contains Session-isolated queries, expected answers, required evidence quads, and Gold/Real evaluation metadata. Gold Context is evaluation-only and never enters production retrieval.\n",
        encoding="utf-8",
    )
    return manifest


def _rubric_config(version: str) -> dict[str, str]:
    if version not in {"v1", "v2"}:
        raise ValueError("rubric version must be v1 or v2")
    if version == "v1":
        return {
            "faithfulness": "The answer is supported by the retrieval context and does not invent facts.",
            "pedagogical_geval": "The answer is useful to an education researcher, separates student misconception from tutor strategy, and proposes an evidence-grounded pedagogical intervention without inventing facts.",
        }
    return {
        "faithfulness": "Every direct factual statement must be supported by the final retrieval context. A reasoned inference is acceptable only when it is explicitly qualified as an inference and its evidence basis is cited; unsupported claims or presenting inference as a verbatim fact should fail.",
        "pedagogical_geval": "The answer directly addresses the user question, separates student misconception from tutor strategy, explains which evidence supports the diagnosis, and proposes an actionable pedagogical intervention. Concise answers are acceptable when all required elements are present; unsupported facts and generic advice should fail.",
    }


def _metrics(judge_model: Any, rubric_version: str = "v1") -> list[tuple[str, Any, float]]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import SingleTurnParams

    rubric = _rubric_config(rubric_version)
    faithfulness_metric = (
        FaithfulnessMetric(
            threshold=None,
            model=judge_model,
            async_mode=False,
            verbose_mode=False,
        )
        if rubric_version == "v1"
        else GEval(
            name="FaithfulnessV2",
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            criteria=rubric["faithfulness"],
            model=judge_model,
            threshold=None,
            async_mode=False,
        )
    )
    return [
        ("faithfulness", faithfulness_metric, 0.90),
        ("answer_relevancy", AnswerRelevancyMetric(threshold=None, model=judge_model, async_mode=False, verbose_mode=False), 0.85),
        ("contextual_recall", ContextualRecallMetric(threshold=None, model=judge_model, async_mode=False, verbose_mode=False), 0.90),
        ("contextual_precision", ContextualPrecisionMetric(threshold=None, model=judge_model, async_mode=False, verbose_mode=False), 0.85),
        (
            "pedagogical_geval",
            GEval(
                name="PedagogicalAdaptation",
                evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
                criteria=rubric["pedagogical_geval"],
                model=judge_model,
                threshold=None,
                async_mode=False,
            ),
            0.80,
        ),
    ]


def _metric_eval_payload(
    metric_name: str,
    structured_output: str,
    *,
    actual_output_by_metric: dict[str, str] | None = None,
    retrieval_context: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Select metric-specific output while preserving one evaluation context contract."""

    output = (actual_output_by_metric or {}).get(metric_name, structured_output)
    context = list(retrieval_context or [])
    if not context:
        raise ValueError("DeepEval retrieval context cannot be empty")
    return output, context


def _run_deepeval_case(
    payload: DeepEvalCasePayload,
    metrics: list[tuple[str, Any, float]],
    *,
    track: str,
    case_id: str,
    actual_output_by_metric: dict[str, str] | None = None,
    retrieval_context: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, metric, threshold in metrics:
        started = time.perf_counter()
        try:
            actual_output, context = _metric_eval_payload(
                metric_name,
                payload.actual_output,
                actual_output_by_metric=actual_output_by_metric,
                retrieval_context=retrieval_context or payload.final_retrieval_context,
            )
            metric_payload = payload.model_copy(
                update={
                    "actual_output": actual_output,
                    "final_retrieval_context": context,
                }
            )
            test_case = build_deepeval_test_case(metric_payload)
            score = metric.measure(test_case, _show_indicator=False)
            value = float(score)
            rows.append({
                "case_id": case_id,
                "track": track,
                "metric": metric_name,
                "status": "SUCCESS",
                "score": value,
                "threshold": threshold,
                "passed": value >= threshold,
                "reason": getattr(metric, "reason", None),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            })
        except Exception as exc:
            rows.append({
                "case_id": case_id,
                "track": track,
                "metric": metric_name,
                "status": "ERROR",
                "score": None,
                "threshold": threshold,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            })
    return rows


def _aggregate_trace_diagnostics(
    trace_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate non-scoring diagnostics while preserving case-level provenance."""

    warnings_by_key: dict[str, dict[str, str]] = {}
    coverage_by_track: dict[str, list[dict[str, Any]]] = {"gold": [], "real": []}
    context_coverage_by_track: dict[str, list[dict[str, Any]]] = {"gold": [], "real": []}
    budget_by_track: dict[str, list[dict[str, Any]]] = {"gold": [], "real": []}
    node_stats_by_track: dict[str, list[dict[str, int]]] = {"gold": [], "real": []}
    for row in trace_rows:
        track = str(row.get("track", ""))
        if track not in coverage_by_track:
            continue
        for warning in row.get("gold_alignment_warnings", []) or []:
            key = json.dumps(warning, ensure_ascii=False, sort_keys=True)
            warnings_by_key[key] = warning
        coverage = row.get("claim_coverage")
        if isinstance(coverage, dict):
            coverage_by_track[track].append(coverage)
        context_coverage = row.get("context_coverage")
        if isinstance(context_coverage, dict):
            context_coverage_by_track[track].append(context_coverage)
        node_stats_by_track[track].append(
            {
                "node_count": int(row.get("deepeval_context_node_count", 0) or 0),
                "context_chars": int(row.get("generator_context_chars", 0) or 0),
            }
        )
        budget_by_track[track].append(
            {
                "budget_tokens": int(row.get("context_budget_tokens", 0) or 0),
                "context_tokens": int(row.get("generator_context_token_count", 0) or 0),
                "catalog_tokens": int(row.get("catalog_token_count", 0) or 0),
                "budget_violation": bool(row.get("budget_violation", False)),
            }
        )

    def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return sum(values) / len(values) if values else None

    deep_eval_metrics = metric_rows or []
    return {
        "gold_alignment_warning_count": len(warnings_by_key),
        "gold_alignment_warnings": list(warnings_by_key.values()),
        "claim_coverage": {
            track: {
                "measured": len(rows),
                "mean_claim_count": mean_field(rows, "claim_count"),
                "mean_fact_claim_count": mean_field(rows, "fact_claim_count"),
                "mean_fact_claim_coverage": mean_field(rows, "fact_claim_coverage"),
                "mean_cited_evidence_count": mean_field(rows, "cited_evidence_count"),
            }
            for track, rows in coverage_by_track.items()
        },
        "context_nodes": {
            track: {
                "measured": len(rows),
                "mean_node_count": mean_field(rows, "node_count"),
                "mean_context_chars": mean_field(rows, "context_chars"),
            }
            for track, rows in node_stats_by_track.items()
        },
        "context_coverage": {
            track: {
                "measured": len(rows),
                "mean_turn_recall": mean_field(rows, "turn_recall"),
                "mean_claim_recall": mean_field(rows, "claim_recall"),
                "mean_required_turn_count": mean_field(rows, "required_turn_count"),
                "mean_covered_turn_count": mean_field(rows, "covered_turn_count"),
                "mean_parent_pointer_recall": mean_field(rows, "parent_pointer_recall"),
                "mean_candidate_turn_recall": mean_field(rows, "candidate_turn_recall"),
                "mean_parent_pointer_count": mean_field(rows, "parent_pointer_count"),
                "mean_candidate_turn_count": mean_field(rows, "candidate_turn_count"),
                "claim_recall_method": "lexical_context_coverage_proxy",
            }
            for track, rows in context_coverage_by_track.items()
        },
        "deep_eval": {
            track: {
                "measured_contextual_recall": sum(
                    row.get("track") == track
                    and row.get("metric") == "contextual_recall"
                    and row.get("status") == "SUCCESS"
                    and row.get("score") is not None
                    for row in deep_eval_metrics
                ),
                "mean_contextual_recall": mean_field(
                    [
                        row
                        for row in deep_eval_metrics
                        if row.get("track") == track
                        and row.get("metric") == "contextual_recall"
                        and row.get("status") == "SUCCESS"
                    ],
                    "score",
                ),
                "measured_faithfulness": sum(
                    row.get("track") == track
                    and row.get("metric") == "faithfulness"
                    and row.get("status") == "SUCCESS"
                    and row.get("score") is not None
                    for row in deep_eval_metrics
                ),
                "mean_faithfulness": mean_field(
                    [
                        row
                        for row in deep_eval_metrics
                        if row.get("track") == track
                        and row.get("metric") == "faithfulness"
                        and row.get("status") == "SUCCESS"
                    ],
                    "score",
                ),
            }
            for track in ("gold", "real")
        },
        "context_budget": {
            track: {
                "measured": len(rows),
                "budget_tokens": mean_field(rows, "budget_tokens"),
                "mean_context_tokens": mean_field(rows, "context_tokens"),
                "max_context_tokens": max(
                    (int(row.get("context_tokens", 0) or 0) for row in rows),
                    default=None,
                ),
                "mean_catalog_tokens": mean_field(rows, "catalog_tokens"),
                "budget_violations": sum(bool(row.get("budget_violation")) for row in rows),
            }
            for track, rows in budget_by_track.items()
        },
    }


def _gold_alignment_status(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Expose unresolved Golden/card conflicts as an explicit data-quality gate."""

    warning_count = int(diagnostics.get("gold_alignment_warning_count", 0) or 0)
    return {
        "status": "BLOCKED_DATA_CONFLICT" if warning_count else "CLEAN",
        "warning_count": warning_count,
        "warnings": list(diagnostics.get("gold_alignment_warnings", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--l1-closeout", type=Path, required=True)
    parser.add_argument("--l1-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-artifact-hash",
        default=None,
        help="explicit current artifact SHA-256 when the L1 manifest predates a controlled staging update",
    )
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--case-start", type=int, default=1)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key-env", default=None)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--generator-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=2000)
    parser.add_argument("--generator-contract", choices=("v1", "v2"), default="v1")
    parser.add_argument("--gold-context-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--rubric-version", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--reranker-backend",
        choices=("none", "siliconflow"),
        default="siliconflow",
        help="reranker backend for the real track; SiliconFlow defaults to Anchored Parent-3/W5",
    )
    parser.add_argument(
        "--rerank-unit",
        choices=("card", "logical_evidence", "anchored_logical_window"),
        default=None,
    )
    parser.add_argument("--reranker-pool-size", type=int, default=None)
    parser.add_argument("--parent-card-count", type=int, default=3)
    parser.add_argument("--evidence-selection-count", type=int, default=5)
    parser.add_argument("--retrieval-mode", choices=("dense", "bm25_dense"), default="bm25_dense")
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    parser.add_argument("--allow-l1-blocked", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.case_start <= 0:
        parser.error("--case-start must be positive")
    try:
        route = _resolve_l2_route(
            reranker_backend=args.reranker_backend,
            rerank_unit=args.rerank_unit,
            reranker_pool_size=args.reranker_pool_size,
            parent_card_count=args.parent_card_count,
            evidence_selection_count=args.evidence_selection_count,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 <= args.bm25_weight <= 1.0:
        parser.error("--bm25-weight must be in [0, 1]")
    if args.generator_timeout_seconds <= 0:
        parser.error("--generator-timeout-seconds must be positive")
    if args.max_prompt_tokens <= 0:
        parser.error("--max-prompt-tokens must be positive")
    l1_closeout = json.loads(args.l1_closeout.resolve(strict=True).read_text(encoding="utf-8"))
    if l1_closeout.get("decision") != "GO_TO_L2_TECHNICAL" and not args.allow_l1_blocked:
        raise SystemExit("L1 is not GO_TO_L2_TECHNICAL; pass --allow-l1-blocked only for explicit pipeline validation")
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    db_path = args.db.resolve(strict=True)
    chroma_path = args.chroma.resolve(strict=True)
    artifact_hash = hash_storage_artifacts(db_path, chroma_path)
    l1_manifest = json.loads(args.l1_manifest.resolve(strict=True).read_text(encoding="utf-8"))
    expected_artifact_hash, matches_frozen_manifest = _resolve_expected_artifact_hash(
        l1_manifest, args.expected_artifact_hash
    )
    if artifact_hash != expected_artifact_hash:
        raise ValueError(
            "L2 artifact hash does not match expected hash: "
            f"actual={artifact_hash}, expected={expected_artifact_hash}"
        )
    dataset_output = args.dataset_output.resolve()
    if args.resume and (dataset_output / "manifest.json").exists():
        dataset_manifest = json.loads(
            (dataset_output / "manifest.json").read_text(encoding="utf-8")
        )
    else:
        dataset_manifest = build_l2_dataset(
            output_dir=dataset_output,
            golden_path=(PROJECT_ROOT / "data/golden_test_set.json"),
            l1_manifest_path=args.l1_manifest.resolve(strict=True),
            artifact_hash=artifact_hash,
            generator_model=os.getenv("LLM_MODEL", "explicit-generator-model"),
            generator_config={
                "mode": "production_llm",
                "citation_contract": "quadruple",
                "retrieval_mode": args.retrieval_mode,
                "bm25_weight": args.bm25_weight,
                "reranker_backend": route["reranker_backend"],
                "rerank_unit": route["rerank_unit"],
                "reranker_pool_size": route["reranker_pool_size"],
                "parent_card_count": route["parent_card_count"],
                "evidence_selection_count": route["evidence_selection_count"],
                "generator_contract": args.generator_contract,
                "gold_context_version": args.gold_context_version,
                "rubric_version": args.rubric_version,
                "max_prompt_tokens": args.max_prompt_tokens,
            },
        )
    l2_split_manifest = json.loads((args.dataset_output.resolve() / "split_manifest.json").read_text(encoding="utf-8"))
    split_assignments = l2_split_manifest["query_assignments"]
    golden_all = json.loads((PROJECT_ROOT / "data/golden_test_set.json").read_text(encoding="utf-8"))
    selected_cases = _select_l2_cases(
        golden_all, case_start=args.case_start, max_cases=args.max_cases
    )
    sessions = _load_sessions(PROJECT_ROOT / "data/cleaned_sessions.jsonl")
    provider = SiliconFlowQwen3EmbeddingFunction.from_env()
    scratch_root = args.output_root.resolve() / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    checkpoint_dir = args.output_root.resolve() / ".l2-checkpoints" / args.run_id
    completed_case_indices: set[int] = set()
    if args.resume:
        trace_rows, metric_rows, completed_case_indices = _load_checkpoint(checkpoint_dir)
    readiness: Any = None
    judge_metrics: list[tuple[str, Any, float]] = []
    settings = None
    judge_model = args.judge_model or __import__("os").getenv("DEEPEVAL_MODEL")
    judge_key_env = args.judge_api_key_env or __import__("os").getenv("DEEPEVAL_API_KEY_ENV")
    judge_base_url = args.judge_base_url or __import__("os").getenv("DEEPEVAL_BASE_URL")
    if judge_model and judge_key_env:
        settings = DeepEvalSettings(
            evaluation_provider="openai_compatible",
            evaluation_model=judge_model,
            api_key_env=judge_key_env,
            base_url=judge_base_url,
            temperature=args.judge_temperature,
        )
        readiness = probe_deepeval(settings)
        if readiness.status.value == "SUCCESS":
            try:
                judge_model_obj = make_deepeval_model(settings)
                judge_metrics = _metrics(judge_model_obj, rubric_version=args.rubric_version)
            except Exception as exc:
                readiness = readiness.model_copy(update={"status": "ERROR", "error_message": f"judge initialization failed: {type(exc).__name__}: {exc}"})
    else:
        readiness = {"status": "UNMEASURED", "installed": True, "credentials_configured": False, "model_configured": False, "error_message": "DEEPEVAL_MODEL and DEEPEVAL_API_KEY_ENV are required"}

    copied_db: Path | None = None
    copied_chroma: Path | None = None
    with tempfile.TemporaryDirectory(prefix="l2-e2e-", dir=str(scratch_root), ignore_cleanup_errors=True) as temp_name:
        scratch = Path(temp_name)
        copied_db = scratch / "tutoring_knowledge.duckdb"
        copied_chroma = scratch / "chroma"
        shutil.copy2(db_path, copied_db)
        shutil.copytree(chroma_path, copied_chroma)
        storage = None
        try:
            storage = __import__("src.storage_manager", fromlist=["DualEngineStorageManager"]).DualEngineStorageManager(
                db_path=copied_db,
                chroma_dir=copied_chroma,
                embedding_function=provider,
                embedding_index_version=provider.index_version,
            )
            from src.retriever import DualMetricRetriever
            from src.reranker import PedagogicalGoldAssembler
            from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
            model_reranker = None
            if route["reranker_backend"] == "siliconflow":
                from src.reranker_provider import SiliconFlowQwen3Reranker

                model_reranker = SiliconFlowQwen3Reranker.from_env()
            retriever = DualMetricRetriever(
                storage_manager=storage,
                query_rewrite_mode="deterministic",
                fusion_strategy="raw_first",
                retrieval_mode=args.retrieval_mode,
                bm25_weight=args.bm25_weight,
            )
            assembler = PedagogicalGoldAssembler(
                lambda_diversity=0.7,
                max_prompt_tokens=args.max_prompt_tokens,
                model_reranker=model_reranker,
                rerank_unit=route["rerank_unit"],
                reranker_pool_size=route["reranker_pool_size"],
                parent_card_count=route["parent_card_count"],
                evidence_selection_count=route["evidence_selection_count"],
            )
            pipeline = EndToEndPedagogicalRAGPipeline(
                retriever=retriever,
                assembler=assembler,
                model_name=os.getenv("LLM_MODEL"),
                llm_timeout_seconds=args.generator_timeout_seconds,
                generator_contract_version=args.generator_contract,
            )
            for index, case in selected_cases:
                if index in completed_case_indices:
                    continue
                case_id = f"eedi-l2-{index:04d}"
                required, authoritative = _evidence_refs(case, sessions)
                case_trace_rows: list[dict[str, Any]] = []
                case_metric_rows: list[dict[str, Any]] = []
                for track in ("gold", "real"):
                    started = time.perf_counter()
                    retrieval_trace: dict[str, Any] = {}
                    errors: list[str] = []
                    gold_alignment_warnings: list[dict[str, str]] = []
                    if track == "gold":
                        context = (
                            _build_gold_context_v2(case, sessions, storage)
                            if args.gold_context_version == "v2"
                            else _build_gold_context(case, sessions)
                        )
                        gold_alignment_warnings = audit_gold_alignment(
                            {**case, "case_id": case_id},
                            storage.query_misconceptions_sql(),
                        )
                        retrieval_trace = {"policy": "gold_context_evaluation_only", "retrieved_evidence": [item.model_dump() for item in required]}
                    else:
                        try:
                            retrieval_trace, context = pipeline.prepare_context(
                                case["question"],
                                top_k_each=route["reranker_pool_size"],
                                fetch_evidence=True,
                            )
                        except Exception as exc:
                            errors.append(f"retrieval/assembly {type(exc).__name__}: {exc}")
                            context = None
                    generator_context_text = None
                    if context is not None:
                        generator_context_text = pipeline._generator_context_text(context)
                    response = None
                    if context is not None:
                        try:
                            response = pipeline._generate_with_llm(case["question"], context, retrieval_trace)
                            pipeline._audit_citations(response, context)
                        except Exception as exc:
                            errors.append(f"generation/audit {type(exc).__name__}: {exc}")
                    citation = None
                    if response is not None:
                        output = [
                            EvidenceRef(
                                session_id=item.session_id,
                                turn_id=item.turn_id,
                                speaker=item.speaker,
                                quote_text=item.quote_text,
                            )
                            for item in response.dialogue_citations
                        ]
                        retrieved = [
                            EvidenceRef(
                                session_id=int(item["session_id"]),
                                turn_id=int(item["turn_id"]),
                                speaker=item["speaker"],
                                quote_text=item.get("quote_text", item.get("text", "")),
                            )
                            for item in (context.evidence_turns if context else [])
                        ]
                        citation = _safe(audit_grounding(output, authoritative, required, retrieved_evidence=retrieved).__dict__)
                    context_coverage = _context_coverage(
                        case,
                        required,
                        context.evidence_turns if context is not None else [],
                        context.deepeval_context_nodes if context is not None else [],
                        parent_cards=(retrieval_trace.get("anchored_parent_cards", []) if retrieval_trace else []),
                        candidate_units=(retrieval_trace.get("evidence_units", []) if retrieval_trace else []),
                    )
                    trace = {
                        "case_id": case_id,
                        "track": track,
                        "split": split_assignments[f"eedi-l1-{index:04d}"],
                        "query": case["question"],
                        "status": "SUCCESS" if response is not None and not errors else ("ERROR" if errors else "UNMEASURED"),
                        "errors": errors,
                        "answer": response.answer_content if response is not None else None,
                        "citations": [item.model_dump() for item in response.dialogue_citations] if response is not None else [],
                        "final_context": context.prompt_context_markdown if context is not None else None,
                        "generator_context": generator_context_text,
                        "generator_contract_version": args.generator_contract,
                        "generator_contract_repair_attempted": (
                            response.__dict__.get("generator_contract_repair_attempted", False)
                            if response is not None
                            else False
                        ),
                        "generator_contract_auto_merged_ids": (
                            response.__dict__.get("generator_contract_auto_merged_ids", [])
                            if response is not None
                            else []
                        ),
                        "gold_context_version": args.gold_context_version,
                        # Preserve the validated claim-level contract for auditability.
                        # Citation materialization remains backend-owned; this field is
                        # only the model's structured claim ledger.
                        "generator_claims": _safe(
                            getattr(response, "__dict__", {}).get("generator_claims", [])
                        )
                        if response is not None
                        else [],
                        "generator_core_answer": (
                            response.__dict__.get("generator_core_answer")
                            if response is not None
                            else None
                        ),
                        "generator_grounding_text": (
                            response.__dict__.get("generator_grounding_text")
                            if response is not None
                            else None
                        ),
                        "claim_coverage": (
                            summarize_claim_coverage(response)
                            if response is not None and response.__dict__.get("generator_contract_version") == "v2"
                            else None
                        ),
                    "context_coverage": context_coverage,
                        "context_budget_tokens": (
                            int(context.context_budget_tokens)
                            if context is not None
                            else 0
                        ),
                        "catalog_token_count": (
                            int(context.catalog_token_count)
                            if context is not None
                            else 0
                        ),
                        "generator_context_token_count": (
                            estimate_text_tokens(generator_context_text)
                            if generator_context_text
                            else 0
                        ),
                        "budget_violation": bool(
                            context is not None and context.budget_violation
                        ),
                        "gold_alignment_warnings": gold_alignment_warnings,
                        "deepeval_context_node_count": (
                            len(context.deepeval_context_nodes)
                            if context is not None and context.deepeval_context_nodes
                            else (1 if context is not None else 0)
                        ),
                        "generator_context_chars": (
                            len(generator_context_text)
                            if generator_context_text is not None
                            else 0
                        ),
                        "retrieval_trace": _safe(retrieval_trace),
                        "assembler_evidence": _safe(context.evidence_turns if context is not None else []),
                        "citation_audit": citation,
                        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    }
                    trace_rows.append(trace)
                    case_trace_rows.append(trace)
                    if citation is not None:
                        citation_score = min(
                            float(citation.get("citation_precision") or 0.0),
                            float(citation.get("citation_recall") or 0.0),
                            float(citation.get("role_accuracy") or 0.0),
                            float(citation.get("quote_grounding_precision") or 0.0),
                            float(citation.get("candidate_authorization_rate") or 0.0),
                        )
                        metric_row = {
                            "case_id": case_id,
                            "track": track,
                            "metric": "citation_audit",
                            "status": "SUCCESS",
                            "score": citation_score,
                            "threshold": 0.95,
                            "passed": citation_score >= 0.95,
                            "reason": "minimum of precision/recall/role/quote/authorization",
                            "latency_ms": 0.0,
                        }
                        metric_rows.append(metric_row)
                        case_metric_rows.append(metric_row)
                    if response is not None and context is not None and judge_metrics:
                        deepeval_rows = _run_deepeval_case(
                            DeepEvalCasePayload(
                                case_id=f"{case_id}-{track}",
                                input=case["question"],
                                actual_output=response.answer_content or response.misconception_diagnosis,
                                expected_output=case["ground_truth"],
                                final_retrieval_context=(
                                    list(context.deepeval_context_nodes)
                                    if context.deepeval_context_nodes
                                    else [generator_context_text or context.prompt_context_markdown]
                                ),
                            ),
                            judge_metrics,
                            track=track,
                            case_id=case_id,
                            actual_output_by_metric=(
                                {
                                    "answer_relevancy": response.__dict__.get("generator_core_answer"),
                                    "faithfulness": response.__dict__.get("generator_grounding_text"),
                                }
                                if response.__dict__.get("generator_core_answer")
                                else None
                            ),
                            retrieval_context=(
                                list(context.deepeval_context_nodes)
                                if context.deepeval_context_nodes
                                else [generator_context_text or context.prompt_context_markdown]
                            ),
                        )
                        metric_rows.extend(deepeval_rows)
                        case_metric_rows.extend(deepeval_rows)
                _append_checkpoint(checkpoint_dir, case_trace_rows, case_metric_rows)
        finally:
            if storage is not None:
                storage.close()
            try:
                from chromadb.api.client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except Exception:
                pass
            gc.collect()
    after_hash = hash_storage_artifacts(db_path, chroma_path)
    if after_hash != artifact_hash:
        raise RuntimeError("L2 evaluation changed the source artifact")
    trace_path = args.output_root.resolve() / args.run_id / "traces.jsonl"
    run_dir = trace_path.parent
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in trace_rows), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metric_rows), encoding="utf-8")
    readiness_payload = _safe(readiness.model_dump() if hasattr(readiness, "model_dump") else readiness)
    metric_aggregates: list[dict[str, Any]] = []
    for track in ("gold", "real"):
        for metric in sorted({row["metric"] for row in metric_rows}):
            values = [row["score"] for row in metric_rows if row["track"] == track and row["metric"] == metric and row["status"] == "SUCCESS"]
            threshold = next((row["threshold"] for row in metric_rows if row["track"] == track and row["metric"] == metric), None)
            metric_aggregates.append({"track": track, "metric": metric, "measured": len(values), "mean": sum(values) / len(values) if values else None, "threshold": threshold})
    errors = [row for row in trace_rows if row["status"] == "ERROR"] + [row for row in metric_rows if row["status"] == "ERROR"]
    diagnostics = _aggregate_trace_diagnostics(trace_rows, metric_rows)
    judge_ready = readiness_payload.get("status") == "SUCCESS"
    all_thresholds_pass = bool(metric_aggregates) and all(item["mean"] is not None and item["mean"] >= item["threshold"] for item in metric_aggregates)
    if l1_closeout.get("decision") != "GO_TO_L2_TECHNICAL":
        release_decision = "BLOCKED_L1_PRECONDITION"
    elif not judge_ready:
        release_decision = "BLOCKED_JUDGE_UNMEASURED"
    elif errors:
        release_decision = "BLOCKED_ERRORS"
    elif diagnostics.get("gold_alignment_warning_count", 0):
        release_decision = "BLOCKED_GOLD_DATA_CONFLICT"
    elif not all_thresholds_pass:
        release_decision = "BLOCKED_METRIC_GATE"
    else:
        release_decision = "L2_PASSED"
    report = {
        "schema_version": "l2-deepeval-report/v1",
        "run_id": args.run_id,
        "dataset": dataset_manifest,
        "artifact_hash_expectation": {
            "expected": expected_artifact_hash,
            "matches_frozen_l1_manifest": matches_frozen_manifest,
            "override_used": args.expected_artifact_hash is not None,
            "frozen_l1_manifest_hash": l1_manifest["artifact_hashes"]["qwen_combined_sha256"],
        },
        "runtime_route": {
            **route,
            "retrieval_mode": args.retrieval_mode,
            "bm25_weight": args.bm25_weight,
            "chunk_strategy": "card",
            "window_size": 6,
            "window_step": 3,
            "max_prompt_tokens": args.max_prompt_tokens,
            "generator_timeout_seconds": args.generator_timeout_seconds,
            "generator_contract": args.generator_contract,
            "gold_context_version": args.gold_context_version,
            "rubric_version": args.rubric_version,
        },
        "l1_precondition": l1_closeout.get("decision"),
        "judge_readiness": readiness_payload,
        "judge_config": {
            "provider": settings.evaluation_provider if settings else None,
            "model": settings.evaluation_model if settings else None,
            "base_url_configured": bool(settings and settings.base_url),
            "api_key_env": settings.api_key_env if settings else None,
            "temperature": settings.temperature if settings else None,
            "deepeval_version": readiness_payload.get("deepeval_version"),
        },
        "case_shard": {
            "case_start": args.case_start,
            "case_count": len(selected_cases),
            "case_end": args.case_start + len(selected_cases) - 1,
            "golden_total": len(golden_all),
        },
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "resume_requested": args.resume,
            "completed_case_count": len(
                {
                    int(row["case_id"].rsplit("-", 1)[-1])
                    for row in trace_rows
                    if row.get("track") == "gold"
                }
            ),
        },
        "tracks": {"gold_case_count": sum(row["track"] == "gold" for row in trace_rows), "real_case_count": sum(row["track"] == "real" for row in trace_rows)},
        "metric_aggregates": metric_aggregates,
        "diagnostics": diagnostics,
        "gold_alignment": _gold_alignment_status(diagnostics),
        "trace_status_counts": {status: sum(row["status"] == status for row in trace_rows) for status in ("SUCCESS", "ERROR", "UNMEASURED")},
        "artifact_hash_before": artifact_hash,
        "artifact_hash_after": after_hash,
        "artifact_mutated": artifact_hash != after_hash,
        "errors": errors,
        "release_decision": release_decision,
        "gold_real_interpretation": "Gold high/Real low indicates upstream context loss; both low indicates generator or data issue; interpret only when judge results are measured.",
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# L2 DeepEval Report",
        "",
        f"Release decision: **{release_decision}**",
        f"L1 precondition: `{l1_closeout.get('decision')}`",
        f"Judge readiness: `{readiness_payload.get('status')}`",
        f"Source artifact unchanged: `{artifact_hash == after_hash}`",
        "",
        "## Metric aggregates",
        "",
        "| Track | Metric | Measured | Mean | Threshold |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in metric_aggregates:
        lines.append(f"| {item['track']} | {item['metric']} | {item['measured']} | {item['mean']} | {item['threshold']} |")
    lines.extend(["", "## Context coverage diagnostics", "", "| Track | Parent Pointer | Candidate Turn | Final Turn | Claim Recall (proxy) | DeepEval Recall | Faithfulness |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for track in ("gold", "real"):
        coverage = diagnostics["context_coverage"][track]
        deep_eval = diagnostics["deep_eval"][track]
        lines.append(
            f"| {track} | {coverage['mean_parent_pointer_recall']} | {coverage['mean_candidate_turn_recall']} | "
            f"{coverage['mean_turn_recall']} | {coverage['mean_claim_recall']} | "
            f"{deep_eval['mean_contextual_recall']} | {deep_eval['mean_faithfulness']} |"
        )
    lines.extend(["", "## Reproduce", "", f"`.venv\\Scripts\\python.exe scripts\\run_l2_deepeval.py --db {args.db} --chroma {args.chroma} --l1-closeout {args.l1_closeout} --l1-manifest {args.l1_manifest} --expected-artifact-hash {expected_artifact_hash} --dataset-output {args.dataset_output} --output-root {args.output_root} --run-id <new-run-id> --case-start {args.case_start} --max-cases {args.max_cases} --generator-timeout-seconds {args.generator_timeout_seconds} --max-prompt-tokens {args.max_prompt_tokens} --judge-model {judge_model or '<DEEPEVAL_MODEL>'} --judge-api-key-env {judge_key_env or '<DEEPEVAL_API_KEY_ENV>'} --judge-base-url {judge_base_url or '<DEEPEVAL_BASE_URL>'} --reranker-backend {route['reranker_backend']} --rerank-unit {route['rerank_unit']} --reranker-pool-size {route['reranker_pool_size']} --parent-card-count {route['parent_card_count']} --evidence-selection-count {route['evidence_selection_count']} --retrieval-mode {args.retrieval_mode} --bm25-weight {args.bm25_weight} --allow-l1-blocked`", ""])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "release_decision": release_decision, "judge_status": readiness_payload.get("status"), "trace_count": len(trace_rows), "metric_count": len(metric_rows), "report_dir": str(run_dir), "artifact_hash": artifact_hash}, ensure_ascii=False))
    return 0 if release_decision == "L2_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
