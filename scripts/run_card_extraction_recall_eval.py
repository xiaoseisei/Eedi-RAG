from __future__ import annotations

"""Run a small, real-LLM A/B evaluation of card source-turn completeness.

The comparison uses the read-only DuckDB cards that are the source of the
frozen Qwen index as baseline and re-extracts only selected pointer-loss
sessions with the current evidence-completeness prompt. It never writes the
production cards, DuckDB, or Chroma artifacts.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - the project environment includes dotenv
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_knowledge import EVIDENCE_AUDIT_VERSION, EXTRACTION_PROMPT_VERSION, extract_knowledge_from_session
from src.models import CleanedSession


DEFAULT_SESSION_IDS = (23, 14, 206, 33, 287, 469)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.resolve(strict=True).read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_sessions(path: Path) -> dict[int, CleanedSession]:
    sessions: dict[int, CleanedSession] = {}
    for row in _jsonl(path):
        session = CleanedSession.model_validate(row)
        sessions[session.intervention_id] = session
    return sessions


def _load_cards_from_db(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    import duckdb

    connection = duckdb.connect(database=str(path.resolve(strict=True)), read_only=True)
    cards: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        for slot, table in (("misconception", "misconception_chunks"), ("strategy", "tutor_strategy_chunks")):
            for session_id, source_turn_ids in connection.execute(
                f"SELECT session_id, source_turn_ids FROM {table}"
            ).fetchall():
                cards[(int(session_id), slot)] = {"session_id": int(session_id), "source_turn_ids": list(source_turn_ids or [])}
    finally:
        connection.close()
    return cards


def _target_turn_ids(case: dict[str, Any], session: CleanedSession) -> list[int]:
    target_speaker = case["pointer_coverage"]["target_speaker"]
    target_is_tutor = target_speaker == "tutor"
    turn_map = {turn.turn_id: turn for turn in session.turns}
    required = {
        int(pair[1])
        for pair in case.get("required_turns", [])
        if int(pair[0]) == session.intervention_id
    }
    return sorted(turn_id for turn_id in required if turn_id in turn_map and turn_map[turn_id].is_tutor == target_is_tutor)


def _recall(source_turn_ids: list[int], target_turn_ids: list[int]) -> float | None:
    if not target_turn_ids:
        return None
    return len(set(source_turn_ids) & set(target_turn_ids)) / len(set(target_turn_ids))


def _card_ids(card: dict[str, Any] | None) -> list[int]:
    if not card:
        return []
    return [int(turn_id) for turn_id in card.get("source_turn_ids", [])]


def _parse_ids(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return list(DEFAULT_SESSION_IDS)
    ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not ids:
        raise ValueError("--session-ids 不能为空")
    return list(dict.fromkeys(ids))


def _safe_error(exc: Exception) -> str:
    # Keep the report useful without copying potentially large provider payloads.
    text = str(exc).replace("\n", " ").strip()
    return text[:500]


def build_eval(
    *,
    sessions_path: Path,
    existing_db_path: Path,
    graded_cases_path: Path,
    output_dir: Path,
    session_ids: list[int],
    model: str | None = None,
    base_url: str | None = None,
    max_retries: int = 3,
    evidence_audit: bool = False,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite extraction eval directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("LLM_API_KEY")
    effective_model = model or os.getenv("LLM_MODEL")
    effective_base_url = base_url or os.getenv("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError("LLM_API_KEY 未配置，无法进行真实抽取评测；拒绝伪造结果")
    if not effective_model:
        raise RuntimeError("LLM_MODEL 未配置，无法进行真实抽取评测；拒绝隐式模型默认值")
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - dependency is in .venv
        raise RuntimeError("缺少 openai 依赖，无法进行真实抽取评测") from exc

    sessions = _load_sessions(sessions_path)
    existing_cards = _load_cards_from_db(existing_db_path)
    cases_by_session = {
        int(case["session_id"]): case
        for case in _jsonl(graded_cases_path)
        if int(case["session_id"]) in set(session_ids)
        and float(case["pointer_coverage"]["role_correct_card_pointer_recall"]) < 1.0
        and not bool(case.get("card_slot_fallback_due_to_missing_role_evidence"))
    }
    missing_case_ids = [session_id for session_id in session_ids if session_id not in cases_by_session]
    if missing_case_ids:
        raise ValueError(f"指定会话没有可评估的非 fallback pointer-loss case: {missing_case_ids}")
    missing_sessions = [session_id for session_id in session_ids if session_id not in sessions]
    if missing_sessions:
        raise ValueError(f"cleaned sessions 缺少指定会话: {missing_sessions}")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds 必须为正数")
    client_kwargs["timeout"] = timeout_seconds
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url
    client = openai.OpenAI(**client_kwargs)

    rows: list[dict[str, Any]] = []
    for session_id in session_ids:
        session = sessions[session_id]
        case = cases_by_session[session_id]
        slot = case["pointer_coverage"]["target_slot"]
        target_ids = _target_turn_ids(case, session)
        baseline_ids = _card_ids(existing_cards.get((session_id, slot)))
        baseline_recall = _recall(baseline_ids, target_ids)
        row: dict[str, Any] = {
            "case_id": case["case_id"],
            "session_id": session_id,
            "category": case["category"],
            "target_slot": slot,
            "target_speaker": case["pointer_coverage"]["target_speaker"],
            "target_turn_ids": target_ids,
            "baseline_source_turn_ids": baseline_ids,
            "baseline_pointer_recall": baseline_recall,
            "prompt_version": EXTRACTION_PROMPT_VERSION,
            "evidence_audit": evidence_audit,
            "evidence_audit_version": EVIDENCE_AUDIT_VERSION if evidence_audit else None,
            "model": effective_model,
        }
        try:
            extracted = extract_knowledge_from_session(
                session,
                llm_client=client,
                model=effective_model,
                max_retries=max_retries,
                evidence_audit=evidence_audit,
            )
            if extracted.extraction_status != "success" or extracted.misconception is None or extracted.tutor_strategy is None:
                raise RuntimeError("抽取返回非 success 或缺少卡片")
            new_card = extracted.misconception if slot == "misconception" else extracted.tutor_strategy
            new_ids = list(new_card.source_turn_ids)
            new_recall = _recall(new_ids, target_ids)
            row.update(
                {
                    "status": "SUCCESS",
                    "new_source_turn_ids": new_ids,
                    "new_pointer_recall": new_recall,
                    "new_missing_target_turn_ids": sorted(set(target_ids) - set(new_ids)),
                    "new_card": new_card.model_dump(),
                }
            )
        except Exception as exc:  # preserve explicit failure in the result row
            row.update({"status": "FAILED", "error": _safe_error(exc), "new_source_turn_ids": [], "new_pointer_recall": None})
        rows.append(row)

    measured = [row for row in rows if row["status"] == "SUCCESS" and row["new_pointer_recall"] is not None]
    baseline_values = [float(row["baseline_pointer_recall"]) for row in rows if row["baseline_pointer_recall"] is not None]
    new_values = [float(row["new_pointer_recall"]) for row in measured]
    conservative_values = [float(row["new_pointer_recall"]) if row["status"] == "SUCCESS" and row["new_pointer_recall"] is not None else 0.0 for row in rows]
    by_slot: dict[str, dict[str, Any]] = {}
    for slot in sorted({row["target_slot"] for row in rows}):
        slot_rows = [row for row in rows if row["target_slot"] == slot]
        slot_success = [row for row in slot_rows if row["status"] == "SUCCESS" and row["new_pointer_recall"] is not None]
        by_slot[slot] = {
            "case_count": len(slot_rows),
            "success_count": len(slot_success),
            "baseline_mean": mean(float(row["baseline_pointer_recall"]) for row in slot_rows),
            "new_mean_measured": mean(float(row["new_pointer_recall"]) for row in slot_success) if slot_success else None,
            "new_mean_conservative": mean(float(row["new_pointer_recall"]) if row["status"] == "SUCCESS" and row["new_pointer_recall"] is not None else 0.0 for row in slot_rows),
        }

    summary = {
        "schema_version": "card-extraction-recall-eval/v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "MEASURED" if measured else "FAILED",
        "scope": "small real-LLM re-extraction of selected non-fallback pointer-loss cases; production artifacts are read-only",
        "prompt_version": EXTRACTION_PROMPT_VERSION,
        "evidence_audit": evidence_audit,
        "evidence_audit_version": EVIDENCE_AUDIT_VERSION if evidence_audit else None,
        "model": effective_model,
        "timeout_seconds": timeout_seconds,
        "case_count": len(rows),
        "success_count": len(measured),
        "failure_count": len(rows) - len(measured),
        "baseline_pointer_recall_mean": mean(baseline_values) if baseline_values else None,
        "new_pointer_recall_mean_measured": mean(new_values) if new_values else None,
        "new_pointer_recall_mean_conservative": mean(conservative_values) if conservative_values else None,
        "delta_measured_vs_baseline_on_success_cases": (
            mean(float(row["new_pointer_recall"]) - float(row["baseline_pointer_recall"]) for row in measured)
            if measured else None
        ),
        "improved_case_count": sum(
            row["status"] == "SUCCESS" and float(row["new_pointer_recall"]) > float(row["baseline_pointer_recall"])
            for row in rows
            if row["status"] == "SUCCESS"
        ),
        "unchanged_case_count": sum(
            row["status"] == "SUCCESS" and float(row["new_pointer_recall"]) == float(row["baseline_pointer_recall"])
            for row in rows
        ),
        "worse_case_count": sum(
            row["status"] == "SUCCESS" and float(row["new_pointer_recall"]) < float(row["baseline_pointer_recall"])
            for row in rows
        ),
        "by_slot": by_slot,
        "inputs": {
            "sessions": {"path": str(sessions_path.resolve()), "sha256": _sha256(sessions_path)},
            "existing_db": {"path": str(existing_db_path.resolve()), "sha256": _sha256(existing_db_path)},
            "graded_cases": {"path": str(graded_cases_path.resolve()), "sha256": _sha256(graded_cases_path)},
            "session_ids": session_ids,
        },
        "limitations": [
            "This is a small technical A/B sample, not a full 100-card re-extraction or human judgment.",
            "Labels come from provisional graded qrels and should not be interpreted as business truth.",
            "A successful extraction with more pointers may improve recall while requiring a separate precision/noise audit.",
            "No model switch is tested in this run; the comparison isolates the evidence-completeness prompt and optional second-pass audit.",
        ],
    }
    manifest = {
        "schema_version": "card-extraction-recall-eval-manifest/v1",
        "summary_sha256": "",
        "result_sha256": "",
        "summary": summary,
    }
    result_path = output_dir / "results.jsonl"
    result_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["summary_sha256"] = _sha256(summary_path)
    manifest["result_sha256"] = _sha256(result_path)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Card Extraction Turn Recall Small Eval",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"- Model: `{effective_model}`; prompt: `{EXTRACTION_PROMPT_VERSION}`; evidence audit: `{evidence_audit}`",
        f"- Cases: `{summary['case_count']}`; success/failure: `{summary['success_count']}/{summary['failure_count']}`",
        f"- Baseline pointer recall mean: `{summary['baseline_pointer_recall_mean']}`",
        f"- New pointer recall mean (measured): `{summary['new_pointer_recall_mean_measured']}`",
        f"- New pointer recall mean (failures counted as 0, conservative): `{summary['new_pointer_recall_mean_conservative']}`",
        f"- Measured delta on same successful cases: `{summary['delta_measured_vs_baseline_on_success_cases']}`",
        f"- Improved/unchanged/worse: `{summary['improved_case_count']}/{summary['unchanged_case_count']}/{summary['worse_case_count']}`",
        "",
        "This run isolates the prompt/contract and optional evidence-audit change with the existing model; it does not claim a model replacement is necessary.",
        "Full per-case source IDs and generated cards are in `results.jsonl`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=PROJECT_ROOT / "data" / "cleaned_sessions.jsonl")
    parser.add_argument("--existing-db", type=Path, default=PROJECT_ROOT / "data" / "db" / "tutoring_knowledge.duckdb")
    parser.add_argument("--graded-cases", type=Path, default=PROJECT_ROOT / "evals" / "datasets" / "turn-card-graded-provisional-20260903-110000" / "raw_cases.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-ids", type=str, default=None, help="逗号分隔的 session IDs；默认 6 个代表性 pointer-loss case")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--evidence-audit", action="store_true", help="在首轮抽取后启用第二遍最小充分证据审计")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="单次 LLM 请求超时秒数")
    args = parser.parse_args(argv)
    summary = build_eval(
        sessions_path=args.sessions.resolve(strict=True),
        existing_db_path=args.existing_db.resolve(strict=True),
        graded_cases_path=args.graded_cases.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        session_ids=_parse_ids(args.session_ids),
        model=args.model,
        base_url=args.base_url,
        max_retries=args.max_retries,
        evidence_audit=args.evidence_audit,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "status": summary["status"], "success_count": summary["success_count"], "new_pointer_recall_mean_measured": summary["new_pointer_recall_mean_measured"]}, ensure_ascii=False))
    return 0 if summary["status"] == "MEASURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
