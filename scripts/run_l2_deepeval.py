from __future__ import annotations

"""Build and run the L2 Gold/Real end-to-end DeepEval loop."""

import argparse
import gc
import hashlib
import json
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


def _metrics(judge_model: Any) -> list[tuple[str, Any, float]]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import SingleTurnParams

    return [
        ("faithfulness", FaithfulnessMetric(threshold=None, model=judge_model, async_mode=False, verbose_mode=False), 0.90),
        ("answer_relevancy", AnswerRelevancyMetric(threshold=None, model=judge_model, async_mode=False, verbose_mode=False), 0.85),
        ("contextual_recall", ContextualRecallMetric(threshold=None, model=judge_model, async_mode=False, verbose_mode=False), 0.90),
        ("contextual_precision", ContextualPrecisionMetric(threshold=None, model=judge_model, async_mode=False, verbose_mode=False), 0.85),
        (
            "pedagogical_geval",
            GEval(
                name="PedagogicalAdaptation",
                evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
                criteria="The answer is useful to an education researcher, separates student misconception from tutor strategy, and proposes an evidence-grounded pedagogical intervention without inventing facts.",
                model=judge_model,
                threshold=None,
                async_mode=False,
            ),
            0.80,
        ),
    ]


def _run_deepeval_case(payload: DeepEvalCasePayload, metrics: list[tuple[str, Any, float]], *, track: str, case_id: str) -> list[dict[str, Any]]:
    test_case = build_deepeval_test_case(payload)
    rows: list[dict[str, Any]] = []
    for metric_name, metric, threshold in metrics:
        started = time.perf_counter()
        try:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--chroma", type=Path, required=True)
    parser.add_argument("--l1-closeout", type=Path, required=True)
    parser.add_argument("--l1-manifest", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key-env", default=None)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--allow-l1-blocked", action="store_true")
    args = parser.parse_args(argv)
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
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
    if artifact_hash != l1_manifest["artifact_hashes"]["qwen_combined_sha256"]:
        raise ValueError("L2 artifact hash does not match frozen L1 manifest")
    dataset_manifest = build_l2_dataset(
        output_dir=args.dataset_output.resolve(),
        golden_path=(PROJECT_ROOT / "data/golden_test_set.json"),
        l1_manifest_path=args.l1_manifest.resolve(strict=True),
        artifact_hash=artifact_hash,
        generator_model=__import__("os").getenv("LLM_MODEL", "explicit-generator-model"),
        generator_config={"mode": "production_llm", "citation_contract": "quadruple"},
    )
    l2_split_manifest = json.loads((args.dataset_output.resolve() / "split_manifest.json").read_text(encoding="utf-8"))
    split_assignments = l2_split_manifest["query_assignments"]
    golden = json.loads((PROJECT_ROOT / "data/golden_test_set.json").read_text(encoding="utf-8"))[: args.max_cases]
    sessions = _load_sessions(PROJECT_ROOT / "data/cleaned_sessions.jsonl")
    provider = SiliconFlowQwen3EmbeddingFunction.from_env()
    scratch_root = args.output_root.resolve() / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    trace_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
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
                judge_metrics = _metrics(judge_model_obj)
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
            retriever = DualMetricRetriever(storage_manager=storage, query_rewrite_mode="deterministic", fusion_strategy="raw_first")
            assembler = PedagogicalGoldAssembler(lambda_diversity=0.7, max_prompt_tokens=1500)
            pipeline = EndToEndPedagogicalRAGPipeline(retriever=retriever, assembler=assembler, model_name=__import__("os").getenv("LLM_MODEL"))
            for index, case in enumerate(golden, 1):
                case_id = f"eedi-l2-{index:04d}"
                required, authoritative = _evidence_refs(case, sessions)
                for track in ("gold", "real"):
                    started = time.perf_counter()
                    retrieval_trace: dict[str, Any] = {}
                    errors: list[str] = []
                    if track == "gold":
                        context = _build_gold_context(case, sessions)
                        retrieval_trace = {"policy": "gold_context_evaluation_only", "retrieved_evidence": [item.model_dump() for item in required]}
                    else:
                        try:
                            retrieval_trace = retriever.retrieve_multi_perspective_rrf(case["question"], top_k_each=5, fetch_evidence=True)
                            context = assembler.assemble(case["question"], retrieval_trace)
                        except Exception as exc:
                            errors.append(f"retrieval/assembly {type(exc).__name__}: {exc}")
                            context = None
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
                        "retrieval_trace": _safe(retrieval_trace),
                        "assembler_evidence": _safe(context.evidence_turns if context is not None else []),
                        "citation_audit": citation,
                        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    }
                    trace_rows.append(trace)
                    if citation is not None:
                        citation_score = min(
                            float(citation.get("citation_precision") or 0.0),
                            float(citation.get("citation_recall") or 0.0),
                            float(citation.get("role_accuracy") or 0.0),
                            float(citation.get("quote_grounding_precision") or 0.0),
                            float(citation.get("candidate_authorization_rate") or 0.0),
                        )
                        metric_rows.append({
                            "case_id": case_id,
                            "track": track,
                            "metric": "citation_audit",
                            "status": "SUCCESS",
                            "score": citation_score,
                            "threshold": 0.95,
                            "passed": citation_score >= 0.95,
                            "reason": "minimum of precision/recall/role/quote/authorization",
                            "latency_ms": 0.0,
                        })
                    if response is not None and context is not None and judge_metrics:
                        metric_rows.extend(_run_deepeval_case(
                            DeepEvalCasePayload(
                                case_id=f"{case_id}-{track}",
                                input=case["question"],
                                actual_output=response.answer_content or response.misconception_diagnosis,
                                expected_output=case["ground_truth"],
                                final_retrieval_context=[context.prompt_context_markdown],
                            ),
                            judge_metrics,
                            track=track,
                            case_id=case_id,
                        ))
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
    judge_ready = readiness_payload.get("status") == "SUCCESS"
    all_thresholds_pass = bool(metric_aggregates) and all(item["mean"] is not None and item["mean"] >= item["threshold"] for item in metric_aggregates)
    if l1_closeout.get("decision") != "GO_TO_L2_TECHNICAL":
        release_decision = "BLOCKED_L1_PRECONDITION"
    elif not judge_ready:
        release_decision = "BLOCKED_JUDGE_UNMEASURED"
    elif errors:
        release_decision = "BLOCKED_ERRORS"
    elif not all_thresholds_pass:
        release_decision = "BLOCKED_METRIC_GATE"
    else:
        release_decision = "L2_PASSED"
    report = {
        "schema_version": "l2-deepeval-report/v1",
        "run_id": args.run_id,
        "dataset": dataset_manifest,
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
        "tracks": {"gold_case_count": sum(row["track"] == "gold" for row in trace_rows), "real_case_count": sum(row["track"] == "real" for row in trace_rows)},
        "metric_aggregates": metric_aggregates,
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
    lines.extend(["", "## Reproduce", "", f"`.venv\\Scripts\\python.exe scripts\\run_l2_deepeval.py --db {args.db} --chroma {args.chroma} --l1-closeout {args.l1_closeout} --l1-manifest {args.l1_manifest} --dataset-output {args.dataset_output} --output-root {args.output_root} --run-id <new-run-id> --max-cases {args.max_cases} --judge-model {judge_model or '<DEEPEVAL_MODEL>'} --judge-api-key-env {judge_key_env or '<DEEPEVAL_API_KEY_ENV>'} --judge-base-url {judge_base_url or '<DEEPEVAL_BASE_URL>'} --allow-l1-blocked`", ""])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "release_decision": release_decision, "judge_status": readiness_payload.get("status"), "trace_count": len(trace_rows), "metric_count": len(metric_rows), "report_dir": str(run_dir), "artifact_hash": artifact_hash}, ensure_ascii=False))
    return 0 if release_decision == "L2_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
