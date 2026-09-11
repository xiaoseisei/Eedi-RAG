"""Run the actual graph-business route against the approved Gold dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business_graph_metrics import evaluate_case, evaluate_dataset, load_gold_cases
from src.business_graph_pipeline import BusinessGraphPipeline, BusinessGraphRuntimeConfig
from src.intent_classifier import OpenAICompatibleIntentClassifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("review_status") != "HUMAN_APPROVED_FOR_THIS_EVALUATION":
        raise ValueError("Gold manifest is not HUMAN_APPROVED_FOR_THIS_EVALUATION")
    if int(payload.get("case_count", -1)) != 83:
        raise ValueError("business-graph-v1 manifest must contain exactly 83 cases")
    return payload


def _read_jsonl_view(path: Path, view_name: str) -> list[dict[str, Any]]:
    """Read one derived view with strict, unique case IDs."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not str(row.get("case_id", "")):
            raise ValueError(f"{view_name} has a blank/invalid case_id at line {line_number}")
        case_id = str(row["case_id"])
        if case_id in seen:
            raise ValueError(f"{view_name} has duplicate case_id: {case_id}")
        seen.add(case_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"{view_name} is empty: {path}")
    return rows


def load_exported_cases(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Join the lossless evaluator views without exposing expected data to SUT."""

    root = Path(dataset_dir).resolve(strict=True)
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "business-graph-dataset-split/v1":
        raise ValueError("unsupported business graph split dataset schema")
    views = {
        "queries": _read_jsonl_view(root / "queries.jsonl", "queries"),
        "facts": _read_jsonl_view(root / "expected_facts.jsonl", "expected_facts"),
        "paths": _read_jsonl_view(root / "expected_graph_paths.jsonl", "expected_graph_paths"),
        "evidence": _read_jsonl_view(root / "expected_evidence.jsonl", "expected_evidence"),
    }
    ids = {name: [str(row["case_id"]) for row in rows] for name, rows in views.items()}
    expected_ids = ids["queries"]
    if any(view_ids != expected_ids for name, view_ids in ids.items() if name != "queries"):
        raise ValueError("business graph split views do not have identical ordered case IDs")
    manifest_ids = [str(item) for item in manifest.get("case_ids", [])]
    if manifest_ids != expected_ids:
        raise ValueError("split manifest case_ids do not match query view")
    if int(manifest.get("case_count", -1)) != len(expected_ids):
        raise ValueError("split manifest case_count does not match views")

    by_name = {
        name: {str(row["case_id"]): row for row in rows}
        for name, rows in views.items()
    }
    cases: list[dict[str, Any]] = []
    for case_id in expected_ids:
        query = by_name["queries"][case_id]
        case = {
            **query,
            **{key: value for key, value in by_name["facts"][case_id].items() if key != "case_id"},
            **{key: value for key, value in by_name["paths"][case_id].items() if key != "case_id"},
            **{key: value for key, value in by_name["evidence"][case_id].items() if key != "case_id"},
        }
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"split dataset query is blank: {case_id}")
        cases.append(case)
    return cases


def artifact_snapshot(source_db: Path, graph_db: Path, chroma_dir: Path) -> dict[str, Any]:
    sqlite = chroma_dir / "chroma.sqlite3"
    return {
        "source_db": {"path": str(source_db), "sha256": sha256(source_db)},
        "graph_db": {"path": str(graph_db), "sha256": sha256(graph_db)},
        "chroma_sqlite": {"path": str(sqlite), "sha256": sha256(sqlite)},
    }


def load_intent_classifier_from_env(
    model: str | None = None,
) -> OpenAICompatibleIntentClassifier | None:
    """Load the optional second-layer classifier without hiding configuration errors.

    ``try_from_env`` returns ``None`` only when ``INTENT_MODEL`` (and the
    explicit override) is absent.  Once a model is configured, malformed
    credentials, dependencies, or client parameters must propagate to the
    evaluator so a broken setup cannot be reported as an unmeasured run.
    The adapter itself reuses ``EMBEDDING_BASE_URL`` and
    ``EMBEDDING_API_KEY_ENV`` when intent-specific values are absent.
    """

    return OpenAICompatibleIntentClassifier.try_from_env(model=model)


def validate_graph_provenance(manifest: dict[str, Any], *, source_db: str | Path, graph_db: str | Path) -> dict[str, Any]:
    """Validate a candidate graph without silently promoting it to Gold input."""

    import duckdb

    source_path = Path(source_db).resolve(strict=True)
    graph_path = Path(graph_db).resolve(strict=True)
    source_hash = sha256(source_path)
    graph_hash = sha256(graph_path)
    connection = duckdb.connect(str(graph_path), read_only=True)
    try:
        ready_count = int(connection.execute("SELECT COUNT(*) FROM graph_manifest WHERE build_status='READY'").fetchone()[0])
        row = connection.execute("SELECT source_artifact_hash FROM graph_manifest WHERE build_status='READY'").fetchone()
        source_bound = bool(row and str(row[0]).lower() == source_hash.lower())
        graph_tables = {str(item[0]) for item in connection.execute("SHOW TABLES").fetchall()}
        topology_ready = {"graph_nodes", "graph_edges", "graph_label_assignments"} <= graph_tables
    finally:
        connection.close()
    return {
        "ready_manifest": ready_count == 1,
        "source_hash_matches": source_bound and source_hash.lower() == str(manifest["source_db_sha256"]).lower(),
        "gold_graph_hash_matches": graph_hash.lower() == str(manifest["graph_db_sha256"]).lower(),
        "topology_tables_present": topology_ready,
        "candidate_graph_sha256": graph_hash,
        "approval_status": "GOLD_APPROVED" if graph_hash.lower() == str(manifest["graph_db_sha256"]).lower() else "PENDING_GOLD_REAPPROVAL",
    }


def select_cases(cases: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split in {"dev", "holdout"}:
        if not all("split" in case for case in cases):
            return [], {"status": "UNMEASURED", "reason": "Gold JSONL has no split field; no dev/holdout partition was invented."}
        selected = [case for case in cases if case.get("split") == split]
        if not selected:
            return [], {"status": "UNMEASURED", "reason": f"Gold JSONL contains no {split} cases."}
        return selected, {"status": "MEASURED", "case_count": len(selected)}
    if split == "smoke":
        wanted_categories = (
            "MICRO_STUDENT_ERROR", "MICRO_TUTOR_METHOD", "MACRO_RECURRING_MISCONCEPTION",
            "MIXED_MACRO_GRAPH_MICRO", "UNSUPPORTED",
        )
        selected: list[dict[str, Any]] = []
        for category in wanted_categories:
            selected.append(next(case for case in cases if case.get("category") == category))
        return selected, {"status": "MEASURED", "case_count": len(selected), "kind": "five-case smoke"}
    return cases, {"status": "UNMEASURED", "reason": "Gold has no split field; run covers all approved cases."}


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Business Graph Gold Evaluation",
        "",
        f"- Decision: `{report['release_decision']}`",
        f"- Cases: `{report['case_count']}`",
        f"- Split: `{report['split']}` (`{report['split_status']['status']}`)",
        f"- Intent classifier: `{report['external_models']['intent_classifier']['status']}`",
        f"- Model reranker: `{report['external_models']['qwen_reranker']['status']}`",
        f"- Generator: `{report['external_models']['generator']['status']}`",
        f"- Graph approval: `{report['graph_provenance']['approval_status']}`",
        "",
        "## Hard gates",
        "",
        "| Gate | Value | Result |",
        "| --- | ---: | --- |",
    ]
    for name, value in report["metrics"].items():
        rendered = "UNMEASURED" if value is None else f"{value:.6f}"
        result = "UNMEASURED" if value is None else ("PASS" if value == 1.0 else "FAIL")
        lines.append(f"| {name} | {rendered} | {result} |")
    lines.extend([
        "",
        "## First failure stages",
        "",
        "```json",
        json.dumps(report["first_failure_by_stage"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "Failures remain failures; no threshold, Gold field, or expected status was removed.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    gold_or_dataset = parser.add_mutually_exclusive_group()
    gold_or_dataset.add_argument("--gold", type=Path)
    gold_or_dataset.add_argument("--dataset-dir", type=Path, help="lossless business-graph-v1 split dataset directory")
    parser.add_argument("--source-db", type=Path, default=ROOT / "data/db/tutoring_knowledge.duckdb")
    parser.add_argument("--graph-db", type=Path, default=ROOT / "reports/staging/session-card-graph-v1/graph-taxonomy-verify-2-20260908.duckdb")
    parser.add_argument("--chroma-dir", type=Path, default=ROOT / "data/chroma")
    parser.add_argument("--split", choices=("all", "dev", "holdout", "smoke"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("deterministic", "llm"), default="deterministic")
    parser.add_argument(
        "--intent-model",
        default=None,
        help="second-layer intent model override; defaults to INTENT_MODEL in the environment",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=("disabled", "isolated-copy"),
        default="disabled",
        help="use existing Card dense/RRF retrieval only through an isolated Chroma copy",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("deterministic", "siliconflow"),
        default="deterministic",
        help="embedding function used to encode query text against the selected Card index",
    )
    parser.add_argument(
        "--allow-candidate-graph",
        action="store_true",
        help="evaluate a READY source-bound graph whose hash is not yet Gold-approved; release remains blocked pending reapproval",
    )
    args = parser.parse_args(argv)

    manifest_path = ROOT / "data/business/business-graph-gold-v1.manifest.json"
    manifest = read_manifest(manifest_path)
    dataset_dir = args.dataset_dir.resolve(strict=True) if args.dataset_dir is not None else None
    gold_path = (args.gold or (ROOT / "data/business/business-graph-gold-v1.jsonl")).resolve(strict=True)
    source_db = args.source_db.resolve(strict=True)
    graph_db = args.graph_db.resolve(strict=True)
    chroma_dir = args.chroma_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    dataset_manifest: dict[str, Any] | None = None
    if dataset_dir is not None:
        cases = load_exported_cases(dataset_dir)
        dataset_manifest = json.loads((dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
        if str(dataset_manifest.get("source_gold_sha256", "")).lower() != str(manifest.get("gold_sha256", "")).lower():
            raise ValueError("split dataset source Gold hash does not match approved manifest")
    else:
        cases = load_gold_cases(gold_path)
    if len(cases) != int(manifest["case_count"]):
        raise ValueError("Gold case count does not match the approved manifest")
    selected_cases, split_status = select_cases(cases, args.split)
    if not selected_cases:
        report = {
            "schema_version": "business-graph-evaluation/v1",
            "release_decision": "UNMEASURED",
            "case_count": 0,
            "split": args.split,
            "split_status": split_status,
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "report.md").write_text("# Business Graph Gold Evaluation\n\n`UNMEASURED`: " + split_status["reason"] + "\n", encoding="utf-8")
        return 2

    expected_source_hash = str(manifest["source_db_sha256"])
    expected_graph_hash = str(manifest["graph_db_sha256"])
    before = artifact_snapshot(source_db, graph_db, chroma_dir)
    graph_provenance = validate_graph_provenance(
        manifest, source_db=source_db, graph_db=graph_db
    )
    if before["source_db"]["sha256"] != expected_source_hash:
        raise ValueError("source DB hash does not match Gold manifest")
    if before["graph_db"]["sha256"] != expected_graph_hash and not args.allow_candidate_graph:
        raise ValueError("graph DB hash does not match Gold manifest")
    if args.allow_candidate_graph and not (
        graph_provenance["ready_manifest"]
        and graph_provenance["source_hash_matches"]
        and graph_provenance["topology_tables_present"]
    ):
        raise ValueError("candidate graph must have exactly one READY manifest, source binding, and graph tables")
    runtime_config = BusinessGraphRuntimeConfig(
        source_sha256=expected_source_hash,
        model_reranker_available=False,
        generator_available=False,
    )
    # The evaluator must exercise the same two-layer router as the business
    # CLI.  Missing INTENT_MODEL is an honest unmeasured configuration; any
    # configured-but-invalid environment is allowed to raise from the strict
    # adapter and fail the run before cases are evaluated.
    intent_classifier = load_intent_classifier_from_env(args.intent_model)
    embedding_retriever = None
    if args.embedding_mode == "isolated-copy":
        from src.business_graph_embedding import ChromaCardEmbeddingRetriever

        embedding_retriever = ChromaCardEmbeddingRetriever(
            chroma_dir,
            embedding_backend=args.embedding_backend,
        )
    runtime = BusinessGraphPipeline(
        source_db,
        graph_db,
        chroma_dir,
        runtime_config,
        embedding_retriever=embedding_retriever,
        intent_classifier=intent_classifier,
    )
    try:
        measured = evaluate_dataset(selected_cases, runtime)
    finally:
        runtime.close()
    after = artifact_snapshot(source_db, graph_db, chroma_dir)
    artifacts_unchanged = before == after
    for item in measured["cases"]:
        item["evaluation"]["checks"]["source_artifact_unchanged"] = artifacts_unchanged
        item["evaluation"]["measurement_status"] = (
            "PASS"
            if artifacts_unchanged and all(item["evaluation"]["checks"].values())
            else "FAIL"
        )
        if not artifacts_unchanged:
            reasons = item["evaluation"].setdefault("reason_codes", [])
            if "DATA_ARTIFACT_MUTATED" not in reasons:
                reasons.append("DATA_ARTIFACT_MUTATED")
    measured["metrics"]["source_artifact_unchanged"] = 1.0 if artifacts_unchanged else 0.0
    measured["hard_gates_passed"] = all(value == 1.0 for value in measured["metrics"].values())
    measured["measurement_status"] = (
        "PASS" if measured["hard_gates_passed"] else "FAIL"
    )

    category_counts = Counter(case["category"] for case in selected_cases)
    case_by_id = {case["case_id"]: case for case in selected_cases}
    reason_counts = Counter(
        reason
        for item in measured["cases"]
        for reason in item["evaluation"].get("reason_codes", [])
    )
    failure_by_category = Counter()
    failure_by_intent = Counter()
    minimal_reproducers: dict[str, dict[str, Any]] = {}
    for item in measured["cases"]:
        evaluation = item["evaluation"]
        if not evaluation.get("first_failure_stage"):
            continue
        case = case_by_id[item["case_id"]]
        failure_by_category[case["category"]] += 1
        failure_by_intent[case["expected_router"]["intent"]] += 1
        minimal_reproducers[item["case_id"]] = {
            "query": case["query"],
            "first_failure_stage": evaluation["first_failure_stage"],
            "reason_codes": evaluation["reason_codes"],
        }
    external_models = {
        "intent_classifier": (
            {
                "status": "MEASURED",
                "model": intent_classifier.model,
                "source": "environment" if args.intent_model is None else "--intent-model",
            }
            if intent_classifier is not None
            else {
                "status": "UNMEASURED",
                "model": None,
                "reason": "INTENT_MODEL is not configured; ambiguous routes remain UNMEASURED.",
            }
        ),
        "qwen_reranker": {"status": "UNMEASURED", "reason": "Model reranker was not configured for deterministic run."},
        "generator": {"status": "UNMEASURED", "reason": "Generator was not configured for deterministic run."},
    }
    report = {
        **{key: value for key, value in measured.items() if key != "cases"},
        "release_decision": (
            "PASS"
            if measured["hard_gates_passed"]
            and graph_provenance["approval_status"] == "GOLD_APPROVED"
            and args.mode == "llm"
            and all(item["status"] == "MEASURED" for item in external_models.values())
            else "BLOCKED"
        ),
        "split": args.split,
        "split_status": split_status,
        "dataset": {"path": str(dataset_dir or gold_path), "source_gold_path": str(gold_path), "manifest_path": str(manifest_path), "manifest_sha256": manifest.get("gold_sha256"), "category_counts": dict(category_counts)},
        "artifacts_before": before,
        "artifacts_after": after,
        "artifacts_unchanged": artifacts_unchanged,
        "graph_provenance": graph_provenance,
        "candidate_graph_allowed": args.allow_candidate_graph,
        "embedding": {
            "mode": args.embedding_mode,
            "backend": args.embedding_backend if args.embedding_mode == "isolated-copy" else None,
            "source_open_policy": "isolated-copy-only",
        },
        "external_models": external_models,
        "mode": args.mode,
        "failure_buckets": {
            "reason_code_counts": dict(reason_counts),
            "first_failure_by_category": dict(failure_by_category),
            "first_failure_by_intent": dict(failure_by_intent),
            "split_status": split_status,
            "minimal_reproducers": minimal_reproducers,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    with (output / "cases.jsonl").open("x", encoding="utf-8") as handle:
        for item in measured["cases"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "release_decision": report["release_decision"], "case_count": report["case_count"], "first_failure_by_stage": report["first_failure_by_stage"]}, ensure_ascii=False))
    return 0 if report["release_decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
