"""Export the approved business Graph Gold into isolated evaluation views.

The exporter is a lossless projection of the human-authored Gold JSONL.  It
never reads runtime traces or generates expected values from the SUT.  The
original monolithic Gold remains the approval source of truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business_graph_gold import load_gold_cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def export_dataset(gold_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create lossless query/fact/path/evidence projections and a manifest."""

    source = Path(gold_path).resolve(strict=True)
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite dataset directory: {target}")
    cases = load_gold_cases(source)
    if not cases:
        raise ValueError("Gold dataset is empty")
    target.mkdir(parents=True, exist_ok=False)

    queries = [
        {"case_id": case["case_id"], "category": case["category"], "query": case["query"]}
        for case in cases
    ]
    facts = [
        {
            "case_id": case["case_id"],
            "expected_router": case["expected_router"],
            "expected_plan": case["expected_plan"],
            "expected_sql_scope": case["expected_sql_scope"],
        }
        for case in cases
    ]
    graph_paths = [
        {"case_id": case["case_id"], "expected_graph_paths": case["expected_graph_paths"]}
        for case in cases
    ]
    evidence = [
        {
            "case_id": case["case_id"],
            "expected_card_evidence": case["expected_card_evidence"],
            "expected_window_evidence": case["expected_window_evidence"],
            "expected_prompt": case["expected_prompt"],
            "expected_citation_contract": case["expected_citation_contract"],
            "expected_status": case["expected_status"],
            "ideal_answer": case["ideal_answer"],
        }
        for case in cases
    ]
    _write_jsonl(target / "queries.jsonl", queries)
    _write_jsonl(target / "expected_facts.jsonl", facts)
    _write_jsonl(target / "expected_graph_paths.jsonl", graph_paths)
    _write_jsonl(target / "expected_evidence.jsonl", evidence)

    categories = Counter(str(case["category"]) for case in cases)
    manifest = {
        "schema_version": "business-graph-dataset-split/v1",
        "artifact_status": "DERIVED_FROM_HUMAN_APPROVED_SOURCE",
        "source_gold_path": str(source),
        "source_gold_sha256": sha256(source),
        "case_count": len(cases),
        "case_ids": [str(case["case_id"]) for case in cases],
        "category_counts": dict(sorted(categories.items())),
        "split_status": {
            "status": "UNMEASURED",
            "reason": "The approved source Gold has no split field; no partition was invented.",
        },
        "sut_input_contract": "Only queries.jsonl query values may be passed to the SUT.",
        "expected_data_contract": "facts, graph_paths, and evidence are evaluator-only and must never be passed to the SUT.",
    }
    (target / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "README.md").write_text(
        """# Business Graph v1 split evaluation dataset\n\n"
        "This directory is a lossless projection of the approved source Gold.\n"
        "The source Gold and its human approval remain authoritative.\n\n"
        "- `queries.jsonl` is the only SUT input view.\n"
        "- `expected_facts.jsonl`, `expected_graph_paths.jsonl`, and\n"
        "  `expected_evidence.jsonl` are evaluator-only views.\n"
        "- `split_manifest.json` records source hash, case IDs, and the fact that\n"
        "  no dev/holdout split was invented.\n\n"
        "Never regenerate expected data from runtime output or mark this derived\n"
        "directory approved independently of the source Gold review.\n""",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold", type=Path, default=ROOT / "data/business/business-graph-gold-v1.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = export_dataset(args.gold, args.output_dir)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
