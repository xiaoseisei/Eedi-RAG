from __future__ import annotations

"""Create a bounded human-review queue from provisional graded qrels."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_review_queue(*, dataset_dir: Path, output_dir: Path, top_k: int = 20) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite review queue: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dataset_dir / "raw_cases.jsonl"
    window_qrels_path = dataset_dir / "graded_qrels.jsonl"
    card_qrels_path = dataset_dir / "card_graded_qrels.jsonl"
    cases = _read_jsonl(raw_path)
    qrels_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for kind, path in (("window", window_qrels_path), ("card", card_qrels_path)):
        for row in _read_jsonl(path):
            qrels_by_key[(kind, row["query_id"], row["document_id"])] = row

    queue: list[dict[str, Any]] = []
    for case in cases:
        selected: dict[tuple[str, str], int | None] = {}
        for document_id in case["ranked_ids"][:top_k]:
            selected[("window", document_id)] = case["ranked_ids"].index(document_id) + 1
        for document_id in case["card_ranked_ids"][:top_k]:
            selected[("card", document_id)] = case["card_ranked_ids"].index(document_id) + 1
        for kind in ("window", "card"):
            for key, qrel in qrels_by_key.items():
                source_kind, query_id, document_id = key
                if source_kind == kind and query_id == case["query_id"] and qrel["relevance"] >= 2:
                    selected.setdefault((kind, document_id), None)
        for (kind, document_id), rank in sorted(selected.items()):
            qrel = qrels_by_key[(kind, case["query_id"], document_id)]
            relevance = int(qrel["relevance"])
            queue.append({
                "review_id": f"{case['query_id']}:{kind}:{document_id}",
                "query_id": case["query_id"],
                "case_id": case["case_id"],
                "document_type": kind,
                "document_id": document_id,
                "retrieval_rank": rank,
                "query": case["query"],
                "required_turns": case["required_turns"],
                "provisional_relevance": relevance,
                "provisional_rationale": qrel["rationale"],
                "provisional_evidence_turns": qrel["evidence_turns"],
                "review_status": "PENDING",
                "human_relevance": None,
                "review_priority": "HIGH" if relevance >= 2 or (rank is not None and rank <= 5) else "MEDIUM",
                "label_source": qrel["label_source"],
                "needs_human_review": True,
            })
    queue.sort(key=lambda item: (item["query_id"], item["document_type"], item["retrieval_rank"] is None, item["retrieval_rank"] or 10**9, item["document_id"]))
    queue_path = output_dir / "review_queue.jsonl"
    queue_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in queue), encoding="utf-8")
    summary = {
        "schema_version": "graded-qrels-review-queue/v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_dataset_dir": str(dataset_dir.resolve()),
        "source_files": {
            name: _sha256(dataset_dir / name)
            for name in ("raw_cases.jsonl", "graded_qrels.jsonl", "card_graded_qrels.jsonl", "manifest.json")
        },
        "review_queue_file": {"path": str(queue_path.resolve()), "sha256": _sha256(queue_path)},
        "row_count": len(queue),
        "by_document_type": {kind: sum(row["document_type"] == kind for row in queue) for kind in ("window", "card")},
        "by_priority": {priority: sum(row["review_priority"] == priority for row in queue) for priority in ("HIGH", "MEDIUM")},
        "rule": "Queue includes every Top-K ranked candidate and all provisional relevance>=2 documents; all labels remain PENDING until human review.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    summary = build_review_queue(dataset_dir=args.dataset_dir.resolve(strict=True), output_dir=args.output_dir.resolve(), top_k=args.top_k)
    print(json.dumps({"output_dir": str(args.output_dir), "row_count": summary["row_count"], "by_priority": summary["by_priority"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
