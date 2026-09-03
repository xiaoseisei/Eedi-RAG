from __future__ import annotations

import json
from pathlib import Path


def test_sweep_selector_prefers_recall_then_smaller_cutoff(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_bm25_dense_sweep as sweep

    def fake_manifest(recall):
        metrics = {
            "graded_recall_at_5": {"mean": recall}, "graded_recall_at_10": {"mean": recall},
            "graded_recall_at_15": {"mean": recall}, "graded_recall_at_20": {"mean": recall},
            "card_graded_recall_at_5": {"mean": recall}, "card_graded_recall_at_10": {"mean": recall},
            "card_graded_recall_at_15": {"mean": recall}, "card_graded_recall_at_20": {"mean": recall},
            "turn_coverage_at_5": {"mean": recall}, "turn_coverage_at_10": {"mean": recall},
            "turn_coverage_at_15": {"mean": recall}, "turn_coverage_at_20": {"mean": recall},
            "graded_ndcg_at_5": {"mean": 0.8}, "graded_mrr": {"mean": 0.8}, "graded_precision_at_5": {"mean": 0.8},
        }
        return {"aggregates": metrics, "attribution": {}, "artifact_hash_before": "x", "artifact_hash_after": "x", "artifact_mutated": False}
    manifests = {5: fake_manifest(0.8), 10: fake_manifest(0.9)}
    monkeypatch.setattr(sweep, "hash_storage_artifacts", lambda db, chroma: "x")
    monkeypatch.setattr(sweep, "evaluate", lambda **kwargs: manifests[kwargs["top_k"]])
    result = sweep.run_sweep(
        db_path=tmp_path / "db",
        chroma_path=tmp_path / "chroma",
        golden_path=tmp_path / "golden",
        l1_manifest_path=tmp_path / "l1",
        split_manifest_path=tmp_path / "split",
        output_dir=tmp_path / "out",
        cutoffs=[5, 10],
        embedding_backend="deterministic",
        retrieval_mode="bm25_dense",
        bm25_weight=0.35,
    )
    assert result["selected_rerank_pool_size"] == 10
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["selected_rerank_pool_size"] == 10
