from __future__ import annotations

from pathlib import Path

from src.business_graph_pipeline import BusinessGraphPipeline, BusinessGraphRuntimeConfig
from src.business_graph_embedding import ChromaCardEmbeddingRetriever
from src.business_models import EvidencePerspective


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = ROOT / "data/db/tutoring_knowledge.duckdb"
GRAPH_DB = ROOT / "reports/staging/session-card-graph-v1/graph-taxonomy-verify-2-20260908.duckdb"
SOURCE_SHA = "f089c56f162666ab71b3beecef33a871f3e140608721b3f61f96f9ed384b6157"


class _FakeEmbeddingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve_card_hits(
        self,
        query: str,
        perspectives: list[EvidencePerspective],
        top_k: int,
    ) -> list[dict[str, object]]:
        self.calls.append({"query": query, "perspectives": perspectives, "top_k": top_k})
        return [
            {
                "chunk_id": "session_10_misconception",
                "collection": "student_misconceptions",
                "embedding_rank": 1,
                "embedding_score": 0.91,
            }
        ]


def test_graph_pipeline_records_dense_card_channel_without_making_it_fact_source() -> None:
    dense = _FakeEmbeddingRetriever()
    pipeline = BusinessGraphPipeline(
        source_db=SOURCE_DB,
        graph_db=GRAPH_DB,
        chroma_dir=ROOT / "data/chroma",
        runtime_config=BusinessGraphRuntimeConfig(source_sha256=SOURCE_SHA),
        embedding_retriever=dense,
    )
    try:
        trace = pipeline.prepare_case("学生为什么会把 5.4598 四舍五入成 5.45？")
    finally:
        pipeline.close()

    assert dense.calls
    hit = next(item for item in trace.card_candidates if item["chunk_id"] == "session_10_misconception")
    assert "dense" in hit["retrieval_channels"]
    assert hit["embedding_rank"] == 1
    assert hit["metadata"]["session_id"] == 10
    assert "vector_score" not in hit["metadata"]


def test_chroma_embedding_provider_uses_an_isolated_copy(tmp_path: Path) -> None:
    import chromadb

    source = tmp_path / "source-chroma"
    client = chromadb.PersistentClient(path=str(source))
    collection = client.get_or_create_collection("student_misconceptions")
    collection.add(ids=["session_1_misconception"], documents=["rounding"], embeddings=[[1.0] * 128])
    client.close()

    provider = ChromaCardEmbeddingRetriever(source)
    try:
        hits = provider.retrieve_card_hits(
            "rounding",
            [EvidencePerspective.STUDENT],
            top_k=1,
        )
        assert hits[0]["chunk_id"] == "session_1_misconception"
        assert provider.source_dir == source.resolve()
        assert provider.scratch_dir != provider.source_dir
    finally:
        provider.close()

    assert source.exists()
    assert (source / "chroma.sqlite3").exists()
