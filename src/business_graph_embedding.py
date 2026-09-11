"""Safe dense-card retrieval for the graph-business route.

The dense index is a candidate generator only.  Card metadata used by the
business route is re-bound to the source DuckDB before it can become evidence.
Chroma is opened only after the caller's directory has been copied to a
process-owned scratch directory; this is required because local Chroma clients
may apply migrations while opening an artifact.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import chromadb

from src.business_models import EvidencePerspective
from src.retriever import DualMetricRetriever
from src.storage_manager import FastDeterministicEmbeddingFunction


class EmbeddingRetrievalError(RuntimeError):
    """Dense retrieval could not produce a valid candidate batch."""


class _ChromaStorageView:
    """Small read-only storage shape required by ``DualMetricRetriever``."""

    def __init__(self, client: Any, embedding_function: Any, embedding_backend: str) -> None:
        self.chroma_client = client
        self.embedding_function = embedding_function
        self.embedding_backend = embedding_backend

    def query_collection(self, collection_name: str, kwargs: dict[str, Any]) -> Any:
        """Read a collection without passing a potentially conflicting EF."""

        return self.chroma_client.get_collection(collection_name).query(**kwargs)


class ChromaCardEmbeddingRetriever:
    """Run the existing Card dense/RRF retriever against an isolated Chroma copy."""

    def __init__(
        self,
        chroma_dir: str | Path,
        *,
        embedding_backend: str = "deterministic",
        retrieval_mode: str = "bm25_dense",
        bm25_weight: float = 0.35,
    ) -> None:
        self.source_dir = Path(chroma_dir).resolve(strict=True)
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"Chroma directory not found: {self.source_dir}")
        if not (self.source_dir / "chroma.sqlite3").is_file():
            raise EmbeddingRetrievalError(
                f"Chroma artifact is missing chroma.sqlite3: {self.source_dir}"
            )

        self._scratch_root = Path(tempfile.mkdtemp(prefix="eedi-rag-embedding-"))
        self.scratch_dir = self._scratch_root / "chroma"
        self._client: Any | None = None
        self._storage: _ChromaStorageView | None = None
        self._retriever: DualMetricRetriever | None = None
        try:
            shutil.copytree(self.source_dir, self.scratch_dir)
            if embedding_backend == "deterministic":
                embedding_function = FastDeterministicEmbeddingFunction()
                backend_name = embedding_function.name()
            elif embedding_backend == "siliconflow":
                from src.embedding_provider import SiliconFlowQwen3EmbeddingFunction

                embedding_function = SiliconFlowQwen3EmbeddingFunction.from_env()
                backend_name = embedding_function.name()
            else:
                raise ValueError(
                    "embedding_backend must be deterministic or siliconflow"
                )

            # ``get_collection`` deliberately does not pass an embedding
            # function.  Chroma persists the function identity in collection
            # configuration, and old indexes may legitimately use the
            # ``default`` configuration.  Retrieval supplies explicit query
            # embeddings below, so reopening the collection must not attempt
            # to replace that persisted identity.  Collection existence is
            # checked lazily for only the requested lane.
            self._client = chromadb.PersistentClient(path=str(self.scratch_dir))
            self._storage = _ChromaStorageView(
                self._client, embedding_function, backend_name
            )
            self._retriever = DualMetricRetriever(
                storage_manager=self._storage,
                query_rewrite_mode="deterministic",
                retrieval_mode=retrieval_mode,
                bm25_weight=bm25_weight,
            )
        except Exception as exc:
            client = self._client
            close_client = getattr(client, "close", None)
            if callable(close_client):
                close_client()
            self._client = None
            if self._scratch_root.exists():
                shutil.rmtree(self._scratch_root)
            raise EmbeddingRetrievalError(
                f"failed to open isolated Chroma embedding artifact: {self.source_dir}"
            ) from exc

    def retrieve_card_hits(
        self,
        query: str,
        perspectives: Sequence[EvidencePerspective],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return dense/RRF Card hits with ranking telemetry, never evidence."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-blank string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not perspectives:
            raise ValueError("at least one evidence perspective is required")
        if self._retriever is None:
            raise EmbeddingRetrievalError("embedding retriever is closed")

        lanes = tuple(
            "student" if perspective is EvidencePerspective.STUDENT else "tutor"
            for perspective in perspectives
        )
        required_collections = {
            "student": "student_misconceptions",
            "tutor": "tutor_strategies",
        }
        try:
            for lane in set(lanes):
                self._client.get_collection(required_collections[lane])
            result = self._retriever.retrieve_multi_perspective_rrf(
                query,
                top_k_each=top_k,
                fetch_evidence=False,
                perspectives=lanes,
            )
        except Exception as exc:
            raise EmbeddingRetrievalError(
                f"dense Card retrieval failed for requested lanes {lanes}"
            ) from exc
        hits: list[dict[str, Any]] = []
        lane_specs = (
            ("misconceptions", "MISCONCEPTION", "student_misconceptions"),
            ("strategies", "TUTOR_STRATEGY", "tutor_strategies"),
        )
        for result_key, card_type, collection in lane_specs:
            for rank, item in enumerate(result.get(result_key, []) or [], 1):
                raw_score = item.get("rrf_score", item.get("hybrid_score"))
                if raw_score is None:
                    raise EmbeddingRetrievalError(
                        f"dense Card retrieval returned an unscored hit for {collection}"
                    )
                hits.append(
                    {
                        "chunk_id": str(item["chunk_id"]),
                        "collection": collection,
                        "card_type": card_type,
                        "embedding_rank": rank,
                        "embedding_score": float(raw_score),
                        "retrieval_channels": ["dense"],
                    }
                )
        return hits

    def close(self) -> None:
        """Close Chroma before removing only this instance's scratch copy."""

        client = self._client
        close_client = getattr(client, "close", None)
        if callable(close_client):
            close_client()
        self._client = None
        self._storage = None
        self._retriever = None
        if self._scratch_root.exists():
            shutil.rmtree(self._scratch_root)

    def __enter__(self) -> "ChromaCardEmbeddingRetriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["ChromaCardEmbeddingRetriever", "EmbeddingRetrievalError"]
