from __future__ import annotations

from pathlib import Path

import pytest

from src.business_models import BusinessQueryRequest, BusinessResponseStatus, QueryIntent
from src.cli import PedagogicalCLI
from src.intent_classifier import IntentClassification, OpenAICompatibleIntentClassifier


ROOT = Path(__file__).resolve().parents[1]


def test_cli_keeps_legacy_pipeline_as_the_default() -> None:
    cli = PedagogicalCLI()

    assert cli.pipeline_mode == "legacy"
    assert cli.business_pipeline is None


def test_cli_accepts_explicit_business_graph_configuration(tmp_path: Path) -> None:
    cli = PedagogicalCLI(
        pipeline_mode="business-graph",
        graph_db=tmp_path / "graph.duckdb",
        embedding_mode="isolated-copy",
        debug_graph=True,
    )

    assert cli.pipeline_mode == "business-graph"
    assert cli.graph_db == tmp_path / "graph.duckdb"
    assert cli.embedding_mode == "isolated-copy"
    assert cli.debug_graph is True


def test_cli_requires_explicit_opt_in_for_candidate_graph() -> None:
    cli = PedagogicalCLI(
        pipeline_mode="business-graph",
        graph_db=ROOT / "reports/staging/session-card-graph-v1/graph-concept-topology-20260909.duckdb",
    )

    with pytest.raises(ValueError, match="allow-candidate-graph"):
        cli._validate_business_graph_artifact()


def test_business_graph_cli_cold_start_routes_real_statuses_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise CLI assembly against approved local artifacts without exposing a key.

    The classifier is replaced only at the adapter boundary.  This keeps the
    test deterministic and proves that CLI initialization injects the second
    layer into the same pipeline that serves explicit business requests.
    """

    class FakeClassifier:
        model = "test-intent-model"

        def classify(self, query: str, candidates: tuple[QueryIntent, ...]) -> IntentClassification:
            assert query == "重考学生最常问什么？"
            assert QueryIntent.STUDENT_COHORT_PATTERNS in candidates
            assert QueryIntent.STUDENT_FREQUENT_QUESTIONS in candidates
            return IntentClassification(
                primary_intent=QueryIntent.STUDENT_FREQUENT_QUESTIONS,
                confidence=0.91,
                rationale="The requested output is a frequency summary of questions.",
            )

    fake = FakeClassifier()

    def fake_try_from_env(
        cls: type[OpenAICompatibleIntentClassifier],
        client: object = None,
        *,
        model: str | None = None,
        max_retries: int = 2,
    ) -> FakeClassifier:
        assert model is None
        return fake

    monkeypatch.setattr(
        OpenAICompatibleIntentClassifier,
        "try_from_env",
        classmethod(fake_try_from_env),
    )

    cli = PedagogicalCLI(
        db_path=ROOT / "data/db/tutoring_knowledge.duckdb",
        chroma_dir=ROOT / "data/chroma",
        pipeline_mode="business-graph",
        graph_db=ROOT / "reports/staging/session-card-graph-v1/graph-taxonomy-verify-2-20260908.duckdb",
        embedding_mode="disabled",
    )
    cli.initialize()
    try:
        assert cli.business_pipeline is not None
        assert cli.business_pipeline.router.llm_classifier is fake

        explicit = cli.business_pipeline.ask_business(
            BusinessQueryRequest(query="学生为什么会把 5.4598 四舍五入成 5.45？")
        )
        assert explicit.status is BusinessResponseStatus.SUCCESS
        assert explicit.evidence
        assert explicit.result is not None

        ambiguous = cli.business_pipeline.ask_business(
            BusinessQueryRequest(query="重考学生最常问什么？")
        )
        assert ambiguous.status is BusinessResponseStatus.SUCCESS
        assert ambiguous.understanding.route_source == "llm_classifier"

        unsupported = cli.business_pipeline.ask_business(
            BusinessQueryRequest(query="谁是平台里最优秀的导师？")
        )
        assert unsupported.status is BusinessResponseStatus.UNSUPPORTED
        assert unsupported.result is None
        assert unsupported.evidence == []
    finally:
        cli.business_pipeline.close()
        cli.business_pipeline = None
