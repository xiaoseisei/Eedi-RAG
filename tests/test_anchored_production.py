from __future__ import annotations

from src.rag_pipeline import EndToEndPedagogicalRAGPipeline
from src.reranker import PedagogicalGoldAssembler
from src.reranker_provider import RerankResult
from src.retriever import DualMetricRetriever
from src.cli import PedagogicalCLI


class _FakeReranker:
    def name(self) -> str:
        return "fake-qwen"

    def rerank(self, query: str, documents: list[str], *, top_n: int):
        return [
            RerankResult(index=index, relevance_score=1.0 - index * 0.1, document=documents[index])
            for index in range(len(documents))
        ]


class _FakeStorage:
    def __init__(self) -> None:
        self._turns = {
            10: [
                {"session_id": 10, "turn_id": 1, "speaker": "student", "text": "I chose 5.45"},
                {"session_id": 10, "turn_id": 2, "speaker": "tutor", "text": "Which digit decides?"},
                {"session_id": 10, "turn_id": 3, "speaker": "student", "text": "The hundredths digit"},
                {"session_id": 10, "turn_id": 4, "speaker": "tutor", "text": "Good, now round it."},
                {"session_id": 10, "turn_id": 5, "speaker": "student", "text": "5.5"},
                {"session_id": 10, "turn_id": 6, "speaker": "tutor", "text": "Exactly."},
                {"session_id": 10, "turn_id": 7, "speaker": "student", "text": "Thanks"},
            ]
        }

    def get_dialogue_turns(self, session_id: int, turn_ids=None):
        turns = self._turns[session_id]
        if turn_ids is None:
            return list(turns)
        selected = set(int(value) for value in turn_ids)
        return [turn for turn in turns if int(turn["turn_id"]) in selected]


def _card(chunk_id: str, slot: str, turns: list[int]) -> dict:
    metadata = {
        "session_id": 10,
        "source_turn_ids": turns,
        "subject_path": "Number > Rounding",
    }
    if slot == "misconception":
        metadata.update({"misconception_name": "place-value confusion", "deep_mechanism": "wrong digit"})
    else:
        metadata.update({"key_aha_question": "Which digit decides?", "pedagogical_goal": "round correctly"})
    return {"chunk_id": chunk_id, "document": f"{slot} card", "metadata": metadata}


def test_retriever_anchored_expansion_uses_parent_pointers_only() -> None:
    retriever = object.__new__(DualMetricRetriever)
    retriever.storage = _FakeStorage()
    candidates = [_card("c1", "misconception", [1])]
    parents = [_card("c1", "misconception", [1])]

    units = retriever.expand_anchored_logical_windows(candidates, parents, window_size=6, step=3)

    assert len(units) == 1
    assert all(unit["metadata"]["anchor_turn_ids"] for unit in units)
    assert all("parent_contexts" in unit["metadata"] for unit in units)
    assert all("5.45" in turn["text"] or turn["turn_id"] != 1 for unit in units for turn in unit["evidence_turns"])


def test_retriever_completes_windows_for_a_paired_primary_session() -> None:
    retriever = object.__new__(DualMetricRetriever)
    retriever.storage = _FakeStorage()
    candidates = [
        _card("primary-misconception", "misconception", [1]),
        _card("primary-strategy", "strategy", [3]),
    ]
    parents = list(candidates)

    units = retriever.expand_anchored_logical_windows(candidates, parents, window_size=6, step=3)

    assert len(units) == 2
    assert all(unit["metadata"]["session_completion"] is True for unit in units)
    assert {turn["turn_id"] for unit in units for turn in unit["evidence_turns"]} == set(range(1, 8))


def test_pipeline_runs_card_then_anchored_window_rerank() -> None:
    class FakeRetriever:
        retrieval_mode = "bm25_dense"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def retrieve_multi_perspective_rrf(self, *, raw_query, top_k_each, fetch_evidence):
            return {
                "chunk_strategy": "card",
                "misconceptions": [_card("m1", "misconception", [1])],
                "strategies": [_card("s1", "strategy", [2])],
                "rewritten_queries": {"extracted_keywords": []},
            }

        def expand_anchored_logical_windows(self, candidates, parent_cards, *, window_size, step):
            self.calls.append(f"parents={len(parent_cards)}")
            return [
                {
                    "chunk_id": "w1",
                    "document": "[Turn 1] [student] I chose 5.45",
                    "metadata": {
                        "session_id": 10,
                        "window_start_turn": 1,
                        "window_end_turn": 1,
                        "source_turn_ids": [1],
                    },
                    "evidence_turns": [
                        {"session_id": 10, "turn_id": 1, "speaker": "student", "text": "I chose 5.45"}
                    ],
                }
            ]

    retriever = FakeRetriever()
    assembler = PedagogicalGoldAssembler(
        model_reranker=_FakeReranker(),
        rerank_unit="anchored_logical_window",
        evidence_selection_count=5,
        parent_card_count=3,
    )
    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=retriever,
        assembler=assembler,
        chunk_strategy="card",
    )

    response = pipeline.ask("学生为什么选 5.45？", mode="deterministic")

    assert retriever.calls == ["parents=2"]
    assert response.audit_status == "AUDITED_100_VERIFIED"
    assert [citation.turn_id for citation in response.dialogue_citations] == [1]


def test_anchored_reranker_receives_parent_metadata_and_window_document() -> None:
    class RecordingReranker(_FakeReranker):
        def __init__(self) -> None:
            self.documents: list[str] = []

        def rerank(self, query, documents, *, top_n):
            self.documents = list(documents)
            return super().rerank(query, documents, top_n=top_n)

    model = RecordingReranker()
    assembler = PedagogicalGoldAssembler(
        model_reranker=model,
        rerank_unit="anchored_logical_window",
    )
    candidate = {
        "chunk_id": "w1",
        "document": "[Turn 3] [student] 5.45",
        "metadata": {
            "session_id": 10,
            "source_turn_ids": [3],
            "parent_contexts": [
                {
                    "session_id": "10",
                    "subject_path": "Number",
                    "parent_title": "Rounding",
                    "parent_summary": "Use the next digit",
                }
            ],
        },
    }

    ranked = assembler.rerank_candidates("why?", [candidate], unit_name="anchored_logical_window")

    assert "Rounding" in model.documents[0]
    assert "Use the next digit" in model.documents[0]
    assert "[Turn 3] [student] 5.45" in model.documents[0]
    assert ranked[0]["reranker_unit"] == "anchored_logical_window"


def test_cli_real_reranker_defaults_to_anchored_parent3_w5() -> None:
    production_cli = PedagogicalCLI(
        embedding_backend="siliconflow",
        reranker_backend="siliconflow",
    )
    local_cli = PedagogicalCLI()

    assert production_cli.rerank_unit == "anchored_logical_window"
    assert production_cli.reranker_pool_size == 20
    assert production_cli.evidence_selection_count == 5
    assert production_cli.parent_card_count == 3
    assert local_cli.rerank_unit == "card"


def test_cli_generator_contract_v2_is_explicit() -> None:
    cli = PedagogicalCLI(generator_contract_version="v2")

    assert cli.generator_contract_version == "v2"


def test_cli_initialize_wires_generator_contract_to_pipeline(monkeypatch) -> None:
    captured: dict[str, dict] = {}

    class FakeStorage:
        def __init__(self, **kwargs):
            captured["storage"] = kwargs

    class FakeRetriever:
        def __init__(self, **kwargs):
            captured["retriever"] = kwargs

    class FakeAssembler:
        def __init__(self, **kwargs):
            captured["assembler"] = kwargs

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["pipeline"] = kwargs

    monkeypatch.setattr("src.cli.DualEngineStorageManager", FakeStorage)
    monkeypatch.setattr("src.cli.DualMetricRetriever", FakeRetriever)
    monkeypatch.setattr("src.cli.PedagogicalGoldAssembler", FakeAssembler)
    monkeypatch.setattr("src.cli.EndToEndPedagogicalRAGPipeline", FakePipeline)

    cli = PedagogicalCLI(generator_contract_version="v2")
    cli.initialize()

    assert "generator_contract_version" not in captured["assembler"]
    assert captured["pipeline"]["generator_contract_version"] == "v2"


def test_pipeline_prepare_context_exposes_anchored_route_without_generation() -> None:
    class PreparedRetriever:
        retrieval_mode = "bm25_dense"

        def retrieve_multi_perspective_rrf(self, *, raw_query, top_k_each, fetch_evidence):
            return {
                "chunk_strategy": "card",
                "misconceptions": [_card("m1", "misconception", [1])],
                "strategies": [_card("s1", "strategy", [2])],
                "rewritten_queries": {"extracted_keywords": []},
            }

        def expand_anchored_logical_windows(self, candidates, parent_cards, *, window_size, step):
            return [
                {
                    "chunk_id": "w1",
                    "document": "[Turn 1] [student] I chose 5.45",
                    "metadata": {
                        "session_id": 10,
                        "window_start_turn": 1,
                        "window_end_turn": 1,
                        "source_turn_ids": [1],
                        "parent_contexts": [],
                    },
                    "evidence_turns": [
                        {"session_id": 10, "turn_id": 1, "speaker": "student", "text": "I chose 5.45"}
                    ],
                }
            ]

    pipeline = EndToEndPedagogicalRAGPipeline(
        retriever=PreparedRetriever(),
        assembler=PedagogicalGoldAssembler(
            model_reranker=_FakeReranker(),
            rerank_unit="anchored_logical_window",
            parent_card_count=3,
            evidence_selection_count=5,
        ),
        chunk_strategy="card",
    )

    retrieval, context = pipeline.prepare_context("学生为什么选 5.45？", top_k_each=20)

    assert retrieval["trace"]["rerank_unit"] == "anchored_logical_window"
    assert retrieval["trace"]["parent_card_limit"] == 3
    assert context.rerank_unit == "anchored_logical_window"
    assert context.selected_evidence_units[0]["reranker_rank"] == 1
    assert [(turn["session_id"], turn["turn_id"]) for turn in context.evidence_turns] == [(10, 1)]
    assert len(context.deepeval_context_nodes) >= 2
    assert any(node.startswith("## [DERIVED_FACT]") for node in context.deepeval_context_nodes)
    assert all(node in context.prompt_context_markdown for node in context.deepeval_context_nodes)
