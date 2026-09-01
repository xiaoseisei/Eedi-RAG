import json
from pathlib import Path

import pytest

from scripts import expand_knowledge_base as expansion
from src.extract_knowledge import LLMExtractionError, LLMUnavailableError
from src.models import CleanedSession, ExtractedPIU


@pytest.fixture
def session() -> CleanedSession:
    path = Path("data/sample/cleaned_sessions_sample.jsonl")
    return CleanedSession.model_validate(json.loads(path.read_text(encoding="utf-8").splitlines()[0]))


def test_deterministic_card_fabrication_is_disabled(session: CleanedSession):
    with pytest.raises(LLMUnavailableError, match="禁用"):
        expansion.extract_piu_deterministic_high_fidelity(session)


def test_llm_expansion_extractor_uses_strict_shared_gate(session: CleanedSession):
    class BadCompletions:
        def create(self, **kwargs):
            message = type("Message", (), {"content": '{"misconception": {}, "tutor_strategy": {}}'})
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})]})

    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": BadCompletions()})},
    )()
    with pytest.raises(LLMExtractionError):
        expansion.extract_piu_with_llm(session, client, "test-model", max_retries=1)


def test_run_expansion_rejects_disabled_llm_before_sampling(monkeypatch):
    sampled = False

    def forbidden_sample():
        nonlocal sampled
        sampled = True
        return []

    monkeypatch.setattr(expansion, "select_stratified_100_sessions", forbidden_sample)
    with pytest.raises(LLMUnavailableError, match="LLM"):
        expansion.run_expansion(use_llm=False)
    assert sampled is False


def test_run_expansion_aborts_before_storage_on_any_extraction_failure(
    monkeypatch, session: CleanedSession
):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(expansion, "select_stratified_100_sessions", lambda: [session])
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: False
        if "extracted_pius_sample" in str(self)
        else original_exists(self),
    )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        expansion,
        "extract_piu_with_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(LLMExtractionError("bad extraction")),
    )
    storage_created = False

    class ForbiddenStorage:
        def __init__(self, **kwargs):
            nonlocal storage_created
            storage_created = True

    monkeypatch.setattr(expansion, "DualEngineStorageManager", ForbiddenStorage)
    with pytest.raises(LLMExtractionError, match="bad extraction"):
        expansion.run_expansion(limit=1)
    assert storage_created is False


def test_run_expansion_explicitly_declares_deterministic_embedding(
    monkeypatch, session: CleanedSession
):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(expansion, "select_stratified_100_sessions", lambda: [session])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    piu_path = Path("data/sample/extracted_pius_sample.jsonl")
    piu = ExtractedPIU.model_validate(
        json.loads(piu_path.read_text(encoding="utf-8").splitlines()[0])
    )
    monkeypatch.setattr(expansion, "extract_piu_with_llm", lambda *args, **kwargs: piu)
    storage_kwargs = {}

    class RecordingStorage:
        def __init__(self, **kwargs):
            storage_kwargs.update(kwargs)

        def ingest_all(self, **kwargs):
            pass

        def get_storage_audit_snapshot(self):
            return {"status": "test-only"}

        def close(self):
            pass

    monkeypatch.setattr(expansion, "DualEngineStorageManager", RecordingStorage)
    expansion.run_expansion(limit=1)
    assert storage_kwargs["embedding_backend"] == "deterministic"
