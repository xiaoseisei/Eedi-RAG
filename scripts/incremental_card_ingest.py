"""Incremental, cache-keyed semantic card ingestion.

This command deliberately separates three concerns:

* the production DuckDB is opened read-only and is the source of sessions;
* semantic card extraction is the only potentially expensive LLM operation;
* raw evidence indexing/binding is deterministic and happens for every
  selected session, including cache hits and ``audit-only`` runs.

The command only writes an isolated cache/staging directory.  It never opens
the production DuckDB in write mode and never creates or updates Chroma.
Promotion of the generated cards remains an explicit, separate operation.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models import (
    CleanedSession,
    DialogueTurn,
    ExtractedPIU,
    Question,
    SubjectHierarchy,
)

logger = logging.getLogger(__name__)

IngestionMode = Literal["changed-only", "audit-only", "full"]
DEFAULT_PROMPT_VERSION = "evidence-completeness-v2"
DEFAULT_SCHEMA_VERSION = "extracted-piu-v1"
DEFAULT_BINDING_POLICY = "deterministic_role_non_noise_v1"
DEFAULT_INDEX_VERSION = "evidence-index-v1"

try:  # The modules are independently versioned and may be unavailable in old checkouts.
    from src.extraction_cache import ExtractionCache, compute_session_hash
except ImportError:  # pragma: no cover - exercised only by an incomplete deployment.
    ExtractionCache = None  # type: ignore[assignment,misc]
    compute_session_hash = None  # type: ignore[assignment]

try:
    from src.evidence_index import EVIDENCE_BINDING_POLICY as _INDEX_BINDING_POLICY
except ImportError:  # pragma: no cover - old checkout without provenance constants.
    _INDEX_BINDING_POLICY = None
try:
    from src.evidence_index import EVIDENCE_INDEX_VERSION as _INDEX_VERSION
except ImportError:  # pragma: no cover - old checkout without provenance constants.
    _INDEX_VERSION = None
try:
    from src.evidence_index import bind_card_to_evidence_index
except ImportError:  # pragma: no cover - surfaced at execution time.
    bind_card_to_evidence_index = None
try:
    from src.evidence_index import build_evidence_index
except ImportError:  # pragma: no cover - surfaced at execution time.
    build_evidence_index = None
try:
    from src.evidence_index import build_logical_evidence_index
except ImportError:  # pragma: no cover - old checkout without alias.
    build_logical_evidence_index = None

try:
    from src.extract_knowledge import extract_knowledge_from_session
except ImportError:  # pragma: no cover - import errors are surfaced at execution time.
    extract_knowledge_from_session = None
try:
    from src.extract_knowledge import extract_semantic_cards_from_session
except ImportError:  # pragma: no cover - old checkout without semantic-only API.
    extract_semantic_cards_from_session = None


@dataclasses.dataclass(frozen=True)
class IncrementalRunManifest:
    """Typed shape of the JSON manifest returned by the runner."""

    status: str
    mode: str
    generated_at: str
    source_session_count: int
    indexed_session_count: int
    reused_count: int
    llm_processed_count: int
    audit_count: int
    failed_count: int
    session_statuses: dict[str, dict[str, Any]]
    source_hashes: dict[str, str]
    contract: dict[str, Any]
    binding_contract: dict[str, Any]
    errors: list[dict[str, Any]]
    manifest_path: Optional[str] = None
    cards_path: Optional[str] = None
    evidence_index_path: Optional[str] = None
    reproduction_command: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class _LookupKey:
    """Adapter key used by injected one-argument test caches."""

    session_id: int
    session_hash: str
    contract_hash: str
    prompt_version: str


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _fallback_session_hash(session: CleanedSession) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(session).encode("utf-8")).hexdigest()


def _session_hash(session: CleanedSession) -> str:
    if compute_session_hash is not None:
        return str(compute_session_hash(session))
    return _fallback_session_hash(session)


def _contract_hash(contract: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(dict(contract)).encode("utf-8")).hexdigest()


def _call_with_supported_kwargs(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call an injectable implementation without hiding its real failures."""

    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(*args, **kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in parameters}
    return function(*args, **accepted)


def _cache_get(
    cache: Any,
    session: CleanedSession,
    contract: Mapping[str, Any],
    lookup_key: _LookupKey,
) -> Any:
    getter = getattr(cache, "get", None)
    if not callable(getter):
        raise TypeError("Extraction cache must implement get(session, contract) or get(key)")
    try:
        parameters = inspect.signature(getter).parameters
    except (TypeError, ValueError):
        parameters = {"session": None, "contract": None}
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return getter(session, contract)
    return getter(lookup_key)


def _cache_put_success(
    cache: Any,
    session: CleanedSession,
    contract: Mapping[str, Any],
    lookup_key: _LookupKey,
    extracted: ExtractedPIU,
    evidence_metadata: Mapping[str, Any],
) -> None:
    custom_put = getattr(cache, "put_success", None)
    if callable(custom_put):
        _call_with_supported_kwargs(custom_put, lookup_key, extracted, dict(evidence_metadata))
        return
    put = getattr(cache, "put", None)
    if not callable(put):
        raise TypeError("Extraction cache must implement put(session, contract, result)")
    # The production cache's put signature is intentionally simple.  Extra
    # evidence provenance belongs in the staging card/manifest, not its key.
    put(session, contract, extracted)


def _cache_put_failure(
    cache: Any,
    session: CleanedSession,
    contract: Mapping[str, Any],
    lookup_key: _LookupKey,
    error_type: str,
    error_message: str,
) -> None:
    """Record failure when supported, while keeping failure explicit otherwise."""

    put_failure = getattr(cache, "put_failure", None)
    if callable(put_failure):
        try:
            parameters = inspect.signature(put_failure).parameters
            positional = [
                parameter
                for parameter in parameters.values()
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            positional = []
        if len(positional) >= 4:
            put_failure(session, contract, error_type, error_message)
        else:
            _call_with_supported_kwargs(put_failure, lookup_key, error_type, error_message)
        return
    # ExtractionCache is success-only by design.  Do not write a fake result.
    logger.warning(
        "Session %s failure is retained in the run manifest; success-only cache has no failure record API: %s",
        session.intervention_id,
        error_message,
    )


def _unwrap_cached(value: Any) -> Optional[ExtractedPIU]:
    if value is None:
        return None
    if isinstance(value, ExtractedPIU):
        return value
    for attribute in ("extracted", "result", "value", "payload"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, ExtractedPIU):
            return candidate
    if isinstance(value, Mapping):
        for key in ("extracted", "result", "value", "payload"):
            candidate = value.get(key)
            if isinstance(candidate, ExtractedPIU):
                return candidate
            if isinstance(candidate, Mapping):
                try:
                    return ExtractedPIU.model_validate(candidate)
                except Exception as exc:
                    logger.warning("缓存载荷校验失败，按 cache miss 处理: %s", exc)
                    return None
    return None


def _ensure_success(extracted: Any, session: CleanedSession) -> ExtractedPIU:
    if not isinstance(extracted, ExtractedPIU):
        try:
            extracted = ExtractedPIU.model_validate(extracted)
        except Exception as exc:
            raise ValueError(f"Session {session.intervention_id} 抽取结果不是有效 ExtractedPIU: {exc}") from exc
    if (
        extracted.extraction_status != "success"
        or extracted.misconception is None
        or extracted.tutor_strategy is None
    ):
        raise ValueError(f"Session {session.intervention_id} 抽取结果不是完整 success 卡片")
    if extracted.session_id != session.intervention_id or extracted.question_id != session.question_id:
        raise ValueError(f"Session {session.intervention_id} 抽取结果外键不一致")
    return extracted


def _index_metadata(index: Any, *, binding_policy: str, index_version: str) -> dict[str, Any]:
    windows = getattr(index, "windows", None)
    if windows is None:
        windows = getattr(index, "entries", None)
    windows = list(windows or [])
    source_index_ids = []
    for window in windows:
        identifier = getattr(window, "index_id", None) or getattr(window, "window_id", None)
        if identifier is None and isinstance(window, Mapping):
            identifier = window.get("index_id") or window.get("window_id")
        if identifier is not None:
            source_index_ids.append(str(identifier))
    index_hash = getattr(index, "evidence_index_hash", None) or getattr(index, "artifact_hash", None)
    if index_hash is None and isinstance(index, Mapping):
        index_hash = index.get("evidence_index_hash") or index.get("artifact_hash")
    version = getattr(index, "evidence_index_version", None) or getattr(index, "version", None)
    if version is None and isinstance(index, Mapping):
        version = index.get("evidence_index_version") or index.get("version")
    chain_id = getattr(index, "chain_id", None)
    if chain_id is None and isinstance(index, Mapping):
        chain_id = index.get("chain_id")
    return {
        "source_index_ids": source_index_ids,
        "chain_id": str(chain_id or ""),
        "evidence_binding_policy": binding_policy,
        "evidence_index_version": str(version or index_version),
        "evidence_index_hash": str(index_hash or ""),
    }


def _bind_default(extracted: ExtractedPIU, index: Any) -> ExtractedPIU:
    if bind_card_to_evidence_index is None:
        raise RuntimeError(
            "缺少 src.evidence_index.bind_card_to_evidence_index；不能在没有确定性原文索引时生成卡片"
        )
    if extracted.misconception is None or extracted.tutor_strategy is None:
        raise ValueError("只能绑定完整 ExtractedPIU")
    misconception = bind_card_to_evidence_index(extracted.misconception, index)
    strategy = bind_card_to_evidence_index(extracted.tutor_strategy, index)
    return extracted.model_copy(update={"misconception": misconception, "tutor_strategy": strategy})


def _bind_result(
    extracted: ExtractedPIU,
    index: Any,
    binder: Optional[Callable[..., Any]],
) -> ExtractedPIU:
    if binder is None:
        return _bind_default(extracted, index)
    bound = _call_with_supported_kwargs(binder, extracted, index)
    if isinstance(bound, ExtractedPIU):
        return bound
    try:
        return ExtractedPIU.model_validate(bound)
    except Exception as exc:
        raise ValueError(
            f"绑定器返回了无效 ExtractedPIU (Session {extracted.session_id}): {exc}"
        ) from exc


def _invoke_extractor(
    extractor: Callable[..., Any],
    session: CleanedSession,
    *,
    llm_client: Any,
    model: Optional[str],
    max_retries: int,
    temperature: float,
    prompt_version: str,
) -> ExtractedPIU:
    result = _call_with_supported_kwargs(
        extractor,
        session,
        llm_client=llm_client,
        model=model,
        max_retries=max_retries,
        temperature=temperature,
        prompt_version=prompt_version,
        semantic_only=True,
    )
    return _ensure_success(result, session)


def _load_sessions_from_db(db_path: Path) -> list[CleanedSession]:
    """Load authoritative sessions without ever opening DuckDB for writing."""

    if not db_path.exists():
        raise FileNotFoundError(f"生产 DB 不存在: {db_path}")
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 duckdb 依赖，无法只读加载生产会话") from exc

    connection = duckdb.connect(database=str(db_path.resolve(strict=True)), read_only=True)
    try:
        session_rows = connection.execute(
            """
            SELECT intervention_id, question_id, tutor_id, subject_path,
                   question_text, options_json, total_turns,
                   student_turn_count, tutor_turn_count, has_valid_tutoring
            FROM tutoring_sessions
            ORDER BY intervention_id
            """
        ).fetchall()
        turn_rows = connection.execute(
            """
            SELECT intervention_id, turn_id, speaker, is_tutor, text,
                   talk_moves, is_greeting_or_noise
            FROM session_dialogue_turns
            ORDER BY intervention_id, turn_id
            """
        ).fetchall()
    except Exception as exc:
        raise RuntimeError(f"只读加载生产 DB 失败: {db_path}: {exc}") from exc
    finally:
        connection.close()

    turns_by_session: dict[int, list[DialogueTurn]] = {}
    for session_id, turn_id, speaker, is_tutor, text, talk_moves, is_noise in turn_rows:
        moves: list[str]
        if talk_moves is None:
            moves = []
        elif isinstance(talk_moves, str):
            try:
                parsed = json.loads(talk_moves)
                moves = list(parsed) if isinstance(parsed, list) else [talk_moves]
            except json.JSONDecodeError:
                moves = [talk_moves]
        else:
            moves = list(talk_moves)
        text_value = str(text)
        turns_by_session.setdefault(int(session_id), []).append(
            DialogueTurn(
                turn_id=int(turn_id),
                speaker=str(speaker),
                is_tutor=bool(is_tutor),
                raw_messages=[text_value],
                text=text_value,
                talk_moves=moves,
                is_greeting_or_noise=bool(is_noise),
            )
        )

    sessions: list[CleanedSession] = []
    for row in session_rows:
        (
            session_id,
            question_id,
            tutor_id,
            subject_path,
            question_text,
            options_json,
            total_turns,
            student_turn_count,
            tutor_turn_count,
            has_valid_tutoring,
        ) = row
        try:
            options = json.loads(options_json) if options_json else {}
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Session {session_id} options_json 非法: {exc}") from exc
        if not isinstance(options, dict):
            raise ValueError(f"Session {session_id} options_json 必须是对象")
        turns = turns_by_session.get(int(session_id), [])
        sessions.append(
            CleanedSession(
                intervention_id=int(session_id),
                question_id=int(question_id),
                tutor_id=int(tutor_id),
                subjects=SubjectHierarchy(paths=[str(subject_path)] if subject_path else ["通用数学考点"]),
                question=Question(
                    question_id=int(question_id),
                    question_text=str(question_text or ""),
                    options={str(key): str(value) for key, value in options.items()},
                ),
                turns=turns,
                total_turns=int(total_turns or len(turns)),
                student_turn_count=int(student_turn_count or sum(not turn.is_tutor for turn in turns)),
                tutor_turn_count=int(tutor_turn_count or sum(turn.is_tutor for turn in turns)),
                has_valid_tutoring=bool(has_valid_tutoring),
            )
        )
    return sessions


def _select_sessions(
    sessions: Sequence[CleanedSession], session_ids: Optional[Sequence[int]]
) -> list[CleanedSession]:
    by_id = {session.intervention_id: session for session in sessions}
    if len(by_id) != len(sessions):
        raise ValueError("输入会话包含重复 intervention_id")
    if session_ids is None:
        return list(sessions)
    requested = [int(session_id) for session_id in session_ids]
    if len(set(requested)) != len(requested):
        raise ValueError("session_ids 不得重复")
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"请求的 session_ids 不存在于只读数据源: {missing}")
    return [by_id[session_id] for session_id in requested]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_write_cards(path: Path, extracted: Iterable[ExtractedPIU]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            for item in extracted:
                handle.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)


def _atomic_write_evidence_indexes(path: Path, indexes: Mapping[int, Any]) -> None:
    """Persist deterministic raw evidence indexes separately from card output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            for session_id in sorted(indexes):
                index = indexes[session_id]
                payload = index.model_dump(mode="json") if hasattr(index, "model_dump") else index
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def _default_extractor() -> Callable[..., Any]:
    if extract_semantic_cards_from_session is not None:
        return extract_semantic_cards_from_session
    if extract_knowledge_from_session is not None:
        logger.warning(
            "当前 src.extract_knowledge 缺少 semantic-only API，将使用兼容抽取函数；最终 source_turn_ids 仍由确定性 binder 覆盖"
        )
        return extract_knowledge_from_session
    raise RuntimeError(
        "缺少 src.extract_knowledge.extract_semantic_cards_from_session；拒绝在没有真实语义抽取器时生成卡片"
    )


def _default_index_builder() -> Callable[..., Any]:
    if build_logical_evidence_index is not None:
        return build_logical_evidence_index
    if build_evidence_index is None:
        raise RuntimeError(
            "缺少 src.evidence_index.build_evidence_index；拒绝在没有确定性原文索引时生成卡片"
        )
    return build_evidence_index


def run_incremental_ingestion(
    *,
    db_path: Path | str = Path("data/db/tutoring_knowledge.duckdb"),
    cache_dir: Path | str = Path("reports/staging/incremental-cards"),
    mode: IngestionMode = "changed-only",
    session_ids: Optional[Sequence[int]] = None,
    workers: int = 1,
    model: Optional[str] = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    temperature: float = 0.1,
    max_retries: int = 3,
    binding_policy: Optional[str] = None,
    index_version: Optional[str] = None,
    llm_client: Any = None,
    sessions: Optional[Sequence[CleanedSession]] = None,
    cache: Any = None,
    extractor: Optional[Callable[..., Any]] = None,
    evidence_index_builder: Optional[Callable[..., Any]] = None,
    binder: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Run a resumable semantic-card staging pass.

    ``sessions``, ``cache``, ``extractor``, ``evidence_index_builder`` and
    ``binder`` are explicit injection points for local tests and controlled
    staging.  In production callers should omit them, causing sessions to be
    read from the production DB in read-only mode and the versioned project
    implementations to be selected.
    """

    if mode not in {"changed-only", "audit-only", "full"}:
        raise ValueError(f"mode 必须是 changed-only、audit-only 或 full，收到: {mode!r}")
    if workers < 1:
        raise ValueError("workers 必须为正整数")
    if not prompt_version.strip() or not schema_version.strip():
        raise ValueError("prompt_version 和 schema_version 不得为空")
    if temperature < 0:
        raise ValueError("temperature 不得为负数")
    if max_retries < 1:
        raise ValueError("max_retries 必须至少为 1")

    selected = _select_sessions(
        list(sessions) if sessions is not None else _load_sessions_from_db(Path(db_path)),
        session_ids,
    )
    if not selected:
        raise ValueError("没有可处理的会话，拒绝生成空增量产物")
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    if cache is None:
        if ExtractionCache is None:
            raise RuntimeError("缺少 src.extraction_cache.ExtractionCache，无法执行可恢复增量摄取")
        cache = ExtractionCache(cache_root / "extractions.json")

    effective_binding_policy = binding_policy or _INDEX_BINDING_POLICY or DEFAULT_BINDING_POLICY
    effective_index_version = index_version or _INDEX_VERSION or DEFAULT_INDEX_VERSION
    effective_model = model or os.environ.get("LLM_MODEL") or ""
    # Semantic cache key: only fields that can change LLM-produced card
    # meaning belong here.  Evidence binding/index versions are deliberately
    # separate so an index-policy change never re-runs semantic extraction.
    contract: dict[str, Any] = {
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "model": effective_model,
        "temperature": temperature,
        "max_retries": max_retries,
        "semantic_only": True,
    }
    binding_contract = {
        "evidence_binding_policy": effective_binding_policy,
        "evidence_index_version": effective_index_version,
    }

    index_builder = evidence_index_builder or _default_index_builder()
    selected_index: dict[int, Any] = {}
    source_hashes: dict[str, str] = {}
    statuses: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    # Build all indexes before cache lookup.  This makes evidence provenance
    # available even for reused cards and ensures audit-only remains deterministic.
    for session in selected:
        sid = str(session.intervention_id)
        source_hashes[sid] = _session_hash(session)
        try:
            selected_index[session.intervention_id] = _call_with_supported_kwargs(
                index_builder,
                session,
                window_size=6,
                step=3,
                evidence_index_version=effective_index_version,
                version=effective_index_version,
            )
        except Exception as exc:
            message = f"evidence index build failed: {exc}"
            statuses[sid] = {"status": "FAILED", "error_type": type(exc).__name__, "error": message}
            errors.append({"session_id": session.intervention_id, "stage": "evidence_index", "error": message})

    effective_extractor = extractor
    if effective_extractor is None and mode != "audit-only":
        effective_extractor = _default_extractor()

    # Cache entries are looked up with the same session/contract pair used by
    # src.extraction_cache.  Injected one-argument caches get a stable key.
    lookup: dict[int, tuple[_LookupKey, Any]] = {}
    pending: list[CleanedSession] = []
    extracted_by_session: dict[int, ExtractedPIU] = {}
    counts = {"reused": 0, "llm": 0, "audit": 0, "failed": len(errors)}

    for session in selected:
        sid = str(session.intervention_id)
        if session.intervention_id not in selected_index:
            continue
        key = _LookupKey(
            session_id=session.intervention_id,
            session_hash=source_hashes[sid],
            contract_hash=_contract_hash(contract),
            prompt_version=prompt_version,
        )
        lookup[session.intervention_id] = (key, selected_index[session.intervention_id])
        cached: Optional[ExtractedPIU] = None
        if mode != "full":
            try:
                cached = _unwrap_cached(_cache_get(cache, session, contract, key))
            except Exception as exc:
                logger.warning("Session %s cache lookup failed; treating as miss: %s", session.intervention_id, exc)
        if cached is not None:
            try:
                bound = _bind_result(cached, selected_index[session.intervention_id], binder)
                bound = _ensure_success(bound, session)
                extracted_by_session[session.intervention_id] = bound
                evidence_metadata = _index_metadata(
                    selected_index[session.intervention_id],
                    binding_policy=effective_binding_policy,
                    index_version=effective_index_version,
                )
                status = "AUDITED" if mode == "audit-only" else "REUSED"
                if mode == "audit-only":
                    counts["audit"] += 1
                else:
                    counts["reused"] += 1
                statuses[sid] = {
                    "status": status,
                    "cache_hit": True,
                    "source_hash": source_hashes[sid],
                    **evidence_metadata,
                }
            except Exception as exc:
                message = f"cached card binding failed: {exc}"
                counts["failed"] += 1
                statuses[sid] = {"status": "FAILED", "cache_hit": True, "error_type": type(exc).__name__, "error": message}
                errors.append({"session_id": session.intervention_id, "stage": "binding", "error": message})
            continue
        if mode == "audit-only":
            message = "audit-only requires a matching successful semantic cache entry"
            counts["failed"] += 1
            statuses[sid] = {"status": "FAILED", "cache_hit": False, "error_type": "CacheMiss", "error": message}
            errors.append({"session_id": session.intervention_id, "stage": "cache", "error": message})
            continue
        pending.append(session)

    if pending and effective_extractor is None:
        raise RuntimeError("changed-only/full 有缓存 miss，但未提供真实语义抽取器")

    def process_one(
        session: CleanedSession,
    ) -> tuple[CleanedSession, ExtractedPIU, ExtractedPIU, dict[str, Any]]:
        assert effective_extractor is not None
        semantic = _invoke_extractor(
            effective_extractor,
            session,
            llm_client=llm_client,
            model=model,
            max_retries=max_retries,
            temperature=temperature,
            prompt_version=prompt_version,
        )
        bound = _bind_result(semantic, selected_index[session.intervention_id], binder)
        bound = _ensure_success(bound, session)
        evidence_metadata = _index_metadata(
            selected_index[session.intervention_id],
            binding_policy=effective_binding_policy,
            index_version=effective_index_version,
        )
        # Cache the semantic result before deterministic evidence binding.  This
        # keeps evidence-index policy changes from requiring a semantic LLM call
        # and makes the separation observable in the cache contract.
        return session, semantic, bound, evidence_metadata

    counts["llm"] = len(pending)
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_by_session = {executor.submit(process_one, session): session for session in pending}
            for future in as_completed(future_by_session):
                session = future_by_session[future]
                sid = str(session.intervention_id)
                key, _index = lookup[session.intervention_id]
                try:
                    current, semantic, bound, evidence_metadata = future.result()
                    _cache_put_success(cache, current, contract, key, semantic, evidence_metadata)
                    extracted_by_session[current.intervention_id] = bound
                    statuses[sid] = {
                        "status": "EXTRACTED",
                        "cache_hit": False,
                        "source_hash": source_hashes[sid],
                        **evidence_metadata,
                    }
                except Exception as exc:
                    message = f"semantic extraction/binding failed: {exc}"
                    counts["failed"] += 1
                    statuses[sid] = {"status": "FAILED", "cache_hit": False, "error_type": type(exc).__name__, "error": message}
                    errors.append({"session_id": session.intervention_id, "stage": "semantic_extraction", "error": message})
                    try:
                        _cache_put_failure(cache, session, contract, key, type(exc).__name__, message)
                    except Exception as cache_exc:
                        errors.append({
                            "session_id": session.intervention_id,
                            "stage": "cache_failure_record",
                            "error": f"{type(cache_exc).__name__}: {cache_exc}",
                        })

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    cards_path = cache_root / f"cards-{run_id}.jsonl"
    evidence_index_path = cache_root / f"evidence-index-{run_id}.jsonl"
    _atomic_write_cards(cards_path, (extracted_by_session[sid] for sid in sorted(extracted_by_session)))
    _atomic_write_evidence_indexes(evidence_index_path, selected_index)
    status = "SUCCESS" if counts["failed"] == 0 else "FAILED"
    reproduction = (
        f"python -m scripts.incremental_card_ingest --mode {mode} --cache-dir {cache_root} "
        f"--workers {workers}"
    )
    if session_ids:
        reproduction += " --session-ids " + ",".join(str(session_id) for session_id in session_ids)
    manifest = IncrementalRunManifest(
        status=status,
        mode=mode,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source_session_count=len(selected),
        indexed_session_count=len(selected_index),
        reused_count=counts["reused"],
        llm_processed_count=counts["llm"],
        audit_count=counts["audit"],
        failed_count=counts["failed"],
        session_statuses=statuses,
        source_hashes=source_hashes,
        contract=contract,
        binding_contract=binding_contract,
        errors=errors,
        cards_path=str(cards_path),
        evidence_index_path=str(evidence_index_path),
        reproduction_command=reproduction,
    )
    manifest_dict = manifest.to_dict()
    manifest_path = cache_root / f"manifest-{run_id}.json"
    manifest_dict["manifest_path"] = str(manifest_path)
    _atomic_write_json(manifest_path, manifest_dict)
    _atomic_write_json(cache_root / "manifest.json", {**manifest_dict, "manifest_path": str(manifest_path)})
    return manifest_dict


def _parse_session_ids(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None or not raw.strip():
        return None
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"--session-ids 必须是逗号分隔整数: {raw!r}") from exc
    if not values:
        raise ValueError("--session-ids 不能为空")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental semantic-card staging without production writes")
    parser.add_argument("--db", type=Path, default=Path("data/db/tutoring_knowledge.duckdb"))
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/staging/incremental-cards"))
    parser.add_argument("--mode", choices=("changed-only", "audit-only", "full"), default="changed-only")
    parser.add_argument("--session-ids", default=None, help="逗号分隔的 intervention_id；未指定则处理只读 DB 全部会话")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_incremental_ingestion(
            db_path=args.db,
            cache_dir=args.cache_dir,
            mode=args.mode,
            session_ids=_parse_session_ids(args.session_ids),
            workers=args.workers,
            model=args.model,
            prompt_version=args.prompt_version,
            schema_version=args.schema_version,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )
    except Exception as exc:
        logger.error("增量摄取失败: %s", exc)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
